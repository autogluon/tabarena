"""Fit and predict one agentic MLP with the modules of ``bin.mlp``.

``_dataset`` and ``_train`` follow ``_build_dataset`` and ``train_and_eval`` in ``bin/mlp/pipeline.py``
(agentic-tabular 3d7349b) on arrays in memory instead of a dataset directory. The package predicts only the
rows present when its preprocessing is fit, so a prediction rebuilds the dataset with the new rows as the
``test`` part; the preprocessing is seeded, so the training part comes out identical.
"""

from __future__ import annotations

import copy
import logging
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from bin.mlp._util import is_oom_exception
from bin.mlp.core import Dataset, compute_metrics_for_parts, logits_to_probs, seed_everything, to_torch
from bin.mlp.hparams import resolve
from bin.mlp.pipeline import _build_model, infer_loss_output_dim, predict
from torch.nn.utils import parametrize

logger = logging.getLogger(__name__)


@contextmanager
def torch_threads(n_threads: int):
    """Use ``n_threads`` torch threads inside the block. TF32 stays at torch's defaults, as in the package."""
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(n_threads)
        yield
    finally:
        torch.set_num_threads(old_threads)


class TrainingGuard:
    """Keeps every configuration's fit inside TabArena's requirements without changing the package's train modules.

    - Time limit: the package's train modules run to their epoch budget, as in its standalone experiments. After each
      epoch's validation call, ``patience`` is set to 0 once the next epoch would end past the deadline.
    - Finite results: the guard keeps the weights of the epoch with the best validation score. A configuration whose
      loss or predictions leave the finite range ends training there (an early stop) instead of with the error the
      train modules raise; ``restore`` puts those weights back.
    """

    def __init__(self, model, inference_fn, deadline: float | None, epoch_parts):
        self.inference_fn = inference_fn
        self.deadline = deadline
        self.epoch_parts = epoch_parts
        self.start = time.monotonic()
        self.n_epochs = 0
        self.best_score = float("inf")
        self.best_state = copy.deepcopy(model.state_dict())

    def __call__(self, model, tensors, dataset, trial_params, device, parts):
        predictions = self.inference_fn(model, tensors, dataset, trial_params, device, parts)
        if tuple(parts) != self.epoch_parts:
            return predictions
        if all(np.isfinite(np.asarray(predictions[part])).all() for part in predictions):
            score = float(compute_metrics_for_parts(dataset, predictions, ("val",))["val"]["score"])
            if score < self.best_score:
                self.best_score = score
                self.best_state = copy.deepcopy(model.state_dict())
        self.n_epochs += 1
        now = time.monotonic()
        if self.deadline is not None and now + (now - self.start) / self.n_epochs >= self.deadline:
            trial_params["patience"] = 0
        return predictions

    def restore(self, model, err: Exception) -> None:
        """Load the best finite epoch's weights after a train module stopped on a non-finite loss or prediction."""
        if not any(s in str(err) for s in ("non-finite", "NaN", "infinity", "valid checkpoint")):
            raise err
        model.load_state_dict(self.best_state)


def bounded_inverse(artifacts: dict, y_train: np.ndarray) -> dict:
    """Clip regression outputs to the training targets' range, widened by half its width on each side, before the
    package's inverse target transform (an exp or expm1 of an extreme output would overflow).
    """
    if "inverse_transform" not in artifacts:
        return artifacts
    lo, hi = float(np.min(y_train)), float(np.max(y_train))
    pad = 0.5 * max(hi - lo, 1.0)
    inverse = artifacts["inverse_transform"]
    return {**artifacts, "inverse_transform": lambda values: inverse(np.clip(values, lo - pad, hi + pad))}


def finite_fallback(y_train: np.ndarray, problem_type: str, n_classes: int) -> np.ndarray:
    """What a prediction that is not finite is replaced with: the training mean (regression) or the training class
    frequencies (classification).
    """
    if problem_type == "regression":
        return np.array(float(np.mean(y_train)))
    counts = np.bincount(np.asarray(y_train).astype(np.int64), minlength=n_classes)[:n_classes].astype(np.float64)
    return (counts + 1.0) / (counts.sum() + n_classes)


def replace_non_finite(predictions: np.ndarray, fallback: np.ndarray, problem_type: str) -> np.ndarray:
    """Replace each row with a non-finite value by ``fallback`` (see ``finite_fallback``)."""
    predictions = np.array(predictions, dtype=np.float64)
    if problem_type == "regression":
        return np.where(np.isfinite(predictions), predictions, fallback)
    bad = ~np.isfinite(predictions).reshape(len(predictions), -1).all(axis=1)
    predictions[bad] = fallback[1] if predictions.ndim == 1 else fallback
    return predictions


class MLPImplementation:
    """One fitted agentic MLP (a single bag child)."""

    def __init__(
        self, *, problem_type: str, n_classes: int, device: str, n_threads: int, eval_metric, indices, **params
    ):
        self.problem_type = problem_type
        self.task_type = "binclass" if problem_type == "binary" else problem_type
        self.n_classes = {"regression": 1, "binary": 2}.get(problem_type, n_classes)
        self.device_ = torch.device(device)
        self.n_threads = n_threads
        if problem_type == "regression":
            self.score_name = "rmse"
        elif eval_metric.name == "accuracy":
            self.score_name = "accuracy"
        elif problem_type == "binary" and eval_metric.name == "roc_auc":
            self.score_name = "roc-auc"
        else:
            self.score_name = "cross-entropy"
        self.indices = indices
        self.seed = int(params.pop("random_state") or 0)
        self.params = params
        self.model_ = None

    def _split(self, X: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Float32 numerical columns with NaN kept, and the categorical columns as codes over the training
        categories, with unknown and missing values on the extra code ``len(categories)``.
        """
        x_num = X[self.num_cols_].to_numpy(dtype=np.float32) if self.num_cols_ else None
        x_cat = None
        if self.categories_:
            columns = []
            for col, cats in self.categories_.items():
                codes = cats.get_indexer(X[col])
                columns.append(np.where(codes < 0, len(cats), codes).astype(np.int64))
            x_cat = np.column_stack(columns)
        return x_num, x_cat

    def _parts(self, x_num_test, x_cat_test, y_test) -> tuple[dict | None, dict | None, dict]:
        x_num = None if self.x_num_fit_ is None else {**self.x_num_fit_, "test": x_num_test}
        x_cat = None if self.x_cat_fit_ is None else {**self.x_cat_fit_, "test": x_cat_test}
        return x_num, x_cat, {**self.y_fit_, "test": y_test}

    def _dummy_targets(self, size: int) -> np.ndarray:
        if self.task_type == "regression":
            return np.zeros(size, dtype=np.float32)
        if self.task_type == "binclass":
            # The train modules score the test part every epoch, and roc-auc needs both classes there.
            return np.resize(np.array([0, 1], dtype=np.int64), size)
        return np.arange(size, dtype=np.int64) % self.n_classes

    def _dataset(self, x_num, x_cat, y) -> Dataset:
        config = {"seed": self.seed, "y_train": y["train"], "n_classes": self.n_classes, "task_type": self.task_type}
        x_num, num_artifacts = self._stage("numerical_preprocess_idx")(x_num, config)
        x_cat, cat_artifacts = self._stage("categorical_preprocess_idx")(x_cat, config)
        y_prepared, target_artifacts = self._stage("target_preprocess_idx")(y, config)
        if self.task_type == "regression":
            target_artifacts = bounded_inverse(target_artifacts, y_prepared["train"])
        return Dataset(
            x_num=x_num,
            x_cat=x_cat,
            y=y_prepared,
            y_raw=y,
            task_type=self.task_type,
            score_name=self.score_name,
            n_classes=self.n_classes,
            preprocess_artifacts={"numerical": num_artifacts, "categorical": cat_artifacts, "target": target_artifacts},
        )

    def _stage(self, key: str):
        return resolve(key, self.indices[key], self.task_type)

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, time_limit: float | None):
        cat_cols = [col for col in X.columns if isinstance(X[col].dtype, pd.CategoricalDtype)]
        self.num_cols_ = [col for col in X.columns if col not in cat_cols]
        # A sliced Categorical can keep levels seen only outside this fold; drop them so they map to unknown.
        self.categories_ = {col: X[col].cat.remove_unused_categories().cat.categories for col in cat_cols}
        x_num, x_cat = self._split(X)
        x_num_val, x_cat_val = self._split(X_val)
        self.x_num_fit_ = None if x_num is None else {"train": x_num, "val": x_num_val}
        self.x_cat_fit_ = None if x_cat is None else {"train": x_cat, "val": x_cat_val}
        self.y_fit_ = {"train": np.asarray(y), "val": np.asarray(y_val)}

        # The pipeline needs a test part; two training rows stand in for it during the fit.
        rows = [0, min(1, len(y) - 1)]
        parts = self._parts(
            None if x_num is None else x_num[rows], None if x_cat is None else x_cat[rows], self._dummy_targets(2)
        )
        deadline = None if time_limit is None else time.monotonic() + max(0.0, 0.9 * time_limit)
        with torch_threads(self.n_threads):
            while True:
                try:
                    seed_everything(self.seed)
                    self.model_, self.trial_params_ = self._train(*parts, deadline)
                    break
                except Exception as err:
                    eval_batch_size = int(self.params["eval_batch_size"])
                    if not is_oom_exception(err) or eval_batch_size <= 1:
                        raise
                    logger.warning(f"Agentic MLP OOM with eval_batch_size={eval_batch_size}; retrying with half")
                    self.params["eval_batch_size"] = eval_batch_size // 2
                    if self.device_.type == "cuda":
                        torch.cuda.empty_cache()

        # A model that predicts non-finite values on the validation rows goes back to the guard's best epoch.
        self.fallback_ = finite_fallback(self.y_fit_["train"], self.problem_type, self.n_classes)
        if not np.isfinite(self._predict(X_val)).all():
            self.model_.load_state_dict(self.best_state_)
        del self.best_state_

        # torch cannot pickle a module with parametrizations (the spectral-norm MLP); keep the effective weights.
        for module in self.model_.modules():
            for name in list(getattr(module, "parametrizations", {})):
                parametrize.remove_parametrizations(module, name, leave_parametrized=True)

    def _train(self, x_num, x_cat, y, deadline: float | None) -> tuple[torch.nn.Module, dict]:
        trial_params = {**self.params, "seed": self.seed}
        dataset = self._dataset(x_num, x_cat, y)
        tensors = to_torch(dataset, self.device_)
        loss_fn = self._stage("loss_idx")
        output_dim = infer_loss_output_dim(loss_fn, dataset)
        model = _build_model(dataset, tensors, trial_params, self.indices, output_dim, self.device_)
        optimizer_builder = self._stage("optimizer_idx")
        train_fn = self._stage("train_idx")
        guard = TrainingGuard(model, self._stage("inference_idx"), deadline, ("val", "test"))
        patience = trial_params["patience"]
        try:
            train_fn(model, tensors, dataset, trial_params, guard, loss_fn, optimizer_builder, self.device_, self.seed)
        except (RuntimeError, ValueError) as err:
            guard.restore(model, err)
        trial_params["patience"] = patience
        self.best_state_ = guard.best_state
        return model, trial_params

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        parts = self._parts(*self._split(X), self._dummy_targets(len(X)))
        with torch_threads(self.n_threads):
            seed_everything(self.seed)
            dataset = self._dataset(*parts)
            tensors = to_torch(dataset, self.device_)
            raw = predict(
                self.model_, dataset, tensors, self.trial_params_, self.indices, self.device_, parts=("test",)
            )
        return np.asarray(raw["test"])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return replace_non_finite(self._predict(X).reshape(-1), self.fallback_, self.problem_type)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = replace_non_finite(logits_to_probs(self._predict(X), self.task_type), self.fallback_, self.problem_type)
        if self.problem_type == "binary":
            return np.column_stack([1.0 - probs, probs])
        return probs
