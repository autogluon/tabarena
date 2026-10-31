"""Fit and predict one agentic TabM with the modules of ``bin.tabm``.

Follows ``_build_dataset``, ``train_and_eval`` and ``predict`` in ``bin/tabm/pipeline.py`` (agentic-tabular
3d7349b) on arrays in memory, with ``n_classes`` from AutoGluon (a fold missing a rare class keeps the output
width) and the global RNGs seeded before the network is built and before each prediction. The preprocessing
fits inside the call that transforms every part, so a prediction rebuilds the dataset with the new rows as the
``test`` part; it is seeded, so the training part comes out identical.
"""

from __future__ import annotations

import contextlib
import copy
import io
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from bin.tabm import pipeline as tabm_pipeline
from bin.tabm.core import Dataset, compute_metrics_for_parts, logits_to_probs, seed_everything, to_torch
from bin.tabm.hparams import resolve


@contextmanager
def torch_state(device: torch.device, n_threads: int):
    """Use ``n_threads`` torch threads, and TF32 on CUDA, inside the block; restore both afterwards."""
    old = torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.set_num_threads(n_threads)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        yield
    finally:
        torch.set_num_threads(old[0])
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[1:]


@contextmanager
def bf16_forward(model: torch.nn.Module, device: torch.device):
    """The package runs TabM under bf16 autocast (``pipeline._enable_bf16_autocast``). AutoGluon pickles fitted
    models, so the autocast is installed only while the model runs.
    """
    model.__dict__.pop("forward", None)
    tabm_pipeline._enable_bf16_autocast(model, device)
    try:
        yield
    finally:
        model.__dict__.pop("forward", None)


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


class AgenticTabMImplementation:
    """One fitted agentic TabM (a single bag child)."""

    def __init__(
        self,
        *,
        problem_type: str,
        n_classes: int,
        device: str,
        n_threads: int,
        verbosity: int,
        eval_metric,
        indices,
        **params,
    ):
        self.problem_type = problem_type
        self.task_type = "binclass" if problem_type == "binary" else problem_type
        self.n_classes = {"regression": 0, "binary": 2}.get(problem_type, n_classes)
        self.device_ = torch.device(device)
        self.n_threads = n_threads
        self.verbosity = verbosity
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
        self.trial_params_ = {**params, "seed": self.seed}
        self.model_ = None

    def _split(self, X: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Float32 numerical columns with NaN kept, and the categorical columns as codes over the training
        categories, with missing values on ``len(categories)`` (a category of its own, as in
        ``bin.tabm.core.convert_bin_to_cat``) and unseen ones on ``len(categories) + 1``, the unknown bucket.
        """
        x_num = X[self.num_cols_].to_numpy(dtype=np.float32) if self.num_cols_ else None
        x_cat = None
        if self.categories_:
            columns = []
            for col, cats in self.categories_.items():
                codes = cats.get_indexer(X[col])
                codes = np.where(X[col].isna().to_numpy(), len(cats), np.where(codes < 0, len(cats) + 1, codes))
                columns.append(codes.astype(np.int64))
            x_cat = np.column_stack(columns)
        return x_num, x_cat

    def _labels(self, y: pd.Series) -> np.ndarray:
        return np.asarray(y).reshape(-1).astype(np.float32 if self.problem_type == "regression" else np.int64)

    def _dataset(self, x_num_test, x_cat_test, y_test) -> tuple[Dataset, object]:
        """The dataset and its device tensors, with the fit's training and validation parts and the given test part."""
        x_num = None if self.x_num_fit_ is None else {**self.x_num_fit_, "test": x_num_test}
        x_cat = None if self.x_cat_fit_ is None else {**self.x_cat_fit_, "test": x_cat_test}
        y = {**self.y_fit_, "test": y_test}
        config = {"seed": self.seed, "y_train": y["train"], "n_classes": self.n_classes, "task_type": self.task_type}
        x_num, num_artifacts = self._stage("numerical_preprocess_idx")(x_num, config)
        x_cat, cat_artifacts = self._stage("categorical_preprocess_idx")(x_cat, config)
        y_prepared, target_artifacts = self._stage("target_preprocess_idx")(y, config)
        if self.task_type == "regression":
            target_artifacts = bounded_inverse(target_artifacts, y_prepared["train"])
        dataset = Dataset(
            x_num=x_num,
            x_cat=x_cat,
            y=y_prepared,
            y_raw=y,
            task_type=self.task_type,
            score_name=self.score_name,
            n_classes=self.n_classes,
            preprocess_artifacts={"numerical": num_artifacts, "categorical": cat_artifacts, "target": target_artifacts},
        )
        return dataset, to_torch(dataset, self.device_)

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
        self.y_fit_ = {"train": self._labels(y), "val": self._labels(y_val)}

        # The train modules score a test part once after training; the validation rows stand in for it.
        epoch_log = contextlib.nullcontext() if self.verbosity >= 3 else contextlib.redirect_stdout(io.StringIO())
        with torch_state(self.device_, self.n_threads), epoch_log:
            deadline = None if time_limit is None else time.monotonic() + max(0.0, time_limit)
            dataset, tensors = self._dataset(x_num_val, x_cat_val, self.y_fit_["val"])
            seed_everything(self.seed)
            self.model_ = tabm_pipeline._build_model(dataset, tensors, self.trial_params_, self.indices, self.device_)
            loss_fn = self._stage("loss_idx")
            optimizer_builder = self._stage("optimizer_idx")
            train_fn = self._stage("train_idx")
            guard = TrainingGuard(self.model_, self._stage("inference_idx"), deadline, ("val",))
            patience = self.trial_params_["patience"]
            try:
                train_fn(
                    self.model_,
                    tensors,
                    dataset,
                    self.trial_params_,
                    guard,
                    loss_fn,
                    optimizer_builder,
                    self.device_,
                    self.seed,
                )
            except (RuntimeError, ValueError) as err:
                guard.restore(self.model_, err)
            finally:
                self.model_.__dict__.pop("forward", None)  # the bf16 closure; see ``bf16_forward``
            self.trial_params_["patience"] = patience
        # A model that predicts non-finite values on the validation rows goes back to the guard's best epoch.
        self.fallback_ = finite_fallback(self.y_fit_["train"], self.problem_type, self.n_classes)
        if not np.isfinite(self._predict_raw(X_val)).all():
            self.model_.load_state_dict(guard.best_state)

    def _predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Predictions on the target scale for regression, logits for classification."""
        x_num, x_cat = self._split(X)
        # Placeholder targets: no prediction path reads the test labels.
        y_test = np.zeros(len(X), dtype=self.y_fit_["train"].dtype)
        with torch_state(self.device_, self.n_threads):
            dataset, tensors = self._dataset(x_num, x_cat, y_test)
            seed_everything(self.seed)
            with bf16_forward(self.model_, self.device_):
                predictions = tabm_pipeline.predict(
                    self.model_, dataset, tensors, self.trial_params_, self.indices, self.device_, parts=("test",)
                )
        return np.asarray(predictions["test"])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return replace_non_finite(self._predict_raw(X).reshape(-1), self.fallback_, self.problem_type)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = replace_non_finite(
            logits_to_probs(self._predict_raw(X), self.task_type), self.fallback_, self.problem_type
        )
        if self.problem_type == "binary":
            return np.column_stack([1.0 - proba, proba])
        return proba
