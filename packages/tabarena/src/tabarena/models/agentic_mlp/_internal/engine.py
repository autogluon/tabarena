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


class MemberEnsemble(torch.nn.Module):
    """``n_ens`` independent copies of one package model, trained together in one batched pass (as RealMLP's
    ``n_ens``). Each member has its own initialization (seed ``seed + i``), dropout masks and gradient clipping; the
    members share the training batches.

    Parameters and buffers of all members are stacked along a leading member axis and registered here, so the
    optimizer, ``state_dict`` copies and device moves see one module. A template member computes the forward with
    ``torch.func.functional_call``, vectorized over members with ``torch.vmap`` (a loop over members where a layer
    does not support vmap). In training mode the forward returns every member's output (for ``member_loss``); in
    evaluation mode it returns the members' combined output in the package's own format: the mean for
    regression, the logit or log of the mean class probabilities for classification, so the package's inference
    modules and metrics read it as one model.
    """

    def __init__(self, template: torch.nn.Module, params: dict, buffers: dict, problem_type: str):
        super().__init__()
        self.problem_type = problem_type
        self.n_members = len(next(iter(params.values())))
        self.param_names, self.buffer_names = list(params), list(buffers)
        for j, name in enumerate(self.param_names):
            self.register_parameter(f"p{j}", torch.nn.Parameter(params[name].detach()))
        for j, name in enumerate(self.buffer_names):
            self.register_buffer(f"b{j}", buffers[name].detach())
        # Not a child module: its own tensors are never used, only its structure and layer settings.
        object.__setattr__(self, "template", template)
        # Dropout layers stay visible to ``modules()`` (inference modules that switch them on for MC dropout).
        self.dropouts = torch.nn.ModuleList(m for m in template.modules() if isinstance(m, torch.nn.Dropout))
        self.use_vmap = True

    @classmethod
    def from_members(cls, members: list, problem_type: str) -> MemberEnsemble:
        params, buffers = torch.func.stack_module_state(members)
        return cls(members[0], params, buffers, problem_type)

    def _apply(self, fn, recurse=True):
        self.template._apply(fn, recurse)
        return super()._apply(fn, recurse)

    def train(self, mode: bool = True):
        self.template.train(mode)
        return super().train(mode)

    def _stacked(self):
        params = {name: getattr(self, f"p{j}") for j, name in enumerate(self.param_names)}
        buffers = {name: getattr(self, f"b{j}") for j, name in enumerate(self.buffer_names)}
        return params, buffers

    def _member(self, params, buffers, x_num, x_cat):
        return torch.func.functional_call(self.template, (params, buffers), (x_num, x_cat))

    def forward(self, x_num, x_cat):
        params, buffers = self._stacked()
        out = None
        if self.use_vmap:
            try:
                out = torch.vmap(self._member, in_dims=(0, 0, None, None), randomness="different")(
                    params, buffers, x_num, x_cat
                )
            except RuntimeError as err:
                if is_oom_exception(err):
                    raise
                self.use_vmap = False
        if out is None:
            out = torch.stack(
                [
                    self._member(
                        {k: v[i] for k, v in params.items()}, {k: v[i] for k, v in buffers.items()}, x_num, x_cat
                    )
                    for i in range(self.n_members)
                ]
            )
        return out if self.training else self.combine(out)

    def combine(self, out: torch.Tensor) -> torch.Tensor:
        if self.problem_type == "regression":
            return out.mean(0)
        if self.problem_type == "binary" and (out.ndim == 2 or out.shape[-1] == 1):
            return torch.logit(torch.sigmoid(out.float()).mean(0), eps=1e-7)
        return torch.log(torch.softmax(out.float(), dim=-1).mean(0).clamp_min(1e-12))

    def member_loss(self, loss_fn):
        """The sum of the members' losses, each from the package's loss on that member's output."""

        def loss(preds, targets, dataset_meta):
            if not self.training:
                return loss_fn(preds, targets, dataset_meta)
            return sum(loss_fn(preds[i], targets, dataset_meta) for i in range(self.n_members))

        return loss

    def clipped_optimizer(self, optimizer_builder, max_norm: float):
        """The package's optimizer, with each member's gradient clipped to ``max_norm`` on its own before a step."""

        def clip_members(optimizer, args, kwargs):
            grads = [p.grad for p in self.parameters() if p.grad is not None]
            if grads:
                norms = torch.stack([g.detach().float().flatten(1).pow(2).sum(1) for g in grads]).sum(0).sqrt()
                scale = (max_norm / (norms + 1e-6)).clamp(max=1.0)
                for g in grads:
                    g.mul_(scale.view(-1, *([1] * (g.ndim - 1))).to(g.dtype))

        def build(model, trial_params):
            optimizer = optimizer_builder(model, trial_params)
            # Lookahead wraps a base optimizer and does not set up step hooks itself; its step calls the base's.
            hooked = optimizer if hasattr(optimizer, "_optimizer_step_pre_hooks") else optimizer._base
            hooked.register_step_pre_hook(clip_members)
            return optimizer

        return build

    def freeze_parametrizations(self) -> None:
        """Replace the spectral-norm parametrizations (model 11) by each member's effective weight
        ``W / (u^T W v)`` from the stored power-iteration vectors, so the fitted ensemble pickles.
        """
        template = copy.deepcopy(self.template)
        params, buffers = self._stacked()
        params = {k: v.detach() for k, v in params.items()}
        buffers = dict(buffers)
        changed = False
        for module_name, module in template.named_modules():
            for name in list(getattr(module, "parametrizations", {})):
                prefix = f"{module_name}.parametrizations.{name}".lstrip(".")
                weight = params.pop(f"{prefix}.original")
                u, v = buffers.pop(f"{prefix}.0._u"), buffers.pop(f"{prefix}.0._v")
                sigma = torch.einsum("ko,koi,ki->k", u, weight, v)
                parametrize.remove_parametrizations(module, name, leave_parametrized=False)
                params[f"{module_name}.{name}".lstrip(".")] = weight / sigma.view(-1, 1, 1)
                changed = True
        if changed:
            self.__init__(template, params, buffers, self.problem_type)


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
        self.n_ens = int(params.pop("n_ens", 1))
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
        if isinstance(self.model_, MemberEnsemble):
            self.model_.freeze_parametrizations()
        for module in self.model_.modules():
            for name in list(getattr(module, "parametrizations", {})):
                parametrize.remove_parametrizations(module, name, leave_parametrized=True)

    def _train(self, x_num, x_cat, y, deadline: float | None) -> tuple[torch.nn.Module, dict]:
        trial_params = {**self.params, "seed": self.seed}
        dataset = self._dataset(x_num, x_cat, y)
        tensors = to_torch(dataset, self.device_)
        loss_fn = self._stage("loss_idx")
        output_dim = infer_loss_output_dim(loss_fn, dataset)
        optimizer_builder = self._stage("optimizer_idx")
        if self.n_ens == 1:
            model = _build_model(dataset, tensors, trial_params, self.indices, output_dim, self.device_)
        else:
            members = []
            for i in range(self.n_ens):
                seed_everything(self.seed + i)
                members.append(_build_model(dataset, tensors, trial_params, self.indices, output_dim, self.device_))
            model = MemberEnsemble.from_members(members, self.problem_type)
            loss_fn = model.member_loss(loss_fn)
            optimizer_builder = model.clipped_optimizer(optimizer_builder, float(trial_params["gradient_clip"]))
            trial_params["gradient_clip"] = float("inf")
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
