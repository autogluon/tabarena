from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

CAT_MASK_PARAM_NAMES = ("cat_mask_drop_prob", "cat_mask_add_prob", "cat_mask_seed")
"""Wrapper-only hyperparameters that corrupt the categorical mask (see
:func:`perturb_categorical_mask`); popped in :meth:`TabPFN3Model._fit` so they never
reach the tabpfn estimator."""


def perturb_categorical_mask(
    cat_indices: list[int],
    n_cols: int,
    *,
    drop_prob: float,
    add_prob: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Randomly corrupt a categorical-feature mask to simulate imperfect type detection.

    Each true categorical index is removed from the mask independently with probability
    ``drop_prob`` (the model will see that column as numeric) and each non-categorical
    column index is added independently with probability ``add_prob`` (the model will
    treat that numeric column as categorical).

    Returns ``(perturbed_mask, dropped_indices)``, both sorted.
    """
    rng = np.random.default_rng(seed)
    cat_set = set(cat_indices)
    dropped = [i for i in sorted(cat_set) if rng.random() < drop_prob]
    added = [i for i in range(n_cols) if i not in cat_set and rng.random() < add_prob]
    kept = sorted((cat_set - set(dropped)) | set(added))
    return kept, dropped


class TabPFN3Model(AbstractTorchModel):
    """TabPFN-3 TabArena Integration.

    Supports an optional categorical-mask corruption experiment via wrapper-only
    hyperparameters: ``cat_mask_drop_prob`` (probability that each true categorical
    column is passed to tabpfn as numeric), ``cat_mask_add_prob`` (probability that
    each numeric column is falsely marked categorical) and ``cat_mask_seed``
    (perturbation RNG seed, default 0). Dropped columns are label-encoded to numeric
    codes at fit and predict time, so tabpfn cannot recover them from their dtype.
    """

    ag_key = "TA-TABPFN-3"
    ag_name = "TA-TabPFN-3"
    ag_priority = 105
    seed_name = "random_state"

    default_classification_model: str | None = "tabpfn-v3-classifier-v3_default.ckpt"
    default_regression_model: str | None = "tabpfn-v3-regressor-v3_default.ckpt"

    checkpoint_param_name: str = "checkpoint_per_problem_type"
    """Name of the optional config hyperparameter that overrides the checkpoint per problem type.

    Its value is a dict mapping a problem type to a checkpoint. Keys may be ``"binary"`` /
    ``"multiclass"`` / ``"regression"``, or the ``"classification"`` umbrella (used for both
    ``"binary"`` and ``"multiclass"`` unless a more specific key is given). Each value is a bare
    filename (resolved in the tabpfn cache dir) or an absolute path to a ``.ckpt``. Problem types
    not listed fall back to ``default_classification_model`` / ``default_regression_model``. It is
    popped from the hyperparameters in :meth:`_fit` (it is not a tabpfn estimator argument).
    """

    _categorical_indices: list[int] | None
    """The indices of the categorical features, detected during preprocessing."""
    _cat_mask_dropped_cols: list[str] | None = None
    """Columns removed from the categorical mask by the corruption experiment; they are
    label-encoded to numeric codes at fit and predict time."""
    fixed_random_state: int = 0
    """Using a fixed random seed, as in TabPFN-2.6."""

    def _preprocess(self, X: pd.DataFrame, *, is_train=False, **kwargs) -> pd.DataFrame:
        """Minimal model-specific preprocessing to detect the indices of categorical features."""
        X = super()._preprocess(X, **kwargs)

        if is_train:
            categorical_cols = X.select_dtypes(include=["category"]).columns.tolist()
            if categorical_cols:
                self._categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]
            else:
                self._categorical_indices = None

            params = self._get_model_params()
            drop_prob = params.get("cat_mask_drop_prob", 0.0)
            add_prob = params.get("cat_mask_add_prob", 0.0)
            if drop_prob or add_prob:
                mask, dropped = perturb_categorical_mask(
                    self._categorical_indices or [],
                    X.shape[1],
                    drop_prob=drop_prob,
                    add_prob=add_prob,
                    seed=params.get("cat_mask_seed", 0),
                )
                self._categorical_indices = mask or None
                self._cat_mask_dropped_cols = [X.columns[i] for i in dropped]

        # Dropped categoricals must genuinely look numeric to tabpfn (which otherwise
        # re-detects them from their dtype): replace them with their category codes.
        # AutoGluon fixes each column's categories at train time, so the codes are
        # consistent between fit and predict.
        if self._cat_mask_dropped_cols:
            X = X.copy()
            for col in self._cat_mask_dropped_cols:
                codes = X[col].cat.codes.astype("float32")
                X[col] = codes.mask(codes < 0)  # cat.codes encodes NaN as -1

        return X

    def _get_model_class(self):
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        is_classification = self.problem_type in ["binary", "multiclass"]

        return TabPFNClassifier if is_classification else TabPFNRegressor

    def _resolve_checkpoint_for_problem_type(self, checkpoint_per_problem_type: dict[str, str] | None) -> str | None:
        """Select this task's checkpoint name from an optional per-problem-type override.

        Resolution order: the exact ``problem_type`` key (``"binary"`` / ``"multiclass"`` /
        ``"regression"``) -> the ``"classification"`` umbrella key (for binary/multiclass only) ->
        the ``default_classification_model`` / ``default_regression_model`` class attribute.
        """
        is_classification = self.problem_type in ["binary", "multiclass"]
        overrides = checkpoint_per_problem_type or {}
        model = overrides.get(self.problem_type)
        if model is None and is_classification:
            model = overrides.get("classification")
        if model is None:
            model = self.default_classification_model if is_classification else self.default_regression_model
        return model

    def _get_model_checkpoint(self, checkpoint_per_problem_type: dict[str, str] | None = None):
        """Resolve the checkpoint to a full path: pick the name (see
        :meth:`_resolve_checkpoint_for_problem_type`) then prepend the tabpfn cache dir (a no-op
        for an absolute path).
        """
        from tabpfn.model_loading import prepend_cache_path

        return prepend_cache_path(self._resolve_checkpoint_for_problem_type(checkpoint_per_problem_type))

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        X = self.preprocess(X, y=y, is_train=True)

        # Set hyperparameters
        hps = dict(self._get_model_params())
        checkpoint_per_problem_type = hps.pop(self.checkpoint_param_name, None)
        for param in CAT_MASK_PARAM_NAMES:  # wrapper-only, consumed in _preprocess
            hps.pop(param, None)
        default_hps = dict(
            model_path=self._get_model_checkpoint(checkpoint_per_problem_type),
            device=self._resolve_tabpfn_device(num_gpus=num_gpus),
            n_jobs=num_cpus,
            categorical_features_indices=self._categorical_indices,
        )
        default_hps[self.seed_name] = self.fixed_random_state
        hps = {**default_hps, **hps}  # hps later to override any conflicting keys default keys.

        # Initialize and fit the model
        model_class = self._get_model_class()
        self.model = model_class(**hps)
        self.model = self.model.fit(
            X=X,
            y=y,
        )

    # --- Model Behavior Management ---
    def _set_default_params(self):
        default_params = {
            "ignore_pretraining_limits": True,  # to ignore warnings and size limits
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        """Default code here supports all problem types, but can be overridden if needed."""
        return ["binary", "multiclass", "regression"]

    # TODO:
    #  - add support for many-class wrapper to remove the limit fully
    #  - add row/col limit?
    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_classes": 160,
                # Batch inference once we exceed 150_000 samples (batching starts at 150_001).
                "max_batch_size": 150_000,
            },
        )
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """Ensure one fold is fit at a time and refits is enabled by default."""
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
            "fold_fitting_strategy": "sequential_local",
            "refit_folds": default_ag_args_ensemble.pop("refit_folds", True),
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    # --- Resource and GPU Management ---
    @staticmethod
    def _resolve_tabpfn_device(num_gpus: int) -> str | list[str]:
        """Return device type based on number of GPUs, ensuring that if
        GPUs are requested, they are available.
        """
        if num_gpus <= 0:
            return "cpu"

        import torch

        if not torch.cuda.is_available():
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        if num_gpus == 1:
            return "cuda"

        return [f"cuda:{i}" for i in range(num_gpus)]

    def _get_default_resources(self) -> tuple[int, int]:
        # Use only physical cores for better performance based on benchmarks
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 1 if is_gpu_available else 0,
        }

    def get_device(self) -> str:
        base = self.model
        if hasattr(base, "devices_"):
            return base.devices_[0].type

        from collections.abc import Sequence

        device = base.device
        if isinstance(device, Sequence):
            return device[0]

        return device

    def _set_device(self, device: str):
        self.model.to(device)

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()
        return self.estimate_memory_usage_static(
            X=X,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
            hyperparameters=hyperparameters,
            **kwargs,
        )

    @classmethod
    def _class_tags(cls):
        return {"can_estimate_memory_usage_static": True}

    # TODO: obtain memory estimation with/without chunking
    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        """Assume a 10 GB baseline (model + activations) plus the dataset memory footprint."""
        baseline_mem_est = 10 * 1e9  # 10 GB minimum for TabPFN-3 model + activations
        dataset_mem_est = 5 * get_approximate_df_mem_usage(X).sum()
        return int(baseline_mem_est + dataset_mem_est)


def prefetch_weights() -> None:
    """Pre-download all TabPFN checkpoints via the tabpfn loader (warms the cache)."""
    from tabpfn.model_loading import download_all_models, resolve_model_path

    _, model_dir, _, _ = resolve_model_path(model_path=None, which="classifier")
    download_all_models(to=model_dir[0])
