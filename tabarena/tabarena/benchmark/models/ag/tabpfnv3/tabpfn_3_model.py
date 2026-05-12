from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd


class TabPFN3Model(AbstractTorchModel):
    """TabPFN-3 TabArena Integration."""

    ag_key = "TA-TABPFN-3"
    ag_name = "TA-TabPFN-3"
    ag_priority = 105
    seed_name = "random_state"

    default_classification_model: str | None = "tabpfn-v3-classifier-v3_default.ckpt"
    default_regression_model: str | None = "tabpfn-v3-regressor-v3_default.ckpt"

    _categorical_indices: list[int] | None
    """The indices of the categorical features, detected during preprocessing."""
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

        return X

    def _get_model_class(self):
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        is_classification = self.problem_type in ["binary", "multiclass"]

        return TabPFNClassifier if is_classification else TabPFNRegressor

    def _get_model_checkpoint(self):
        from tabpfn.model_loading import prepend_cache_path

        is_classification = self.problem_type in ["binary", "multiclass"]
        default_model = self.default_classification_model if is_classification else self.default_regression_model
        return prepend_cache_path(default_model)

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
        hps = self._get_model_params()
        default_hps = dict(
            model_path=self._get_model_checkpoint(),
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
            }
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
