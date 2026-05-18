from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class TabPFNWideModel(AbstractTorchModel):
    """TabPFN-Wide: a TabPFN variant specialized for wide tabular datasets
    (many features, few samples).

    Paper: TabPFN-Wide (arXiv:2510.06162)
    Authors: Christopher Kolberg, Jules Kreuer, Jonas Huurdeman, Sofiane Ouaari,
        Katharina Eggensperger, Nico Pfeifer
    Codebase: https://github.com/not-a-feature/TabPFN-Wide
    License: Prior Labs License Version 1.1
    """

    ag_key = "TabPFN-Wide"
    ag_name = "TabPFN-Wide"
    ag_priority = 65
    seed_name = "random_state"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)

        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)

        return X

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        import torch

        available_num_gpus = ResourceManager.get_gpu_count_torch(cuda_only=True)
        if num_gpus > available_num_gpus:
            raise AssertionError(
                f"Fit specified to use {num_gpus} GPU, but only {available_num_gpus} "
                "CUDA GPUs are available. Please activate CUDA or switch to CPU usage.",
            )
        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not torch.cuda.is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        from tabpfnwide.classifier import TabPFNWideClassifier

        if self.problem_type not in ["binary", "multiclass"]:
            raise AssertionError(
                f"Unsupported problem_type: {self.problem_type}. "
                "TabPFN-Wide supports only classification."
            )

        hps = self._get_model_params()

        X = self.preprocess(X, y=y, is_train=True)

        self.model = TabPFNWideClassifier(
            device=device,
            **hps,
        )
        self.model.fit(X, y)

    def _set_default_params(self):
        default_params = {}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass"]

    def get_device(self) -> str:
        if hasattr(self.model, "device"):
            return self.model.device
        return "cpu"

    def _set_device(self, device: str):
        if hasattr(self.model, "to"):
            self.model.to(device)

    def _get_default_resources(self) -> tuple[int, int]:
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 1 if is_gpu_available else 0,
        }

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """Set fold_fitting_strategy to sequential_local to avoid crashes
        if model weights aren't pre-downloaded when fitting in parallel.
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        default_ag_args_ensemble.update({"fold_fitting_strategy": "sequential_local"})
        return default_ag_args_ensemble

    @classmethod
    def _class_tags(cls) -> dict:
        return {"can_estimate_memory_usage_static": False}

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
