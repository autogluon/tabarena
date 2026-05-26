from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class TabDPTModel(AbstractTorchModel):
    ag_key = "TA-TABDPT"
    ag_name = "TA-TabDPT"
    seed_name = "seed"
    default_random_seed = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._predict_hps = None
        self._use_flash_og = None


    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        from torch.cuda import is_available

        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not is_available()):
            # FIXME: warn instead and switch to CPU.
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )
        from tabdpt import TabDPTClassifier, TabDPTRegressor

        model_cls = (
            TabDPTClassifier
            if self.problem_type in [BINARY, MULTICLASS]
            else TabDPTRegressor
        )
        supported_predict_hps = (
            ("context_size", "permute_classes", "temperature")
            if model_cls is TabDPTClassifier
            else ("context_size",)
        )

        hps = self._get_model_params()
        random_seed = hps.pop(self.seed_name, self.default_random_seed)
        self._predict_hps = {k: v for k, v in hps.items() if k in supported_predict_hps}
        self._predict_hps["seed"] = random_seed
        X = self.preprocess(X, y=y)
        y = y.to_numpy()
        self.model = model_cls(
            device=device,
            use_flash=self._use_flash(),
            normalizer=hps.get("normalizer", "standard"),
            missing_indicators=hps.get("missing_indicators", False),
            clip_sigma=hps.get("clip_sigma", 4),
            feature_reduction=hps.get("feature_reduction", "pca"),
            faiss_metric=hps.get("faiss_metric", "l2"),
        )
        self.model.fit(X=X, y=y)

    @staticmethod
    def _use_flash() -> bool:
        """Detect if torch's native flash attention is available on the current machine."""
        import torch

        if not torch.cuda.is_available():
            return False

        device = torch.device("cuda:0")
        capability = torch.cuda.get_device_capability(device)

        return capability != (7, 5)

    def _post_fit(self, **kwargs):
        super()._post_fit(**kwargs)
        self._use_flash_og = self.model.use_flash
        return self

    def get_device(self) -> str:
        return self.model.device

    def _set_device(self, device: str):
        self.model.to(device)
        if device == "cpu":
            self.model.use_flash = False
            self.model.model.use_flash = False
        else:
            self.model.use_flash = self._use_flash_og
            self.model.model.use_flash = self._use_flash_og

    def _get_default_resources(self) -> tuple[int, int]:
        # Use only physical cores for better performance based on benchmarks
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)

        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))

        return num_cpus, num_gpus

    def get_minimum_resources(
        self, is_gpu_available: bool = False
    ) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 0.5 if is_gpu_available else 0,
        }

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)

        if self.problem_type in [REGRESSION]:
            return self.model.predict(X, **self._predict_hps)

        y_pred_proba = self.model.ensemble_predict_proba(X, **self._predict_hps)
        return self._convert_proba_to_unified_form(y_pred_proba)

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """TabDPT requires numpy array as input."""
        X = super()._preprocess(X, **kwargs)
        if self._feature_generator is None:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(
                X=X
            )
        return X.to_numpy()

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
            "refit_folds": True,
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    # FIXME: This is copied from TabPFN, but TabDPT is not the same
    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Heuristic memory estimate based on TabPFN's memory estimate logic in:
        https://github.com/PriorLabs/TabPFN/blob/57a2efd3ebdb3886245e4d097cefa73a5261a969/src/tabpfn/model/memory.py#L147.

        This is based on GPU memory usage, but hopefully with overheads it also approximates CPU memory usage.
        """
        # TODO: update, this is not correct anymore, consider using internal TabPFN functions directly.
        features_per_group = 3  # Based on TabPFNv2 default (unused)
        n_layers = 12  # Based on TabPFNv2 default
        embedding_size = 192  # Based on TabPFNv2 default
        dtype_byte_size = 2  # Based on TabPFNv2 default

        model_mem = 14489108  # Based on TabPFNv2 default

        n_samples, n_features = X.shape[0], min(X.shape[1], 500)
        n_feature_groups = (n_features) / features_per_group + 1  # TODO: Unsure how to calculate this

        X_mem = n_samples * n_feature_groups * dtype_byte_size
        activation_mem = n_samples * n_feature_groups * embedding_size * n_layers * dtype_byte_size

        baseline_overhead_mem_est = 1e9  # 1 GB generic overhead

        # Add some buffer to each term + 1 GB overhead to be safe
        memory_estimate = model_mem + 4 * X_mem + 2 * activation_mem + baseline_overhead_mem_est

        # TabDPT memory estimation is very inaccurate because it is using TabPFN memory estimate. Double it to be safe.
        memory_estimate = memory_estimate * 2

        # Note: This memory estimate is way off if `context_size` is not None
        return int(memory_estimate)
