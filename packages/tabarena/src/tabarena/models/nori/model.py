from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class NoriModel(AbstractTorchModel):
    """Nori: a tabular foundation model for regression via in-context learning.

    Paper/citation: Synthefy Nori
    Authors: Synthefy (Li Po-han, Aditya Narayanan, Sai Shankar Narasimhan, et al.)
    Codebase: https://github.com/Synthefy/synthefy-nori
    License: Apache-2.0

    Notes:
        - ``NoriRegressor`` is a scikit-learn estimator (``fit``/``predict``) and
          normalizes the target internally, so we pass ``y`` through unchanged and
          rely on the default regression ``_predict_proba`` path.
        - ``NoriRegressor.fit`` coerces ``X`` to a float32 array, so categoricals are
          label-encoded here. NaN is forwarded as-is: Nori's inference pipeline
          handles missing values natively (``allow-nan``).
        - ``NoriRegressor`` exposes no random seed (inference is deterministic given
          the context), so ``seed_name`` is left unset.
    """

    ag_key = "TA-NORI"
    ag_name = "TA-Nori"
    ag_priority = 65

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator: LabelEncoderFeatureGenerator | None = None
        self._cat_indices: list[int] | None = None

    def _preprocess(self, X: pd.DataFrame, *, is_train: bool = False, **kwargs) -> np.ndarray:
        """Label-encode categoricals to numeric and return a float32 array.

        NaN is preserved (Nori handles missing values natively); only categorical
        columns are encoded, numeric columns pass through untouched.
        """
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)

        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
            if is_train:
                self._cat_indices = [X.columns.get_loc(c) for c in self._feature_generator.features_in]

        return np.asarray(X.to_numpy(), dtype=np.float32)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_gpus: int = 0,
        **kwargs,
    ):
        import torch
        from synthefy_nori import NoriRegressor

        from tabarena.models.nori._patch import ensure_picklable_preprocessing

        # Nori's fitted preprocessing pipeline uses local lambdas that the stdlib pickle
        # AutoGluon saves models with cannot serialize; patch before fit builds the pipeline.
        ensure_picklable_preprocessing()

        if self.problem_type != "regression":
            raise AssertionError(f"{self.ag_name} only supports regression, got problem_type={self.problem_type!r}.")

        available_num_gpus = ResourceManager.get_gpu_count_torch(cuda_only=True)
        if num_gpus > available_num_gpus:
            raise AssertionError(
                f"Fit specified to use {num_gpus} GPU, but only {available_num_gpus} "
                "CUDA GPUs are available. Please activate CUDA or switch to CPU usage.",
            )
        device = "cuda" if num_gpus != 0 else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        X = self.preprocess(X, y=y, is_train=True)

        hps = self._get_model_params()
        hps.pop("device", None)  # device is set explicitly from the allocated resources

        # NoriRegressor normalizes y internally and denormalizes its predictions, so
        # we pass y through unchanged (it is coerced to float64 inside fit).
        self.model = NoriRegressor(device=device, **hps)
        self.model.fit(X, y)

    def _set_default_params(self):
        default_params = {}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["regression"]

    # --- Resource and GPU management ---
    def get_device(self) -> str:
        device = self.model.device
        if device is None:
            return "cpu"
        return device if isinstance(device, str) else device.type

    def _set_device(self, device: str):
        import torch

        torch_device = torch.device(device)
        self.model.device = torch_device
        # The predictor (and its torch module) are built lazily on the first predict
        # call; move them only if they already exist.
        predictor = getattr(self.model, "_predictor", None)
        if predictor is not None:
            predictor.device = torch_device
            if getattr(predictor, "model", None) is not None:
                predictor.model.to(torch_device)

    def _get_default_resources(self) -> tuple[int, int]:
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 1 if is_gpu_available else 0,
        }

    def _get_default_auxiliary_params(self) -> dict:
        """Cap context size at 100k rows; no feature or class limits (regression-only)."""
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_rows": 100_000,
            },
        )
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """Fit one fold at a time (avoids contention on the shared checkpoint cache) and
        refit by default (a single forward-pass model has no per-fold validation cost).
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        default_ag_args_ensemble.update(
            {
                "fold_fitting_strategy": "sequential_local",
                "refit_folds": default_ag_args_ensemble.pop("refit_folds", True),
            },
        )
        return default_ag_args_ensemble

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        return self.estimate_memory_usage_static(
            X=X,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
            hyperparameters=self._get_model_params(),
            **kwargs,
        )

    @classmethod
    def _estimate_memory_usage_static(cls, *, X: pd.DataFrame, **kwargs) -> int:
        """Assume a small-model baseline (weights + activations) plus the dataset footprint."""
        baseline_mem_est = 3 * 1e9  # 3 GB for the model + activations
        dataset_mem_est = 5 * get_approximate_df_mem_usage(X).sum()
        return int(baseline_mem_est + dataset_mem_est)

    @classmethod
    def _class_tags(cls) -> dict:
        return {"can_estimate_memory_usage_static": True}

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def prefetch_weights(cls) -> None:
        """Pre-download the default Nori checkpoint from the Hugging Face Hub.

        Used by the foundation-model pre-download scripts to warm the cache before
        parallel fit runs. The Hub repo is gated, so this requires a Hugging Face
        token (``HF_TOKEN`` / ``hf auth login``).
        """
        from synthefy_nori.hf import download_checkpoint

        download_checkpoint()
