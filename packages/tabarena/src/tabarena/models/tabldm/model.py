from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

_HF_REPO = "occams/Xiaomi-TabLDM"
_CLASSIFIER_CHECKPOINT = "checkpoints/clf_stage3_moe1_step-10000.ckpt"
_REGRESSOR_CHECKPOINT = "checkpoints/reg_stage3_moe1_step-10000.ckpt"


class TabLDMModel(AbstractTorchModel):
    """TabLDM: a tabular foundation model with a dual-stream column embedder and MoE backbone.

    Uses the enhanced sklearn estimators (``TabLDMEnhancedClassifier``/``TabLDMEnhancedRegressor``),
    which wrap the base MoE1 model with an ensemble/calibration pipeline on top.

    Codebase: vendored from the ``xiaomi-tabldm`` package (pip name ``tabldm``), not on PyPI.
    Checkpoints: https://huggingface.co/occams/Xiaomi-TabLDM
    License: Apache-2.0 (Copyright Xiaomi Corporation)
    """

    ag_key = "TA-TABLDM"
    ag_name = "TA-TabLDM"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1
    # Sequential fold fitting avoids contention on the shared HF checkpoint cache; refitting one
    # model on all data (like the other TFM wrappers: TabICL, TabSwift, LimiX, ...) gives faster
    # inference at similar quality to the bagged ensemble for an in-context-learning model.
    _default_ag_args_ensemble_extra = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }

    def get_model_cls(self):
        if self.problem_type in ("binary", "multiclass"):
            from tabarena.models.tabldm._vendor._sklearn.classifier_enhanced import TabLDMEnhancedClassifier

            return TabLDMEnhancedClassifier
        from tabarena.models.tabldm._vendor._sklearn.regressor_enhanced import TabLDMEnhancedRegressor

        return TabLDMEnhancedRegressor

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        """Fit TabLDM.

        As an in-context-learning foundation model there is no training loop and no early
        stopping, so (like the other TFM wrappers) ``X_val``/``y_val`` and ``time_limit`` are
        intentionally ignored. Categorical columns and missing values need no upfront handling
        here: the vendored estimator's own ``TransformToNumerical`` preprocessing ordinal-encodes
        categorical dtypes and mean-imputes numeric NaNs internally when given a DataFrame, and
        for regression it standardizes/inverse-transforms the target itself, so ``X``/``y`` are
        passed straight through.
        """
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

        model_cls = self.get_model_cls()
        hps = self._get_model_params()

        X = self.preprocess(X, y=y, is_train=True)

        self.model = model_cls(device=device, n_jobs=num_cpus, **hps)
        self.model.fit(X, y)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)

        if self.problem_type == "regression":
            y_pred_proba = self.model.predict(X)
        else:
            y_pred_proba = self.model.predict_proba(X)

        return self._convert_proba_to_unified_form(y_pred_proba)

    def get_device(self) -> str:
        return self.model.device_.type

    def _set_device(self, device: str):
        device = self.to_torch_device(device)
        self.model.device_ = device
        self.model.model_ = self.model.model_.to(device)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def prefetch_weights(cls) -> None:
        """Pre-download both checkpoints (classifier and regressor) from the public HF repo.

        Constructs each vendored estimator with its default ``checkpoint_version`` and calls its
        own ``_load_model()`` directly (matching ``tabicl``'s wrapper), rather than reimplementing
        the cache-first/download-fallback logic that ``_load_model()`` already provides.
        """
        from tabarena.models.tabldm._vendor._sklearn.classifier_enhanced import TabLDMEnhancedClassifier
        from tabarena.models.tabldm._vendor._sklearn.regressor_enhanced import TabLDMEnhancedRegressor

        TabLDMEnhancedClassifier(checkpoint_version=_CLASSIFIER_CHECKPOINT)._load_model()
        TabLDMEnhancedRegressor(checkpoint_version=_REGRESSOR_CHECKPOINT)._load_model()
