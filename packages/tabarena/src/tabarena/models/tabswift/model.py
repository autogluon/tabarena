from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel
from sklearn.impute import SimpleImputer

if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger(__name__)

_DEFAULT_HF_REPO = "LAMDA-Tabular/TabSwift"
_DEFAULT_HF_FILENAME = "swift.ckpt"
"""A single ``swift.ckpt`` checkpoint serves both the classifier and the regressor."""


class TabSwiftModel(AbstractTorchModel):
    """TabSwift: an efficient tabular foundation model with row-wise attention.

    Paper: TabSwift: An Efficient Tabular Foundation Model with Row-Wise Attention (ICML 2026)
    Authors: Si-Yang Liu, Han-Jia Ye
    Codebase: https://github.com/LAMDA-Tabular/TabSwift
    License: MIT
    """

    ag_key = "TA-TABSWIFT"
    ag_name = "TA-TabSwift"
    ag_priority = 65
    seed_name = "random_state"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator: LabelEncoderFeatureGenerator | None = None
        self._imputer: SimpleImputer | None = None
        self._y_mean: float | None = None
        self._y_std: float | None = None

    def _preprocess(self, X: pd.DataFrame, *, is_train: bool = False, **kwargs) -> np.ndarray:
        """Produce a dense float32 matrix for TabSwift.

        TabSwift's forward pass is a linear projection over numeric row vectors — it has no
        native categorical or missing-value handling — so (matching TALENT's ``cat_policy=
        'indices'`` + NaN processing) we ordinal-encode categoricals to integer codes and
        mean-impute any remaining missing numeric cells. Encoder and imputer are fit on train
        and reused at predict so the two stay aligned.
        """
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)

        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)

        X = np.asarray(X.to_numpy(), dtype=np.float32)

        if is_train:
            # keep_empty_features=True: an all-NaN column becomes 0 instead of being dropped,
            # so the feature count stays constant between fit and predict.
            self._imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
            X = self._imputer.fit_transform(X)
        else:
            X = self._imputer.transform(X)

        return np.asarray(X, dtype=np.float32)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        """Fit TabSwift.

        As an in-context-learning foundation model there is no training loop and no early
        stopping, so (like the other TFM wrappers) ``X_val`` / ``y_val`` and ``time_limit``
        are intentionally ignored — fitting only prepares the data transforms and loads the
        pre-trained checkpoint.
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

        from tabarena.models.tabswift._vendor.classifier import TabSwiftClassifier
        from tabarena.models.tabswift._vendor.regressor import TabSwiftRegressor

        hps = self._get_model_params()
        model_path = hps.pop("model_path", None) or self.prefetch_weights()

        X_np = self.preprocess(X, y=y, is_train=True)

        if self.problem_type == "regression":
            # TabSwift's regressor does not standardize the target internally (matching
            # TALENT, which standardizes y up front and inverse-transforms predictions).
            y_np = np.asarray(y.to_numpy(), dtype=np.float32)
            self._y_mean = float(y_np.mean())
            self._y_std = float(y_np.std()) or 1.0
            y_fit = (y_np - self._y_mean) / self._y_std
            model_cls = TabSwiftRegressor
        else:
            y_fit = np.asarray(y.to_numpy())
            model_cls = TabSwiftClassifier

        self.model = model_cls(
            model_path=str(model_path),
            device=device,
            **hps,
        )
        self.model.fit(X_np, y_fit)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)

        if self.problem_type == "regression":
            out = np.asarray(self.model.predict(X), dtype=np.float32).reshape(-1)
            y_pred_proba = out * self._y_std + self._y_mean
        else:
            y_pred_proba = self.model.predict_proba(X)

        return self._convert_proba_to_unified_form(y_pred_proba)

    def _set_default_params(self):
        # TALENT's default TabSwift configuration (TALENT/model/methods/tabswift.py). The
        # framework seed is injected via ``seed_name`` into ``random_state``.
        default_params = {
            "n_estimators": 16,
            "norm_methods": ["none", "power"],
            "feat_shuffle_method": "latin",
            "class_shift": True,
            "outlier_threshold": 4.0,
            "softmax_temperature": 0.9,
            "average_logits": True,
            "use_hierarchical": True,
            "use_amp": True,
            "batch_size": 16,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def get_device(self) -> str:
        return self.model.device_.type if self.model is not None else "cpu"

    def _set_device(self, device: str):
        device = self.to_torch_device(device)
        self.model.device_ = device
        if self.model.model_ is not None:
            self.model.model_ = self.model.model_.to(device)

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
        """Sequential fold fitting avoids contention on the shared HF checkpoint cache.

        ``refit_folds=True`` matches the other TFM wrappers (TabICL, LimiX, TabPFN-3, ...):
        for an in-context-learning model, refitting one model on all data gives faster
        inference at similar quality to the bagged ensemble.
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        default_ag_args_ensemble.update(
            {
                "fold_fitting_strategy": "sequential_local",
                "refit_folds": True,
            },
        )
        return default_ag_args_ensemble

    @classmethod
    def _class_tags(cls) -> dict:
        # TODO: implement memory estimation and set to True
        return {"can_estimate_memory_usage_static": False}

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def prefetch_weights(cls) -> str:
        """Pre-download the TabSwift checkpoint from Hugging Face and return its local path.

        Used by the foundation-model pre-download scripts to warm the cache before parallel
        fit runs. Tries the local cache first so offline compute nodes skip the etag
        HEAD-request that ``hf_hub_download`` performs by default.
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            return hf_hub_download(
                repo_id=_DEFAULT_HF_REPO,
                filename=_DEFAULT_HF_FILENAME,
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            return hf_hub_download(repo_id=_DEFAULT_HF_REPO, filename=_DEFAULT_HF_FILENAME)
