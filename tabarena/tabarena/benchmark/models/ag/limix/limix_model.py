from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import (
    BINARY,
    MULTICLASS,
)
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_VENDOR_DIR = Path(__file__).resolve().parent / "_vendor"
_CONFIG_DIR = _VENDOR_DIR / "config"

_DEFAULT_HF_REPO = "stableai-org/LimiX-16M"
_DEFAULT_HF_FILENAME = "LimiX-16M.ckpt"
_DEFAULT_CLS_CONFIG = "cls_default_16M_retrieval.json"
_DEFAULT_REG_CONFIG = "reg_default_16M_retrieval.json"


class LimiXModel(AbstractTorchModel):
    """LimiX: Unleashing Structured-Data Modeling Capability for Generalist Intelligence.

    Paper: https://arxiv.org/abs/2509.03505
    Codebase: https://github.com/limix-ldm-ai/LimiX
    License: Apache-2.0

    Upstream is not pip-installable, so the inference-time sources are
    vendored under ``_vendor/`` next to this file.
    """

    ag_key = "TA-LIMIX"
    ag_name = "TA-LimiX"
    ag_priority = 100
    seed_name = "random_state"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator: LabelEncoderFeatureGenerator | None = None
        self._cat_indices: list[int] | None = None
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._y_mean: float | None = None
        self._y_std: float | None = None

    def _preprocess(self, X: pd.DataFrame, *, is_train: bool = False, **kwargs) -> np.ndarray:
        """We preprocess for LimiX to ensure categorical features are passed as correct dtypes to LimiX."""
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

        available_num_gpus = ResourceManager.get_gpu_count_torch(cuda_only=True)
        if num_gpus > available_num_gpus:
            raise AssertionError(
                f"Fit specified to use {num_gpus} GPU, but only {available_num_gpus} "
                "CUDA GPUs are available. Please activate CUDA or switch to CPU usage.",
            )
        device_str = "cuda" if num_gpus != 0 else "cpu"
        if device_str == "cuda" and not torch.cuda.is_available():
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        from tabarena.benchmark.models.ag.limix._vendor.inference.predictor import LimiXPredictor

        hps = self._get_model_params()
        random_state = hps.pop(self.seed_name, 0)
        model_path = hps.pop("model_path", None) or _download_default_checkpoint()
        inference_config = hps.pop("inference_config", None)
        if inference_config is None:
            cfg_filename = _DEFAULT_CLS_CONFIG if self.problem_type in ["binary", "multiclass"] else _DEFAULT_REG_CONFIG
            inference_config = _load_bundled_config(cfg_filename)

        X_np = self.preprocess(X, y=y, is_train=True)
        y_np = np.asarray(y.to_numpy(), dtype=np.float32 if self.problem_type == "regression" else None)

        if self.problem_type == "regression":
            # Following all documentation and examples, we scale at this level and inverse scale later.
            self._y_mean = float(y_np.mean())
            self._y_std = float(y_np.std()) or 1.0
            y_fit = (y_np - self._y_mean) / self._y_std
        else:
            y_fit = y_np

        self.model = LimiXPredictor(
            device=torch.device(device_str),
            model_path=str(model_path),
            inference_config=inference_config,
            categorical_features_indices=self._cat_indices or None,
            seed=int(random_state),
            **hps,
        )
        # Save into model so pickling works better
        self.model._X_train = X_np
        self.model._y_train = y_fit

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """LimiX does not support a sklearn API, thus, we have to call the forward pass this way."""
        import torch

        X = self.preprocess(X, **kwargs)

        # Forward pass call via LimiX code
        task_type = "Classification" if self.problem_type in [BINARY, MULTICLASS] else "Regression"
        out = self.model.predict(self.model._X_train, self.model._y_train, X, task_type=task_type)
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
        out = np.asarray(out)

        if task_type == "Regression":
            out = out * self._y_std + self._y_mean
        y_pred_proba = out

        return self._convert_proba_to_unified_form(y_pred_proba)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def get_device(self) -> str:
        return self.model.device.type if self.model is not None else "cpu"

    def _set_device(self, device: str):
        import torch

        device = torch.device(device)
        self.model.device = device
        if self.model.model is not None:
            self.model.model.to(device)

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
        """Sequential fold fitting avoids contention on the shared HF checkpoint cache."""
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        default_ag_args_ensemble.update(
            {
                "fold_fitting_strategy": "sequential_local",
                "refit_folds": True,
            }
        )
        return default_ag_args_ensemble

    def _get_default_auxiliary_params(self) -> dict:
        """We set the default to 100k to try to run on all of TabArena.

        Note, all examples of LimiX code itself says one should skip above 50k.
        """
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                # "max_rows": 50_000, # Technically from LimiX
                "max_classes": 10,
            }
        )
        return default_auxiliary_params

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}


def _load_bundled_config(filename: str) -> list:
    cfg_path = _CONFIG_DIR / filename
    with cfg_path.open("r") as f:
        return json.load(f)


def _download_default_checkpoint() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=_DEFAULT_HF_REPO, filename=_DEFAULT_HF_FILENAME)
