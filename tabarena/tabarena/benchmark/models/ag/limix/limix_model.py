from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import (
    BINARY,
    MULTICLASS,
)
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel
from torch import nn

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

    subsample_train_n_rows: int = 75_000
    """Empirically, even with 140 GB of VRAM available we still hit OOM on LimiX's retrieval + clustering inference
    path on TabArena-scale datasets, so subsampling is the only reliable lever to keep it running.
    We-sub-sample datasets above 75k rows to 50k rows following the LimiX documentation examples."""
    batch_test_n_rows: int = 5_000
    """We batch forward passes with more than 10k test rows."""

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
            # Stats are computed on the full y before any subsampling so inverse scaling matches the
            # original target distribution.
            self._y_mean = float(y_np.mean())
            self._y_std = float(y_np.std()) or 1.0
            y_fit = (y_np - self._y_mean) / self._y_std
        else:
            y_fit = y_np

        # Cap n_train to keep the LimiX inference pipeline within VRAM.
        # Empirically, even with 140 GB of VRAM available we still hit this on
        # datasets at the TabArena scale, so subsampling is the only reliable lever.
        # LimiX's own documentation / examples flag >50k rows as out-of-distribution
        # for this model (see https://github.com/limix-ldm-ai/LimiX/blob/main/inference_classifier.py#L108-L110).
        if X_np.shape[0] >= self.subsample_train_n_rows:
            n_full = X_np.shape[0]
            target = 50_000
            if self.problem_type in [BINARY, MULTICLASS]:
                from sklearn.model_selection import train_test_split

                try:
                    X_np, _, y_fit, _ = train_test_split(
                        X_np,
                        y_fit,
                        train_size=target,
                        stratify=y_fit,
                        random_state=int(random_state),
                    )
                except ValueError:
                    # Stratification fails on classes with too few samples; fall back to random.
                    rng = np.random.default_rng(int(random_state))
                    idx = rng.choice(n_full, size=target, replace=False)
                    X_np, y_fit = X_np[idx], y_fit[idx]
            else:
                rng = np.random.default_rng(int(random_state))
                idx = rng.choice(n_full, size=target, replace=False)
                X_np, y_fit = X_np[idx], y_fit[idx]
            logger.log(
                20,
                f"LimiX: subsampling train from {n_full} to {target} rows to bound VRAM at predict time",
            )

        self.model = LimiXPredictor(
            device=torch.device(device_str),
            model_path=str(model_path),
            inference_config=inference_config,
            categorical_features_indices=self._cat_indices or None,
            seed=int(random_state),
            **hps,
        )
        # See `_NaNCleanEncoder` docstring for why this wrap is needed. We have to wrap
        # every loaded copy of the FeaturesTransformer, not just `LimiXPredictor.model`:
        # each `InferenceAttentionMap` step in `preprocess_pipelines` calls
        # `load_model(self.model_path)` in its own `__init__` and holds its own model
        # instance, used to compute sample-attention scores for retrieval. Without
        # wrapping those too, the very first attention-map pass at
        # `_vendor/inference/inference_method.py:309` still hits the NaN guard.
        from tabarena.benchmark.models.ag.limix._vendor.inference.inference_method import InferenceAttentionMap

        self.model.model.encoder_x = _NaNCleanEncoder(self.model.model.encoder_x)
        for pipeline in self.model.preprocess_pipelines:
            for step in pipeline:
                if isinstance(step, InferenceAttentionMap):
                    step.model.encoder_x = _NaNCleanEncoder(step.model.encoder_x)
        # Save into model so pickling works better
        self.model._X_train = X_np
        self.model._y_train = y_fit

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """LimiX does not support a sklearn API, thus, we have to call the forward pass this way."""
        import torch

        X = self.preprocess(X, **kwargs)

        # Forward pass call via LimiX code
        task_type = "Classification" if self.problem_type in [BINARY, MULTICLASS] else "Regression"

        # Chunk the test set: a single forward pass over (n_train + n_test) rows can blow
        # past available VRAM on large datasets and surface as cudaErrorInvalidConfiguration.
        chunk_size = self.batch_test_n_rows
        n_test = X.shape[0]
        chunks = []
        for start in range(0, n_test, chunk_size):
            chunk_out = self.model.predict(
                self.model._X_train,
                self.model._y_train,
                X[start : start + chunk_size],
                task_type=task_type,
            )
            # LimiX runs under autocast, so outputs can come back in fp16. Promote to
            # fp32 here so downstream math (e.g. regression `out * self._y_std`) cannot
            # overflow fp16's ~65504 max on large-target regression problems.
            if isinstance(chunk_out, torch.Tensor):
                chunk_out = chunk_out.detach().to(torch.float32).cpu().numpy()
            else:
                chunk_out = np.asarray(chunk_out, dtype=np.float32)
            chunks.append(chunk_out)
        out = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

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

    @classmethod
    def download_model(cls) -> str:
        """Pre-download the default LimiX checkpoint from Hugging Face.

        Returns the local cache path. Used by the foundation-model pre-download
        scripts to warm the cache before parallel fit runs. We try the local
        cache first so offline compute nodes (no internet / proxy timeouts)
        skip the HEAD-request-for-etag that ``hf_hub_download`` performs by
        default.
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


def _load_bundled_config(filename: str) -> list:
    cfg_path = _CONFIG_DIR / filename
    with cfg_path.open("r") as f:
        return json.load(f)


def _download_default_checkpoint() -> str:
    return LimiXModel.download_model()


class _NaNCleanEncoder(nn.Module):
    """Wrap LimiX's ``encoder_x`` so any NaN/inf it emits is sanitized to 0.

    Why: the bundled LimiX 16M checkpoint's preprocess pipeline starts with a
    ``NanEncoder`` (`_vendor/model/encoders.py:361`) that replaces NaN cells in
    ``x`` with the per-column mean computed over the *train portion only*
    (``calc_mean(x[:, :eval_pos, :], dim=1)``). LimiX's retrieval + clustering
    path (`_vendor/inference/inference_method.py`) shards inference into small
    train clusters per test group, and on datasets with heavy missingness the
    selected train rows for a given cluster can end up entirely NaN on some
    column. The per-column "mean" is then itself NaN, the imputation step
    substitutes NaN for NaN, and the NaN propagates through ``process_4_x`` and
    ``encoder_x`` until ``transformer.py:194`` raises:

        ValueError: embedded_all contains NaN values; please add a NanEncoder
        in the encoder

    Sanitizing here is the most surgical place — it catches NaN regardless of
    which upstream stage produced it, without modifying vendor code and without
    blanket-imputing the raw input (the model handles NaN correctly on most
    datasets and we don't want to overwrite that behavior).
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        # Idempotent: collapse nested wraps so re-applying is safe.
        if isinstance(inner, _NaNCleanEncoder):
            inner = inner.inner
        self.inner = inner

    def forward(self, x):
        out = self.inner(x)
        if isinstance(out, dict) and isinstance(out.get("data"), torch.Tensor):
            out["data"] = torch.nan_to_num(out["data"], nan=0.0, posinf=0.0, neginf=0.0)
        return out