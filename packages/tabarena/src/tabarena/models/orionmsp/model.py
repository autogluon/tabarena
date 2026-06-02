from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger(__name__)


_HF_REPO_ID = "Lexsi/Orion-MSP"
_DEFAULT_CHECKPOINT_FILE = "OrionMSP-classifier-v1.5-202603.ckpt"


class OrionMSPModel(AbstractTorchModel):
    """Orion-MSP v1.5: Multi-Scale Sparse Attention for Tabular In-Context Learning.

    We have to use the code from TabTune, as the standalone package does not support the newest
    checkpoints. The standalone package is hardcoded to 1.0 checkpoints.

    Codebase: https://github.com/Lexsi-Labs/Orion-MSP
    Hugging Face: https://huggingface.co/Lexsi/Orion-MSP
    TabTune (wrapper used here): https://github.com/Lexsi-Labs/TabTune
    Paper: Orion-MSP: Multi-Scale Sparse Attention for Tabular In-Context Learning
        (https://arxiv.org/abs/2511.02818)
    Authors: Mohamed Bouadi, Pratinav Seth, Aditya Tanna, Vinay Kumar Sankarapu
    License: MIT
    """

    ag_key = "TA-ORION-MSP"
    ag_name = "TA-OrionMSP"
    ag_priority = 65
    seed_name = "random_state"

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
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

        from tabtune.models.orionmsp_v15.sklearn.classifier import (
            OrionMSPv15Classifier,
        )

        # See `_patch_col_embedder_pos_emb_class` docstring: works around an
        # upstream shape-discriminator bug that breaks inference whenever H != T.
        _patch_col_embedder_pos_emb_class()

        if self.problem_type in ["binary", "multiclass"]:
            model_cls = OrionMSPv15Classifier
        else:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")

        hps = self._get_model_params()

        # Pre-resolve the checkpoint locally so we skip the network entirely when
        # it's already cached. TabTune's loader will then read directly from
        # `model_path` instead of re-querying HuggingFace.
        allow_download = hps.get("allow_auto_download", True)
        if hps.get("model_path") is None:
            hps["model_path"] = _resolve_checkpoint(
                filename=hps.get("checkpoint_version", _DEFAULT_CHECKPOINT_FILE),
                allow_auto_download=allow_download,
            )

        # Needs up to 400GB VRAM for datasets with 1k features.
        # Adjust batch size as needed.
        if X.shape[1] > 500:
            hps["batch_size"] = 1  # avoid OOM for wide datasets; can be slow but is a fallback

        self.model = model_cls(
            **hps,
            device=device,
        )

        X = self.preprocess(X, y=y)
        self.model = self.model.fit(
            X=X,
            y=y,
        )

    def _set_default_params(self):
        default_params = {
            "checkpoint_version": _DEFAULT_CHECKPOINT_FILE,
            "allow_auto_download": True,
            # AMP introduces ~1e-4 fp16 jitter between single-row and batch
            # predict_proba calls, which breaks AutoGluon's determinism check.
            # "use_amp": False, # disabled for now due to VRAM issues
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass"]

    def get_device(self) -> str:
        return self.model.device

    def _set_device(self, device: str):
        self.model.device = device
        if hasattr(self.model, "to"):
            self.model.to(device)

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

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """Set fold_fitting_strategy to sequential_local,
        as parallel folding crashes if model weights aren't pre-downloaded.
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
            "fold_fitting_strategy": "sequential_local",
            "refit_folds": True,
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    @classmethod
    def _class_tags(cls) -> dict:
        # TODO: support memory estimate!
        return {"can_estimate_memory_usage_static": False}

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}


def _resolve_checkpoint(filename: str, allow_auto_download: bool = True) -> str:
    """Return a local path for `filename`, downloading from HF only if not cached."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return hf_hub_download(repo_id=_HF_REPO_ID, filename=filename, local_files_only=True)
    except LocalEntryNotFoundError:
        if not allow_auto_download:
            raise ValueError(
                f"Checkpoint '{filename}' not cached locally and "
                f"allow_auto_download=False. Pre-download it via "
                f"`prefetch_weights(filename='{filename}')` or set "
                f"allow_auto_download=True.",
            ) from None
        logger.info(
            "Orion-MSP checkpoint '%s' not cached; downloading from %s.",
            filename,
            _HF_REPO_ID,
        )
        return hf_hub_download(repo_id=_HF_REPO_ID, filename=filename)


def prefetch_weights(filename: str = _DEFAULT_CHECKPOINT_FILE) -> str:
    # Defaults to the latest v1.5 checkpoint we use in TabArena. Skips the
    # network call if the file is already present in the HF cache.
    return _resolve_checkpoint(filename=filename, allow_auto_download=True)


def _orionmsp_fixed_pos_emb(self, embeddings, feature_indices=None):
    """Layout-robust replacement for `ColEmbedding._add_feature_pos_emb`.

    Upstream TabTune (v0.1.16) discriminates between `(B, H+C, T, E)` and
    `(B, T, H+C, E)` via `shape[1] == reserve_cls_tokens + shape[2]`, which
    is only true when `H == T`. For any other shape (e.g. 1 feature with
    multiple rows) it picks the wrong branch and crashes with a shape
    mismatch. The sole caller in `_inference_forward` always passes
    `(B, H+C, T, E)`, so we unconditionally apply that branch's logic.
    """
    import torch
    from torch.nn import functional as F

    if self.feature_pos_emb is None or embeddings.dim() != 4:
        return embeddings

    _B, HC, _T, _E = embeddings.shape
    H = HC - self.reserve_cls_tokens
    if H <= 0:
        return embeddings

    if self.feature_pos_emb == "subspace":
        base_seed = self.col_embedding_seed.to(embeddings.device)
        if base_seed.shape[0] < H:
            generator = torch.Generator(device=embeddings.device).manual_seed(42)
            additional_seed = torch.randn(
                H - base_seed.shape[0],
                base_seed.shape[1],
                device=embeddings.device,
                dtype=base_seed.dtype,
                generator=generator,
            )
            full_seed = torch.cat([base_seed, additional_seed], dim=0)
        else:
            full_seed = base_seed[:H]

        proj = self.feature_pos_proj
        if full_seed.device != proj.weight.device:
            W = proj.weight.to(full_seed.device)
            b = proj.bias.to(full_seed.device) if proj.bias is not None else None
            pos_emb = F.linear(full_seed, W, b)
        else:
            pos_emb = proj(full_seed)
        embeddings[:, self.reserve_cls_tokens :, :, :] += pos_emb[None, :, None, :]
    elif self.feature_pos_emb == "learned":
        if feature_indices is not None:
            idx = feature_indices.to(device=embeddings.device).long()
        else:
            idx = torch.arange(H, device=embeddings.device).long()
        emb_w = self.feature_pos_embeddings.weight
        if idx.device != emb_w.device:
            pos_emb = F.embedding(idx, emb_w.to(idx.device))
        else:
            pos_emb = self.feature_pos_embeddings(idx)
        embeddings[:, self.reserve_cls_tokens :, :, :] += pos_emb[None, :, None, :]
    return embeddings


_ORIONMSP_POS_EMB_PATCHED = False


def _patch_col_embedder_pos_emb_class() -> None:
    """Patch `ColEmbedding._add_feature_pos_emb` at the class level (idempotent).

    Done at the class so picklable model instances inherit the fix without
    storing a closure on the instance.
    """
    global _ORIONMSP_POS_EMB_PATCHED
    if _ORIONMSP_POS_EMB_PATCHED:
        return
    from tabtune.models.orionmsp_v15.model.embedding import ColEmbedding

    ColEmbedding._add_feature_pos_emb = _orionmsp_fixed_pos_emb
    _ORIONMSP_POS_EMB_PATCHED = True
