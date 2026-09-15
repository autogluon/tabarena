"""OrionMSP's library seam for the shared-weights registry and the network builder.

The classifier ``tabtune.models.orionmsp_v15.sklearn.classifier.OrionMSPv15Classifier`` reads the
checkpoint inside ``fit`` (``_load_model``: ``torch.load``, ``OrionMSPv15(**config)``,
``load_state_dict``, ``eval``) and then moves the module to its device. :data:`SharedOrionMSPv15Classifier`
takes that one hook from :func:`tabarena.models._shared_estimators.derive_shared_estimator`: with a key
the network is the registry's module (the following ``model_.to(device_)`` is a no-op on a module already
on that device type), without one the library loader runs unchanged. :func:`build_network` is the loader
the registry runs: the same construction as ``_load_model``, so the module is bit-identical to a
per-child build.

Imports tabtune and torch at module level; ``model.py`` imports it lazily inside the methods that
need it so ``import tabarena.models`` stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tabtune.models.orionmsp_v15.model.embedding import ColEmbedding
from tabtune.models.orionmsp_v15.model.orionmsp_v15 import OrionMSPv15
from tabtune.models.orionmsp_v15.sklearn.classifier import OrionMSPv15Classifier

from tabarena.models._shared_estimators import derive_shared_estimator

__all__ = [
    "OrionMSPv15",
    "OrionMSPv15Classifier",
    "SharedOrionMSPv15Classifier",
    "build_network",
    "patch_col_embedder_pos_emb",
    "read_checkpoint",
]


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    """The checkpoint dict (``config`` and ``state_dict``) read the way the library reads it."""
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    assert "config" in checkpoint, "The checkpoint doesn't contain the model configuration."
    assert "state_dict" in checkpoint, "The checkpoint doesn't contain the model state."
    return checkpoint


def build_network(checkpoint_path: str | Path, device: str) -> OrionMSPv15:
    """Build the OrionMSP network for ``checkpoint_path`` on ``device``, in eval mode with gradients off.

    Mirrors ``OrionMSPv15Classifier._load_model`` followed by the ``model_.to(device_)`` call of
    ``fit``: the module is constructed on the CPU from the checkpoint's ``config``, every parameter
    and persistent buffer is overwritten by ``load_state_dict`` (strict, so nothing random
    survives; the ``col_embedding_seed`` buffer is part of the state dict and the rope tables are
    deterministic), then moved. The registry runs this under its random-state guard, so the random
    initialization that the load overwrites never advances a global generator.
    """
    checkpoint = read_checkpoint(checkpoint_path)
    network = OrionMSPv15(**checkpoint["config"])
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()
    network.requires_grad_(False)
    return network.to(torch.device(device))


def attach_network(estimator: Any, network: OrionMSPv15) -> None:
    """Set what the library's ``_load_model`` sets: the module and the checkpoint path it came from."""
    estimator.model_path_ = None if estimator.model_path is None else Path(estimator.model_path)
    estimator.model_ = network


SharedOrionMSPv15Classifier = derive_shared_estimator(
    OrionMSPv15Classifier, module=__name__, name="SharedOrionMSPv15Classifier", apply=attach_network
)


def _orionmsp_fixed_pos_emb(self, embeddings, feature_indices=None):
    """Layout-robust replacement for ``ColEmbedding._add_feature_pos_emb``.

    Upstream TabTune discriminates between ``(B, H+C, T, E)`` and ``(B, T, H+C, E)`` via
    ``shape[1] == reserve_cls_tokens + shape[2]``, which is only true when ``H == T``. For any other
    shape (for example 1 feature with multiple rows) it picks the wrong branch and crashes with a
    shape mismatch. The sole caller in ``_inference_forward`` always passes ``(B, H+C, T, E)``, so
    this applies that branch's logic unconditionally. The subspace branch draws its extra seeds
    from an explicit generator, never from the global one.
    """
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


def patch_col_embedder_pos_emb() -> None:
    """Install :func:`_orionmsp_fixed_pos_emb` on ``ColEmbedding`` at the class level; idempotent.

    Done at the class so picklable estimators inherit the fix without storing a closure on the
    instance, and so a network built by the warm-up and one built by a child run the same code.
    """
    if ColEmbedding._add_feature_pos_emb is _orionmsp_fixed_pos_emb:
        return
    ColEmbedding._add_feature_pos_emb = _orionmsp_fixed_pos_emb
