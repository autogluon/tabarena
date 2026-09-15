"""TabICL's library seam for the shared-weights registry.

``TabICLClassifier`` and ``TabICLRegressor`` build their network inside ``_load_model``, which
``fit`` and ``__setstate__`` both call, and they pickle without it. The derived classes below take
that one hook from :func:`tabarena.models._shared_estimators.derive_shared_estimator`: with a key
the network is the registry's module, without one the library's own build runs. The module's
inference managers are reconfigured from the calling estimator's ``inference_config_`` on every
forward, so the children of a bag use the shared module strictly one after another (TabArena fits
and predicts them sequentially).

Imports tabicl and torch at module level; the wrapper imports this module from its fit and build
paths only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from tabicl import TabICLClassifier, TabICLRegressor
from tabicl._model.tabicl import TabICL

from tabarena.models._shared_estimators import derive_shared_estimator


@dataclass(frozen=True)
class SharedTabICLWeights:
    """One loaded TabICL network and the two attributes the library sets next to it.

    Args:
        module: The ``TabICL`` module in eval mode with gradients disabled, on the key's device.
        model_config: The checkpoint's ``config`` dict (what the library stores as ``model_config_``).
        model_path: The checkpoint file the module was built from (``model_path_``).
    """

    module: torch.nn.Module
    model_config: dict
    model_path: Path


def build_module(path: str | Path, device: str) -> SharedTabICLWeights:
    """Build one TabICL network from a checkpoint file exactly as the library's ``_load_model`` does.

    The same steps in the same order (``torch.load`` with ``map_location="cpu"`` and
    ``weights_only=True``, ``TabICL(**config)``, ``load_state_dict``, ``eval()``), then the move to
    ``device`` that the library's ``fit`` performs right after, so the parameters are bit-identical
    to a per-estimator build. Construction stays on the CPU (no meta device): ``RotaryEmbedding``
    registers non-persistent buffers that ``load_state_dict`` would leave uninitialized.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert "config" in checkpoint, "The checkpoint doesn't contain the model configuration."
    assert "state_dict" in checkpoint, "The checkpoint doesn't contain the model state."
    module = TabICL(**checkpoint["config"])
    module.load_state_dict(checkpoint["state_dict"])
    module.eval()
    module.to(device)
    return SharedTabICLWeights(module=module, model_config=checkpoint["config"], model_path=path)


def reset_rope_caches(module: torch.nn.Module) -> None:
    """Empty the rotary-embedding caches so a newly attached estimator starts like a fresh build.

    ``RotaryEmbedding`` memoizes its position tables in the non-persistent buffers ``cached_freqs``
    and ``cached_scales`` when ``interleaved`` is set (the v1.1 classifier checkpoint; the v2
    checkpoints never populate them). The tables are exact functions of the sequence length, so a
    stale cache would not change predictions; resetting them makes the shared module start every
    child from the state a fresh module has. Duck-typed on the buffer names so it needs no library
    import.
    """
    for submodule in module.modules():
        if hasattr(submodule, "cached_freqs"):
            submodule.cached_freqs = None
            submodule.cached_scales = None


def attach_shared_weights(estimator: TabICLClassifier | TabICLRegressor, weights: SharedTabICLWeights) -> None:
    """Set the three post-conditions of the library's ``_load_model`` from a shared payload."""
    estimator.model_ = weights.module
    estimator.model_config_ = weights.model_config
    estimator.model_path_ = weights.model_path
    reset_rope_caches(weights.module)


TabArenaTabICLClassifier = derive_shared_estimator(
    TabICLClassifier, module=__name__, name="TabArenaTabICLClassifier", apply=attach_shared_weights
)
TabArenaTabICLRegressor = derive_shared_estimator(
    TabICLRegressor, module=__name__, name="TabArenaTabICLRegressor", apply=attach_shared_weights
)


def estimator_cls(variant: str) -> type[TabArenaTabICLClassifier | TabArenaTabICLRegressor]:
    """The registry-aware estimator class for ``variant`` (``"classifier"`` or ``"regressor"``)."""
    return TabArenaTabICLClassifier if variant == "classifier" else TabArenaTabICLRegressor
