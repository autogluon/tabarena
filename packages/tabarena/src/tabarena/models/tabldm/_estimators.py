"""TabLDM's library seam for the shared-weights registry.

``TabLDMEnhancedClassifier`` and ``TabLDMEnhancedRegressor`` build their network inside
``_load_model``, which ``fit`` and ``__setstate__`` both call right after ``_resolve_device``, and
they pickle without it (the library's ``__getstate__`` drops ``model_``, ``device_``,
``inference_config_`` and ``model_path_``). The derived classes below take that one hook from
:func:`tabarena.models._shared_estimators.derive_shared_estimator`: with a key the network is the
registry's module, without one the library's own build runs. The library's following
``self.model_.to(self.device_)`` is a no-op on a module already on that device type.

Imports tabldm and torch at module level; the wrapper imports this module from its fit and build
paths only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from tabldm import TabLDMEnhancedClassifier, TabLDMEnhancedRegressor

from tabarena.models._shared_estimators import derive_shared_estimator

if TYPE_CHECKING:
    from tabarena.models._weights import WeightsKey

#: The library estimator that owns each checkpoint variant.
LIBRARY_ESTIMATORS: dict[str, type] = {
    "classifier": TabLDMEnhancedClassifier,
    "regressor": TabLDMEnhancedRegressor,
}


@dataclass(frozen=True)
class TabLDMNetwork:
    """The registry payload: what the library's ``_load_model`` sets on an estimator, stored once per key.

    Args:
        module: The built network in eval mode on the key's device (``model_``).
        config: The checkpoint's ``config`` dict (``model_config_``); each child gets its own copy.
        path: The checkpoint file the network was built from (``model_path_``).
    """

    module: torch.nn.Module
    config: dict[str, Any]
    path: Path


def build_network(key: WeightsKey) -> TabLDMNetwork:
    """Build the network for ``key`` through the library's own ``_load_model``; the registry's loader.

    A bare library estimator is pointed at the resolved checkpoint file (``model_path``) with
    downloads disabled, since the key derivation already resolved the file local-first. The library
    then runs its unmodified construction path (``torch.load`` to the CPU, ``TabLDMSparseMoE`` plus
    ``ColEmbeddingDualStream``, ``drop_dense_ffn``, ``load_state_dict`` with its mismatch check,
    ``eval()``), so the module is the one every child would otherwise build for itself. It is moved
    to ``key.device`` once; every forward in the library runs under ``torch.no_grad`` and never
    mutates the module, so the holders share it by reference. The parameters keep the library's
    ``requires_grad`` flags (``freeze_parameters=False`` in the spec): ``requires_grad_(False)``
    selects different CPU kernels and moves float32 probabilities by about 3e-7, and predictions
    must stay bit-identical to the unshared path. The registry runs this under its random-state
    guard, so the construction-time initializers advance no global generator.
    """
    estimator = LIBRARY_ESTIMATORS[key.variant](model_path=key.checkpoint, allow_auto_download=False, device=key.device)
    estimator._load_model()
    module = estimator.model_.to(torch.device(key.device))
    module.eval()
    return TabLDMNetwork(module=module, config=dict(estimator.model_config_), path=Path(key.checkpoint))


def attach_network(estimator: Any, network: TabLDMNetwork) -> None:
    """Set the three post-conditions of the library's ``_load_model`` from a shared payload."""
    estimator.model_ = network.module
    estimator.model_config_ = dict(network.config)
    estimator.model_path_ = network.path


SharedNetworkTabLDMClassifier = derive_shared_estimator(
    TabLDMEnhancedClassifier, module=__name__, name="SharedNetworkTabLDMClassifier", apply=attach_network
)
SharedNetworkTabLDMRegressor = derive_shared_estimator(
    TabLDMEnhancedRegressor, module=__name__, name="SharedNetworkTabLDMRegressor", apply=attach_network
)


def estimator_cls(variant: str) -> type:
    """The registry-aware estimator class for ``variant`` (``"classifier"`` or ``"regressor"``)."""
    return SharedNetworkTabLDMClassifier if variant == "classifier" else SharedNetworkTabLDMRegressor
