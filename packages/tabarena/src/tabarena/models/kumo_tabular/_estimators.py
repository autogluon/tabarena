"""Kumo Tabular's network load as a separable call, so one network per process serves every fit.

Developer fix. ``sdm.models.KumoTabular.__init__`` (structured-data-models at commit ``98f61289``)
builds the network and loads the checkpoint inside the constructor, resolving it against the Hub tag
``v1.0.1``; it offers no loader call and no ``network=`` argument. :func:`load_network` is the loading
half of that constructor (``KumoTabular._load_from_pretrained``) against the pinned commit
:data:`HF_REVISION`, and :func:`build_estimator` builds the estimator on the meta device around its
result, as NVIDIA's own TabArena adapter (``benchmark/tabular/model.py`` in the same repository) does.
The library could offer both itself, as a public ``load_network(task, size, device)`` and a ``network=``
constructor argument; until it does, re-diff this module against ``_load_from_pretrained`` whenever the
library is bumped.

The fitted wrapper keeps a :class:`FittedNetwork`, not the estimator: AutoGluon's shared-weights pickle
finds the network by identity in the fitted state but treats a ``torch.nn.Module`` as one unit, and the
estimator is a module holding the network as a submodule.

Imports ``sdm`` (and with it torch) at module level; the wrapper imports this module inside ``_fit``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import sdm
import torch

if TYPE_CHECKING:
    from sdm.cache import Cache

HF_REPO_ID = "nvidia/Kumo-Tabular"
#: Commit of the Hub tag ``v1.0.1``, the revision the library itself loads: the ``v1.0.0`` checkpoints plus the
#: ``config.json`` the Hub's download statistics count. Pinned so a push to the repo never silently changes the
#: benchmarked weights.
HF_REVISION = "b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f"
SIZES = ("small", "medium", "large")


def checkpoint_filename(task: str, size: str) -> str:
    """The checkpoint of ``task`` (``"classification"`` / ``"regression"``) and ``size`` in the Hub repo."""
    return f"{size}/{'regressor' if task == 'regression' else 'classifier'}.pt"


def download_checkpoint(task: str, size: str) -> Path:
    """Download (or find in the Hub cache) the pinned checkpoint and return its local path.

    Through the library's own helper with ``_load_from_pretrained``'s arguments: a checkpoint not yet cached first
    fetches the repository's ``config.json``, so the download is counted.
    """
    from sdm.models._huggingface import download_checkpoint as sdm_download_checkpoint

    path = sdm_download_checkpoint(
        repo_id=HF_REPO_ID,
        filename=checkpoint_filename(task, size),
        revision=HF_REVISION,
        config_filename="config.json",
    )
    return Path(path)


def load_network(task: str, size: str, device: str) -> torch.nn.Module:
    """The pretrained network of ``task`` and ``size`` on ``device``, in eval mode.

    The wrapper's ``shared_weights`` loader, keyed on ``task``, ``size`` and ``device``. The steps are
    ``_load_from_pretrained``'s: build the module on the meta device, then assign the checkpoint's
    tensors, loaded straight onto ``device``.
    """
    network = sdm.models.KumoTabular(task=task, size=size, pretrained=False, device="meta").models[task]
    state = torch.load(download_checkpoint(task, size), map_location=device, weights_only=True)
    network.load_state_dict(state, assign=True)
    return network.eval()


def build_estimator(task: str, size: str, network: torch.nn.Module) -> sdm.models.KumoTabular:
    """A ``KumoTabular`` estimator of ``task`` that runs ``network``."""
    estimator = sdm.models.KumoTabular(task=task, size=size, pretrained=False, device="meta")
    estimator.models[task] = network
    return estimator.eval()


@dataclass
class FittedNetwork:
    """Shared checkpoint weights and a fold-local fitted SDM cache, or the raw context when the cache is off.

    A plain object, so the shared-weights pickle finds ``network`` inside it, leaves it out and puts the
    registry's network for the load device back. KV tensors stay on the CPU; SDM stages each estimator
    batch on the GPU during prediction. Fitted processors move with the prediction device. Without the
    cache, ``context`` holds the context features, targets and per-member subsample index (``None`` when
    the context fits) for the library's stateless forward. Both sit here rather than on the wrapper, so
    a bagged fold model that AutoGluon drops after its out-of-fold prediction (``model = None``) drops them.
    """

    task: str
    size: str
    network: torch.nn.Module
    cache: Cache | None = None
    context: tuple | None = None

    def estimator(self) -> sdm.models.KumoTabular:
        estimator = build_estimator(task=self.task, size=self.size, network=self.network)
        if self.cache is not None:
            self.move_processors(next(self.network.parameters()).device)
            # SDM has no public fitted-state export/import API. Keep this seam pinned with SDM.
            estimator._cache = self.cache
        return estimator

    def move_processors(self, device: torch.device | str) -> None:
        """Move fitted processors without moving the CPU-offloaded attention cache."""
        if self.cache is not None:
            recipe = self.cache["recipe_execution"].recipe
            for processor in (recipe.features, recipe.target, recipe.output):
                processor.to(device)


def available_memory(device: torch.device) -> int:
    """Available CUDA bytes, including this process's reusable allocator blocks."""
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reusable = torch.cuda.memory_reserved(device) - allocated
    limit = total * torch.cuda.get_per_process_memory_fraction(device) - allocated
    return max(0, int(min(free + reusable, limit)))
