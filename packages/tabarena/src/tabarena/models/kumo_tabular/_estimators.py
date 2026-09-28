"""Kumo Tabular's network load as a separable call, so one network per process serves every fit.

Developer fix. ``sdm.models.KumoTabular.__init__`` (structured-data-models at commit ``ee6acd40``)
builds the network and loads the checkpoint inside the constructor, resolving it against the Hub tag
``v1.0.0``; it offers no loader call and no ``network=`` argument. :func:`load_network` is the loading
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

import sdm
import torch

HF_REPO_ID = "nvidia/Kumo-Tabular"
#: Commit of the Hub tag ``v1.0.0``, the revision the library itself loads. Pinned so a push to the
#: repo never silently changes the benchmarked weights.
HF_REVISION = "bd7fa122b516c7355583873ffcf38f5e79403ebd"
SIZES = ("small", "medium", "large")


def checkpoint_filename(task: str, size: str) -> str:
    """The checkpoint of ``task`` (``"classification"`` / ``"regression"``) and ``size`` in the Hub repo."""
    return f"{size}/{'regressor' if task == 'regression' else 'classifier'}.pt"


def download_checkpoint(task: str, size: str) -> Path:
    """Download (or find in the Hub cache) the pinned checkpoint and return its local path."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=HF_REPO_ID, filename=checkpoint_filename(task, size), revision=HF_REVISION))


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
    """The network a fit took, with the task and size its estimator is built for.

    A plain object, so the shared-weights pickle finds ``network`` inside it, leaves it out and puts the
    registry's network for the load device back. The estimator holds no state of its own beyond the
    network and takes milliseconds to build on the meta device, so :meth:`estimator` builds it per call.
    """

    task: str
    size: str
    network: torch.nn.Module

    def estimator(self) -> sdm.models.KumoTabular:
        return build_estimator(task=self.task, size=self.size, network=self.network)
