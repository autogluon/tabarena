"""EXAONE Tabular's network load as a separable call, so one network per process serves every fit.

Developer fix. ``EXAONETabularClassifier.from_pretrained`` (and the regressor's) resolves the released
checkpoint, builds the network in the estimator constructor and loads the weights into it, all in one
classmethod; the constructor also accepts a prebuilt ``model``. :func:`load_network` is the loading half
of ``from_pretrained`` at the pinned library commit (``classifier.py`` and ``regressor.py``), and the
wrapper constructs the estimator around its result. The library could offer this itself, as a
``load_network(task, device, compute_dtype)`` next to ``from_pretrained``; until it does, re-diff this
function against ``from_pretrained`` whenever the library is bumped.

Imports ``exaonetabular`` (and with it torch) at module level; the wrapper imports it inside ``_fit``.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from exaonetabular.checkpoint import load_checkpoint
from exaonetabular.classifier import EXAONETabularClassifier
from exaonetabular.presets import released_checkpoint
from exaonetabular.regressor import EXAONETabularRegressor
from exaonetabular.weights import resolve_weights

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def estimator_cls(task: str) -> type[EXAONETabularClassifier | EXAONETabularRegressor]:
    """The library estimator of ``task`` (``"classification"`` or ``"regression"``)."""
    return EXAONETabularRegressor if task == "regression" else EXAONETabularClassifier


def released_manifest(
    task: str,
    *,
    ensemble_count: int | None = None,
    compute_dtype: str | None = None,
    seed: int | None = None,
    support_row_limit: int | None = None,
):
    """The released checkpoint's manifest of ``task`` with the runtime overrides ``from_pretrained`` accepts.

    ``support_row_limit`` replaces the runtime's cap on the in-context support rows (the estimators draw a
    seeded random subset above it); ``with_overrides`` does not expose it, so it is set on the runtime
    config directly.
    """
    manifest = released_checkpoint(task).manifest.with_overrides(
        ensemble_count=ensemble_count, compute_dtype=compute_dtype, seed=seed
    )
    if support_row_limit is not None:
        manifest = replace(manifest, runtime=replace(manifest.runtime, support_row_limit=support_row_limit))
    return manifest


def load_network(task: str, device: str, compute_dtype: str | None = None) -> torch.nn.Module:
    """The released network of ``task`` on ``device``, built and loaded as ``from_pretrained`` does.

    The wrapper's ``shared_weights`` loader, keyed on ``task``, ``compute_dtype`` (the dtype the
    network is built in) and ``device``. The steps are ``from_pretrained``'s: resolve the released
    checkpoint (the library's environment override keeps its precedence), build the module in the
    estimator constructor from the manifest at ``compute_dtype``, load the checkpoint into it with
    the SHA-256 pin enforced for the released file. ``ensemble_count`` and ``seed`` shape how the
    estimator runs, not the network, so they stay out of this function. Custom ``weights`` never reach
    it: the wrapper runs the library's ``from_pretrained`` for them.
    """
    checkpoint = released_checkpoint(task)
    manifest = released_manifest(task, compute_dtype=compute_dtype)
    source = resolve_weights(checkpoint)
    estimator = estimator_cls(task)(manifest, device=device)
    if not source.is_default:
        logger.warning(
            "using user-supplied weights at %s; released-checkpoint integrity pin not enforced (SHA-256 not checked)",
            source.path,
        )
    load_checkpoint(source.path, estimator.model, manifest, verify_checksum=source.is_default)
    return estimator.model
