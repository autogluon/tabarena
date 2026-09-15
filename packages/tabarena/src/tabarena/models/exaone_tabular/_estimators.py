"""EXAONE Tabular's checkpoint plumbing and network build for the shared-weights registry.

The released checkpoints' Hub coordinates live in ``exaonetabular.presets`` (repo, file, revision and
an environment override per task), so the wrapper resolves them through :func:`resolve_checkpoint`
instead of a static ``CheckpointSpec``. The registry key holds the content-addressed blob path
(two snapshots with identical bytes share one network), while the library's checkpoint reader
dispatches on the ``.safetensors`` suffix that the blob path lacks, so :func:`checkpoint_file`
derives the extension-bearing alias the reader is handed. :func:`build_network` reproduces the
library's ``from_pretrained`` for one key.

This module imports ``exaonetabular`` (and with it torch) at import time, so ``model.py`` imports
it lazily inside the hooks that need it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from exaonetabular.checkpoint import load_checkpoint
from exaonetabular.classifier import EXAONETabularClassifier
from exaonetabular.presets import released_checkpoint
from exaonetabular.regressor import EXAONETabularRegressor
from exaonetabular.weights import resolve_weights

from tabarena.models import prefetch as _hub

if TYPE_CHECKING:
    import torch

    from tabarena.models._weights import WeightsKey

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointSource:
    """Where a released checkpoint's local file came from.

    Args:
        path: Absolute local path of the checkpoint file (the Hugging Face blob after symlink
            resolution, or the file an environment override points at). This is the registry key's
            ``checkpoint``: two snapshots with identical bytes resolve to the same blob.
        file: The path the library reads the checkpoint through. The library dispatches on the
            ``.safetensors`` suffix and the blob path has none, so this is the cache's snapshot link
            (or the environment override's own path) whenever that names the same file as ``path``,
            and ``path`` itself otherwise.
        is_default: Whether the file is the released checkpoint (the library then enforces the
            manifest's SHA-256 pin); False for an environment override, where the library only
            warns that the pin is not checked.
        repo_id: Hugging Face repository of the released checkpoint.
        filename: File name of the released checkpoint inside the repository.
        revision: Revision the released coordinates name (``"main"`` at the pinned library commit).
        commit: Commit sha of the cached snapshot the file was resolved through, when known.
    """

    path: str
    file: str
    is_default: bool
    repo_id: str
    filename: str
    revision: str
    commit: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Provenance without absolute host paths, for ``info["shared_weights"]`` and the registry entry."""
        return {
            "repo_id": self.repo_id,
            "filename": self.filename,
            "revision": self.revision,
            "commit": self.commit,
            "is_default": self.is_default,
        }


def _cached_snapshot_link(repo_id: str, filename: str, revision: str) -> str | None:
    """The snapshot link (``.../snapshots/<sha>/<filename>``) the cache serves ``filename`` through.

    Reads the local cache only (no network); ``None`` when the file is not cached or the client
    cannot tell.
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo_id, filename, revision=revision)
    except Exception:
        return None
    return cached if isinstance(cached, str) else None


def _snapshot_commit(repo_id: str, filename: str, revision: str) -> str | None:
    """Commit sha of the cached snapshot serving ``filename``; ``None`` when the cache cannot tell.

    The blob path the resolver returns carries no snapshot sha, and the metadata must record which
    snapshot of a mutable revision a run used.
    """
    link = _cached_snapshot_link(repo_id, filename, revision)
    return None if link is None else _hub.commit_from_snapshot_path(link)


def _readable_file(candidate: str | None, blob: str) -> str:
    """``candidate`` when it names the same file as ``blob`` (symlinks resolved), else ``blob``.

    The library's checkpoint reader dispatches on the ``.safetensors`` suffix and treats any other
    path as a torch archive, so the extension-less blob path must never reach it while an
    extension-bearing alias of the same file exists.
    """
    if candidate is not None and os.path.realpath(candidate) == os.path.realpath(blob):
        return candidate
    return blob


def checkpoint_file(checkpoint: Any, blob: str) -> str:
    """The path the library reads the released ``checkpoint`` (stored at ``blob``) through.

    Mirrors :func:`resolve_checkpoint` from a registry key alone: the environment override's own
    path when the override is set, otherwise the cached snapshot link; ``blob`` when neither names
    that file.
    """
    if os.environ.get(checkpoint.weights_env_var):
        candidate = resolve_weights(checkpoint).path
    else:
        candidate = _cached_snapshot_link(checkpoint.repo_id, checkpoint.filename, checkpoint.revision)
    return _readable_file(candidate, blob)


def resolve_checkpoint(task: str, *, prefer_local: bool = True, allow_download: bool = True) -> CheckpointSource:
    """Resolve one released checkpoint (``"classification"`` or ``"regression"``) to a local file.

    The Hub coordinates come from ``exaonetabular.presets`` rather than being hardcoded here, so a
    repo or revision bump in the library is picked up automatically. The library's environment
    override (``EXAONETABULAR_CLASSIFIER_WEIGHTS`` / ``EXAONETABULAR_REGRESSOR_WEIGHTS``) keeps its
    precedence: when set, the file is resolved through ``exaonetabular.weights.resolve_weights`` and
    reported as user-supplied.

    The released files are served from a mutable ``main`` revision and have been republished in
    place at least once, so a cache hit alone cannot be trusted forever. The two modes split that
    responsibility: with ``prefer_local=False`` (the head node's ``EXAONETabularModel.prefetch_weights``)
    the etag of ``main`` is revalidated online and the snapshot re-downloaded when the bytes changed;
    with ``prefer_local=True`` (warm-up, fit and load on compute nodes) the file is read through the
    cache's ``refs/main`` pointer, that is the snapshot the head node validated, without any HTTPS
    request. A node that was never prefetched falls back to the online call once when
    ``allow_download`` permits.

    Raises:
        tabarena.models.prefetch.WeightsUnavailableError: The file is not cached and
            ``allow_download`` is False.
    """
    checkpoint = released_checkpoint(task)
    if os.environ.get(checkpoint.weights_env_var):
        source = resolve_weights(checkpoint)
        path = str(Path(source.path).resolve())
        return CheckpointSource(
            path=path,
            file=_readable_file(str(source.path), path),
            is_default=False,
            repo_id=checkpoint.repo_id,
            filename=checkpoint.filename,
            revision=checkpoint.revision,
        )
    if prefer_local:
        path = _hub.resolve_hf_file(
            checkpoint.repo_id, checkpoint.filename, revision=checkpoint.revision, allow_download=allow_download
        )
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=checkpoint.repo_id, filename=checkpoint.filename, revision=checkpoint.revision)
        path = str(Path(path).resolve())
    link = _cached_snapshot_link(checkpoint.repo_id, checkpoint.filename, checkpoint.revision)
    return CheckpointSource(
        path=path,
        file=_readable_file(link, path),
        is_default=True,
        repo_id=checkpoint.repo_id,
        filename=checkpoint.filename,
        revision=checkpoint.revision,
        commit=_snapshot_commit(checkpoint.repo_id, checkpoint.filename, checkpoint.revision),
    )


def estimator_cls(task: str) -> type[EXAONETabularClassifier | EXAONETabularRegressor]:
    """The library estimator of ``task`` (``"classification"`` or ``"regression"``)."""
    return EXAONETabularRegressor if task == "regression" else EXAONETabularClassifier


def released_manifest(task: str, **overrides: Any):
    """The released checkpoint's manifest of ``task`` with the per-child runtime ``overrides`` applied."""
    return released_checkpoint(task).manifest.with_overrides(**overrides)


def build_network(key: WeightsKey) -> torch.nn.Module:
    """Build the network for ``key`` exactly as the library's ``from_pretrained`` does.

    The estimator constructor builds the module from the released manifest at ``key.dtype`` on
    ``key.device``; ``exaonetabular.checkpoint.load_checkpoint`` then runs the same structural
    validation (including its transient fp32 CPU layout model and a clone of every target tensor as
    rollback backup) and copies every state dict entry with ``copy_``, so the module is bit-identical
    to a per-child build. The file is read through the extension-bearing path
    :func:`checkpoint_file` derives from ``key.checkpoint``. ``verify_checksum`` follows the library:
    enforced for the released file, skipped with the library's warning for an environment override.

    The module is returned in eval mode. Its parameters keep ``requires_grad=True`` as the library
    leaves them (the wrapper declares ``freeze_parameters=False``): the attention forward picks its
    projection path by grad state and torch selects other CPU kernels for frozen weights, so
    ``requires_grad_(False)`` changes the float32 predictions in the last bits. Every library
    forward already runs under ``inference_mode`` / ``no_grad``, so the flag costs no gradient
    bookkeeping.
    """
    task = key.variant
    checkpoint = released_checkpoint(task)
    is_default = not os.environ.get(checkpoint.weights_env_var)
    file = checkpoint_file(checkpoint, key.checkpoint)
    manifest = checkpoint.manifest.with_overrides(compute_dtype=key.dtype)
    estimator = estimator_cls(task)(manifest, device=key.device)
    if not is_default:
        logger.warning(
            "using user-supplied weights at %s; released-checkpoint integrity pin not enforced (SHA-256 not checked)",
            file,
        )
    load_checkpoint(file, estimator.model, manifest, verify_checksum=is_default)
    network = estimator.model
    network.eval()
    return network
