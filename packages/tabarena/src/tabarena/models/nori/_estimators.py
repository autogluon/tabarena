"""Nori's library seam for the shared-weights registry: checkpoint resolution and the predictor build.

``NoriRegressor`` stores only the fitted context and builds its ``NoriPredictor`` lazily at the
first predict; the predictor reads the checkpoint, random-initializes and loads the network and
moves it to the device. The wrapper instead builds the predictor eagerly through the library's
``NoriPredictor(model=...)`` injection (:func:`build_predictor`), around the registry payload
:func:`load_network` builds once per key. The checkpoint coordinates come from the library's
variant registry (``synthefy_nori.hf``), so :func:`resolve_checkpoint` replaces a static
``CheckpointSpec``.

Every ``synthefy_nori`` import is inside a function: the wrapper's attach hook runs on load and on
device changes without the library imported at module level, and the convention tests replace
:func:`predictor_cls` with a stand-in.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tabarena.models import prefetch as _hub

if TYPE_CHECKING:
    import torch

    from tabarena.models._weights import WeightsKey

#: ``NoriPredictor`` constructor arguments the wrapper supplies itself rather than reading from the estimator.
_PREDICTOR_OWN_ARGS = frozenset({"self", "device", "model_path", "model"})


@dataclass(frozen=True)
class CheckpointSource:
    """Where a Nori checkpoint's local file came from.

    Args:
        path: Absolute local path of the checkpoint (the Hugging Face blob after symlink resolution,
            or the file a ``model_path`` hyperparameter points at).
        repo_id: Hugging Face repository the checkpoint was resolved from; ``None`` for a user file.
        filename: File name of the checkpoint inside the repository (or the user file's basename).
        revision: Revision asked for; ``None`` is the repository's default branch, which the cache
            serves through its ``refs/main`` pointer.
        commit: Commit sha of the cached snapshot serving the file, when the cache can tell.
        user_supplied: Whether the file came from the ``model_path`` hyperparameter.
    """

    path: str
    repo_id: str | None
    filename: str
    revision: str | None = None
    commit: str | None = None
    user_supplied: bool = False

    def to_metadata(self) -> dict[str, Any]:
        """Provenance without absolute host paths, for ``info["shared_weights"]`` and the registry entry."""
        return {
            "repo_id": self.repo_id,
            "filename": self.filename,
            "revision": self.revision,
            "commit": self.commit,
            "user_supplied": self.user_supplied,
        }


def _snapshot_commit(repo_id: str, filename: str) -> str | None:
    """Commit sha of the cached snapshot serving ``filename``; ``None`` when the cache cannot tell.

    Reads the local cache only (no network): the blob path the resolver returns no longer carries
    the sha, and the metadata must record which snapshot of the mutable default branch a run used.
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo_id, filename)
    except Exception:
        return None
    return _hub.commit_from_snapshot_path(cached) if isinstance(cached, str) else None


def resolve_checkpoint(
    *,
    model: str | None = None,
    model_path: str | None = None,
    token: str | bool | None = None,
    allow_download: bool = True,
) -> CheckpointSource:
    """Resolve the checkpoint a Nori configuration uses to a local file, the cache first.

    ``model_path`` wins and is returned as given (resolved to an absolute path), matching
    ``NoriRegressor``, which ignores ``model`` when a path is set. Otherwise the variant name is
    mapped to its repository through the library's own registry (``synthefy_nori.hf.NORI_MODELS``,
    ``None`` selecting the base checkpoint) and the file is read through
    :func:`tabarena.models.prefetch.resolve_hf_file`: a cache hit is served through the cache's
    ``refs/main`` pointer without any Hub request, so the two HEAD requests the library's
    ``download_checkpoint`` makes never land in a timer; a node that was never prefetched downloads
    once when ``allow_download`` permits. The head node's ``NoriModel.prefetch_weights`` stays
    online and revalidates the default branch, so a cache hit is the snapshot it validated. The
    returned path is the content-addressed blob (symlinks resolved), so two snapshots that share
    the same bytes share one registry entry.

    Raises:
        tabarena.models.prefetch.WeightsUnavailableError: The file is not cached and
            ``allow_download`` is False.
    """
    if model_path is not None:
        path = Path(model_path).resolve()
        return CheckpointSource(path=str(path), repo_id=None, filename=path.name, user_supplied=True)
    from synthefy_nori.hf import DEFAULT_CHECKPOINT_FILENAME, resolve_model_repo

    repo_id = resolve_model_repo(model)
    path = _hub.resolve_hf_file(repo_id, DEFAULT_CHECKPOINT_FILENAME, allow_download=allow_download, token=token)
    return CheckpointSource(
        path=path,
        repo_id=repo_id,
        filename=DEFAULT_CHECKPOINT_FILENAME,
        commit=_snapshot_commit(repo_id, DEFAULT_CHECKPOINT_FILENAME),
    )


def load_network(key: WeightsKey) -> torch.nn.Module:
    """Build the network for ``key`` exactly as ``NoriPredictor.__init__`` does.

    ``synthefy_nori.utils.loading.load_model`` reads the checkpoint on the CPU, builds the
    ``FeaturesTransformer`` from the embedded config with the library's default
    ``mask_prediction=False`` (random initialization first, then ``load_state_dict``) and puts it in
    eval mode; the module is then moved to ``key.device`` the way the predictor moves it before its
    first forward. The registry runs this under its random-state guard, so the random
    initialization never advances a global generator.
    """
    from synthefy_nori.utils.loading import load_model

    return load_model(model_path=key.checkpoint, mask_prediction=False).to(key.device)


def predictor_cls() -> type:
    """The library's ``NoriPredictor`` class; the one accessor the convention tests replace with a stand-in."""
    from synthefy_nori.inference.predictor import NoriPredictor

    return NoriPredictor


def predictor_kwargs(estimator: Any) -> dict[str, Any]:
    """The ``NoriPredictor`` constructor arguments ``NoriRegressor._get_predictor`` derives from the estimator.

    The library builds its predictor from the estimator attributes whose names match constructor
    parameters (``inference_config``, ``augmentations``, ``yj_skew_threshold``, ``quantile_collapse``,
    ``bar_temperature``, ``bar_point_estimator``, ``discrete_y_snap_max_unique``, ``memory_policy``
    in synthefy-nori 0.13.0), so the same intersection is taken here: the constructor's parameter
    names, minus ``device``, ``model_path`` and ``model`` (which the wrapper supplies), restricted to
    the estimator's scikit-learn parameters. A library release that adds a constructor-derived
    argument under its own name is picked up without a wrapper change; the ``models``-marked drift
    guard compares the result with the library's own call.
    """
    names = [name for name in inspect.signature(predictor_cls().__init__).parameters if name not in _PREDICTOR_OWN_ARGS]
    params = estimator.get_params(deep=False)
    return {name: getattr(estimator, name) for name in names if name in params}


def build_predictor(estimator: Any, *, device: str, checkpoint: str, network: torch.nn.Module) -> Any:
    """A ``NoriPredictor`` for ``estimator`` on ``device`` that uses ``network`` instead of loading the checkpoint.

    Mirrors ``NoriRegressor._get_predictor`` (``torch.device(device)`` plus :func:`predictor_kwargs`)
    with the library's supported ``model=`` injection, which skips ``load_model``. Constructing the
    predictor reads the inference-config JSON and builds the preprocessing pipeline objects, a
    data-independent millisecond cost; ``model_path`` is stored on the predictor for reference only.
    The constructor seeds Python's global ``random`` with the predictor seed (as the library does
    at its own first predict) and disables mixed precision on the CPU.
    """
    import torch

    return predictor_cls()(
        device=torch.device(device), model_path=checkpoint, model=network, **predictor_kwargs(estimator)
    )
