"""Causilo's library seam for the shared-weights registry.

``causilo.engine.Engine.fit`` reads the pinned checkpoint only when ``engine.model is None`` and
``causilo.estimators.fit_adapter`` reuses an estimator's existing ``_engine`` whenever its device
request matches, so a pre-built engine holding the registry's network is the library's own
injection seam. The two subclasses here add :meth:`_SharedNetworkMixin.attach_network`, which builds
that engine, and a ``__setstate__`` that restores the fitted context without reading the checkpoint:
the wrapper reattaches the registry entry afterwards (``SharedWeightsModelMixin._ensure_network``).
The library's ``__getstate__`` (``export_estimator``) already stores no weights, so the pickle of a
shared estimator is the library's pickle under another class path plus the saved model configuration
when the network is detached at export time.

The checkpoint coordinates (repository and pinned release commit) live in ``causilo.checkpoints``, so
:func:`resolve_checkpoint`, :func:`prefetch_checkpoints` and :func:`build_network` read them from the
library instead of repeating the literals.

Imports causilo (and torch) at module level, so ``model.py`` imports this module lazily.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from causilo.checkpoints import RELEASE_COMMIT, REPOSITORY, load_checkpoint
from causilo.engine import Engine
from causilo.estimators import CausiloClassifier, CausiloRegressor
from causilo.execution.memory import Stage
from causilo.execution.precision import stage_dtype
from causilo.model import Model, ModelConfig
from causilo.serialization import export_estimator, import_estimator, transfer_state, validate_cache_layout

from tabarena.models._shared_estimators import hf_checkpoint_source
from tabarena.models._shared_weights_model import ResolvedCheckpoint
from tabarena.models.prefetch import resolve_hf_snapshot

if TYPE_CHECKING:
    from causilo.execution.runner import ModelCache

    from tabarena.models._weights import WeightsKey

#: Causilo task of each registry variant; the variant doubles as the checkpoint folder name in the
#: ``nums-ai/causilo`` repository.
TASKS: dict[str, str] = {"classifier": "classification", "regressor": "regression"}
#: Files of one checkpoint folder.
CHECKPOINT_FILES: tuple[str, ...] = ("config.json", "model.safetensors")


def _checkpoint_files(variant: str) -> list[str]:
    return [f"{variant}/{name}" for name in CHECKPOINT_FILES]


def resolve_checkpoint(variant: str, *, allow_download: bool) -> ResolvedCheckpoint:
    """Resolve the checkpoint folder of ``variant`` at the pinned release commit, cache first.

    ``causilo.checkpoints.load_pretrained_model`` calls ``snapshot_download`` at the same commit with
    the same patterns; this goes through :func:`tabarena.models.prefetch.resolve_hf_snapshot` so a
    prefetched node answers from the cache without any Hub request and a node that was never
    prefetched downloads once when ``allow_download`` permits. The release commit is immutable, so
    a cached snapshot holding both files is always the right one.

    Raises:
        tabarena.models.prefetch.WeightsUnavailableError: The files are not cached and
            ``allow_download`` is False.
    """
    required = _checkpoint_files(variant)
    snapshot = resolve_hf_snapshot(
        REPOSITORY,
        revision=RELEASE_COMMIT,
        allow_patterns=required,
        required_files=required,
        allow_download=allow_download,
    )
    return ResolvedCheckpoint(
        path=str(Path(snapshot) / variant),
        source=hf_checkpoint_source(
            repo_id=REPOSITORY, filename=f"{variant}/model.safetensors", revision=RELEASE_COMMIT, path=snapshot
        ),
    )


def prefetch_checkpoints() -> list[str]:
    """Ensure both task checkpoints are cached at the pinned release commit; returns the folder paths.

    One snapshot call covers the classifier and the regressor folders so a run over all problem
    types is warmed before the jobs are dispatched; compute nodes then resolve the same snapshot
    cache first through :func:`resolve_checkpoint`.
    """
    files = [name for variant in TASKS for name in _checkpoint_files(variant)]
    snapshot = resolve_hf_snapshot(
        REPOSITORY, revision=RELEASE_COMMIT, allow_patterns=files, required_files=files, allow_download=True
    )
    return [str(Path(snapshot) / variant) for variant in TASKS]


def build_network(key: WeightsKey) -> Model:
    """Build the network for ``key`` exactly as the library does; the registry's loader.

    ``causilo.checkpoints.load_checkpoint`` validates the configuration hash and the safetensors
    metadata, builds the module under ``torch.device("meta")`` (so no random generator advances)
    and assigns the checkpoint tensors, returning it in eval mode; the task check and the move to
    the device are those of ``load_pretrained_model(...).to(device)``, which ``Engine.fit`` would
    otherwise run per child. The parameters keep ``requires_grad=True`` as the library leaves them
    (``freeze_parameters=False`` in the spec): with ``requires_grad=False`` torch picks other CPU
    kernels for the inference-mode forwards and the predictions differ in the last bits. Nothing in
    ``causilo`` trains or writes the module, so the flag is inert.
    """
    network = load_checkpoint(Path(key.checkpoint))
    if network.config.task != TASKS[key.variant]:
        raise ValueError("The official checkpoint has the wrong prediction task")
    return network.to(key.device).eval()


def network_device(network: Model) -> torch.device:
    """The device of the network's parameters."""
    return next(network.parameters()).device


def caches_to_device(caches: tuple[ModelCache, ...], task: str, device: torch.device) -> tuple[ModelCache, ...]:
    """Move fitted K/V caches to ``device`` with the stage precision the library restores them at.

    Mirrors the cache transfer of ``causilo.serialization.import_estimator``. Every benchmark
    config leaves ``use_kv_cache`` at its default ``False``, so the caches are ``None`` there and this
    runs only for user configs and unit tests.
    """
    column_dtype = stage_dtype(task, Stage.COLUMN, device)
    prediction_dtype = stage_dtype(task, Stage.PREDICTION, device)
    return tuple(
        replace(
            cache,
            columns=tuple(transfer_state(column, device, dtype=column_dtype) for column in cache.columns),
            prediction=transfer_state(cache.prediction, device, dtype=prediction_dtype),
        )
        for cache in caches
    )


def _cuda_requested(request: str) -> bool:
    return request != "auto" and torch.device(request).type == "cuda"


def _set_fitted_attributes(estimator: Any, engine: Engine) -> None:
    """The sklearn fitted attributes ``fit_adapter`` and ``import_estimator`` derive from the fitted context."""
    table = engine.require_state().dataset
    estimator.n_features_in_ = table.encoder.width
    if table.encoder.names is not None and all(isinstance(name, str) for name in table.encoder.names):
        estimator.feature_names_in_ = np.asarray(table.encoder.names, dtype=object)
    if engine.task == "classification":
        estimator.classes_ = table.target_encoder.classes_


class _SharedNetworkMixin:
    """Engine management shared by the two estimator subclasses; ``task`` is set per subclass."""

    task: ClassVar[str]

    def network_detached(self) -> bool:
        """Whether the estimator is fitted but its engine has no network (a restored pickle)."""
        engine = getattr(self, "_engine", None)
        return engine is not None and engine.model is None

    def attach_network(self, network: Model, *, device: torch.device | None = None) -> None:
        """Give the estimator an engine that runs on ``network``, which it must never move or train.

        Before a fit the engine is built for the estimator's ``device`` request exactly as
        ``fit_adapter`` would build it, so ``fit_adapter`` reuses it and ``Engine.fit`` skips its
        checkpoint load. After :meth:`__setstate__` the existing engine keeps its fitted context;
        with ``device`` given and of another type than the engine's, the context's caches are moved
        there first and the device fields are normalized to that device.

        Raises:
            RuntimeError: The network lives on another device type than the engine.
            ValueError: The network's configuration differs from the one the fitted context was
                saved with (the same check ``import_estimator`` performs).
        """
        engine = getattr(self, "_engine", None)
        if engine is None or (device is None and engine.device_request != self.device):
            engine = Engine(self.task, self.device)
        elif device is not None and device.type != engine.device.type:
            if engine.state is not None and engine.state.caches is not None:
                engine.state = replace(engine.state, caches=caches_to_device(engine.state.caches, self.task, device))
            engine.device = device
            engine.device_request = str(device)
            self.device = str(device)
        if network_device(network).type != engine.device.type:
            raise RuntimeError(
                f"The shared Causilo network lives on {network_device(network)} but the estimator runs on "
                f"{engine.device}; the registry key and the estimator device must agree."
            )
        saved_config = getattr(self, "_model_config", None)
        if saved_config is not None and network.config.record() != saved_config:
            raise ValueError("Saved cache does not match the model shape")
        engine.model = network
        self._engine = engine
        self._model_config = network.config.record()

    def __getstate__(self) -> dict:
        """The library's weightless payload; the model configuration survives a detached export."""
        saved = export_estimator(self)
        if saved["model_config"] is None and getattr(self, "_model_config", None) is not None:
            saved["model_config"] = self._model_config
        return saved

    def __setstate__(self, saved: dict) -> None:
        """Restore parameters and fitted context without reading the checkpoint.

        The version, schema, dependency and revision checks and the parameter restore are the
        library's (``import_estimator`` with the fitted context withheld). The fitted context is
        then installed into an engine with no network: caches are validated against the saved model
        configuration and moved to the engine's device, the sklearn fitted attributes are set, and
        the wrapper reattaches the registry network through :meth:`attach_network`.

        A pickle written by a CUDA fit records ``device="cuda"`` (the wrapper saves without a device
        round trip); on a host without CUDA the request falls back to the CPU, since ``Engine``
        rejects an unavailable explicit device, so the artifact stays loadable everywhere.
        """
        fitted = saved["fitted"]
        import_estimator(self, {**saved, "fitted": None})
        self._model_config = saved.get("model_config")
        if fitted is None:
            return
        request = self.device
        if _cuda_requested(request) and not torch.cuda.is_available():
            request = "cpu"
            self.device = request
        engine = Engine(saved["task"], request)
        config = ModelConfig(**saved["model_config"]["model"])
        validate_cache_layout(fitted, config)
        if fitted.caches is not None:
            fitted = replace(fitted, caches=caches_to_device(fitted.caches, engine.task, engine.device))
        engine.state = fitted
        self._engine = engine
        _set_fitted_attributes(self, engine)


class SharedCausiloClassifier(_SharedNetworkMixin, CausiloClassifier):
    """``CausiloClassifier`` running on a network attached through :meth:`attach_network`."""

    task: ClassVar[str] = "classification"


class SharedCausiloRegressor(_SharedNetworkMixin, CausiloRegressor):
    """``CausiloRegressor`` running on a network attached through :meth:`attach_network`."""

    task: ClassVar[str] = "regression"


def estimator_cls(variant: str, *, shared: bool) -> type:
    """The estimator class for ``variant``: the shared-network subclass with a key, else the library class.

    The library class stays for a fit that owns its network (sharing off, ``ag.save_pretrained_weights``):
    its ``__setstate__`` (``import_estimator``) reloads the checkpoint, so that pickle is self-contained.
    """
    classification = TASKS[variant] == "classification"
    if shared:
        return SharedCausiloClassifier if classification else SharedCausiloRegressor
    return CausiloClassifier if classification else CausiloRegressor
