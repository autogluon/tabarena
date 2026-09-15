"""Shared pretrained weights for foundation-model wrappers: one mixin, one declarative spec.

A bagged fit of an in-context model builds the same frozen network once per fold child and once
more for the refit child, inside the timed fit. :class:`SharedWeightsModelMixin` moves that work
into the process-wide registry (:mod:`tabarena.models._weights`): the untimed warm-up primes one
network per checkpoint, every child attaches the registry object, pickles without it and takes it
back on load. A wrapper configures the mixin through a :class:`SharedWeightsSpec` and writes one
build hook; everything else (key derivation, the weightless pickle, device swaps, memory accounting,
the metadata block, prefetching) is provided here and is not overridden.

How to make a new foundation model share its weights
-----------------------------------------------------

1. Pick the mode. ``mode="module"`` (the default) shares the built ``torch.nn.Module`` (or a
   small container holding it) by reference: the estimator never mutates the weights, every child
   points at the same object and the child pickles weightless. ``mode="state_dict"`` is for
   fine-tuning models: the registry caches a read-only CPU state dict and every child copies it into
   its own trainable module through :func:`tabarena.models._shared_estimators.build_from_state_dict`
   or the library's own ``from_pretrained(state_dict=...)``; children keep and pickle their own
   weights, and the pickle, attach and device machinery of the mixin stays out of the way.

2. Inherit the mixin as the FIRST base and declare the spec::

       class MyModel(SharedWeightsModelMixin, AbstractTorchModel):
           shared_weights_spec = SharedWeightsSpec(
               library="mylib",
               checkpoint=CheckpointSpec(repo_id="org/MyModel", filename_param="checkpoint", ...),
               ...
           )

   The mixin must precede ``AbstractTorchModel`` so its ``__getstate__``, ``save``,
   ``predict_proba``, ``_set_device``, ``_get_memory_size`` and ``get_info`` run first and its
   class tags win the merge in ``Taggable._get_class_tags``. The spec fields, in the order a
   wrapper author meets them:

   * ``library``: the registry's library name.
   * ``checkpoint``: a :class:`CheckpointSpec` (or one per variant) with the Hub coordinates;
     ``None`` when the coordinates live in the library (its cache dir, a manifest, a variant
     registry), in which case the wrapper overrides ``_resolve_shared_checkpoint``.
   * ``variant``: the estimator the problem type uses. The default maps binary and multiclass to
     ``"classifier"`` and regression to ``"regressor"``; a constant string means one network for
     every task; ``None`` for a problem type (or a checkpoint that resolves to no file) means no
     sharing for that task.
   * ``default_params``: the checkpoint-relevant literals of ``_set_default_params``, overlaid
     on the user hyperparameters before the key is derived, so the warm-up (which sees the raw
     config) and the fit derive the same key. A callable receives the concrete class, for defaults
     that are class attributes.
   * ``dtype``, ``flag_params``, ``key_flags``: what else identifies the built network (a constant
     dtype or a function of the hyperparameters and the device type; hyperparameters the module
     stores; computed flags such as a device-dependent attention kernel).
   * ``disable_when`` and ``unshareable_examples``: configurations that keep the library's own
     per-estimator load (a KV cache, a user checkpoint path, a compile flag). Every example is
     asserted by the convention test to yield no key.
   * ``device_param`` / ``device_from_params``: a hyperparameter that overrides the allocated device
     in the key. ``cache_device`` (state_dict mode) fixes the cached copy to the CPU.
   * ``network_attr``: the dotted path on the library estimator that holds the shared object
     (``"model_"``, ``"model"``, ``"_engine.model"``, ``"models_"`` for a list). ``detach_attr`` is
     what the pickle drops (the same path by default; ``"library"`` when the library's own
     ``__getstate__`` already pickles weightless).
   * ``seam`` and ``post_attach_calls``: how a payload reaches the estimator on reattach.
     ``"attr"`` sets ``network_attr``; ``"load_model"`` drives an estimator derived with
     :func:`tabarena.models._shared_estimators.derive_shared_estimator` (``use_shared_weights``
     then ``_load_model`` then the named zero-argument methods).
   * ``device_attrs``: every device field the library keeps beside the module, as ``(path, kind)``
     with ``kind`` ``"torch"`` or ``"str"``; the mixin rewrites them on every device change and
     load, so a CPU reload of a CUDA-fitted child never keeps a stale ``"cuda"`` field.
   * ``owned_tensor_attrs``: fitted tensors the estimator stores on the fit device (a PCA basis,
     a criterion). They are moved to the CPU in the pickled copy and back to the current device on
     attach; the generic pickle test walks a pickled child and fails on any tensor left off the
     CPU, which is how a forgotten attribute surfaces.
   * ``flag_attrs``, ``estimator_move``, ``freeze_parameters``, ``download_param``,
     ``checkpoint_choices``: see the field docs on :class:`SharedWeightsSpec`.

3. Implement ``_build_shared_weights(key)`` (mandatory in module mode; state_dict mode has a
   default file reader). Read ``key.checkpoint``, build on ``key.device`` with ``key.dtype`` and the
   flags, and return the module or the small container the estimator accepts. Keep the library's
   own construction steps so a shared fit predicts exactly what an unshared fit predicts. This hook
   and ``_resolve_shared_checkpoint`` are the only ones that may import the library; every other
   hook stays library-free (device conversion through ``torch.device``, object construction behind
   a zero-argument accessor in the wrapper's ``_estimators.py`` that the test can monkeypatch).

4. Hand the payload to the library in ``_fit``. After ``device = self._resolve_fit_device(...)``
   (or the wrapper's own rule) call ``key, payload = self._acquire_shared_weights(device=device)``
   and use the seam: a constructor keyword (``Estimator(..., model=payload)``), a derived estimator
   (``est.use_shared_weights(key, payload, type(self)._load_shared_weights)`` before ``fit``;
   the derived class is constructed unconditionally and falls through to the library without a
   key), ``configure_shared_weights(state_dict)`` in state_dict mode, or a constructor replica
   guarded by :func:`tabarena.models._shared_estimators.check_signature_matches` when the library
   has no seam at all. That is two to six lines.

5. Override an optional hook only when the defaults cannot express the library:
   ``_resolve_shared_checkpoint``, ``prefetch_weights``, ``get_device``, ``_network_attached``,
   ``_attach_shared_weights``, ``_detach_for_pickle``, ``_move_owned_network``,
   ``_after_device_change``, ``_shared_modules``, ``_prepare_owned_for_inference``.

6. Declare ``warmup_modules`` and, when useful, ``warmup_dummy_fit_hyperparameters``; point
   ``info.py`` at ``prefetch_weights=<Model>.prefetch_weights``. Then run
   ``python -P -m tabarena.tools.audit_warmup --model <Method>`` and
   ``pytest tests/tabarena/models/test_shared_weights_models.py -k <Model>``.

What the fit does
-----------------

``_acquire_shared_weights`` reads the model's own hyperparameters (never a mutated copy, so the fit
key cannot diverge from the warm-up key), decides whether a Hub download is allowed
(``ag.fetch_pretrained_weights`` and, when ``download_param`` is set, the library's own flag),
derives the key, records whether the entry was present before the fit and its provenance, and
returns the registry object (a hit after the warm-up, one build otherwise). ``_shares_network``
is False when ``ag.save_pretrained_weights`` asks for a self-contained artifact, when the class
setting ``share_weights`` is off (``TabularPredictor.fit(model_class_settings={"<ag_key>":
{"share_weights": False}})``, scoped per registered class through the ``class_settings_cls`` stamp
in ``__init_subclass__``), or when the configuration is unshareable.

Pickling and devices
--------------------

A shared child pickles without its network (``__getstate__`` swaps the estimator for a detached
shallow copy) and with its fitted tensors on the CPU, so AutoGluon's plain-pickle save and load work
on a host without the fit device. ``predict_proba`` and ``prepare_for_inference`` reattach through
``_ensure_network``: the key is re-derived for the current device type (dtype and flags recomputed),
the registry is asked (a hit in the fitting process, one build in a fresh one, download only when
allowed at stage ``"load"``), and the device fields, flag fields and owned tensors follow. A shared
payload is never moved with ``.to()``; ``_set_device`` swaps registry entries per device type. An
estimator that owns its network (sharing off) is moved in place and keeps the CPU round trip on
``save`` that ``AbstractTorchModel`` performs.

Metadata
--------

``get_info()["shared_weights"]`` is the key without host paths (``checkpoint`` reduced to its
basename), ``loaded_by`` (``"warmup"``, ``"fit"`` or ``"load"``), ``present_before_fit`` and
``checkpoint_source`` (repo id, filename, revision), or ``None`` for an unshared fit. The registry's
``report()`` and the warm-up report carry the same provenance.

Fairness
--------

Pretrained weights are part of the environment, like an imported library or a CUDA context: they
are identical for every fit of that checkpoint and a served deployment keeps them resident. The key
is derived from the problem type, the allocated hardware and the configuration only, never from the
task's data, so a primed entry carries no task-specific state into the fit. Anything written during
a fit (fine-tuned parameters, fitted context, KV caches, compiled graphs) is never shared; a
configuration that mutates the network in place is declared in ``disable_when`` and loads its own
network as before. The registry docstring in :mod:`tabarena.models._weights` states the loader
contract (local-first resolution, build on the key's device, random states restored).

This module imports torch and the model libraries inside methods only. Besides the standard library
and ``autogluon.common`` it imports the registry, the Hub resolvers and the estimator helpers
through the one import block below, so it can move into ``autogluon.core`` alongside them.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from autogluon.common.utils.pretrained_weights import (
    PretrainedWeightsUnavailableError,
    fetch_allowed,
    unavailable_message,
)

from tabarena.models import prefetch as _hub
from tabarena.models._shared_estimators import (
    DeviceAttr,
    apply_device_attrs,
    check_payload_device,
    detach_by_path,
    get_by_path,
    hf_checkpoint_source,
    move_tensor_attrs,
    payload_device,
    payload_modules,
    set_by_path,
)
from tabarena.models._weights import (
    LoadedBy,
    SharedWeightsClassSettings,
    WeightsKey,
    contains,
    get_or_load,
    load_state_dict_file,
    loaded_by,
    make_key,
    normalize_device,
    tensor_bytes,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    import torch

logger = logging.getLogger(__name__)

#: Problem types a variant mapping or callable is evaluated for.
PROBLEM_TYPES: tuple[str, ...] = ("binary", "multiclass", "regression")

#: The default ``SharedWeightsSpec.variant``: one classifier network and one regressor network.
DEFAULT_VARIANTS: dict[str, str | None] = {
    "binary": "classifier",
    "multiclass": "classifier",
    "regression": "regressor",
}

#: Placeholder ``ag_key`` values of intermediate wrapper bases that are not registered themselves.
_UNREGISTERED_AG_KEYS: frozenset[str | None] = frozenset({None, "", "NOTSET"})

_SHARED_FIELDS: tuple[str, ...] = ("_shared_key", "_present_before_fit", "_checkpoint_source", "_shared_hps")

_AG_ARG_KEYS: frozenset[str] = frozenset({"ag_args", "ag_args_fit", "ag_args_ensemble"})


def _strip_ag_args(hyperparameters: Mapping[str, Any] | None) -> dict[str, Any]:
    """A copy of ``hyperparameters`` without the ``ag_args*`` keys (the shape a model sees as its own params)."""
    return {k: v for k, v in (hyperparameters or {}).items() if k not in _AG_ARG_KEYS}


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def _looks_fitted(estimator: Any) -> bool:
    """Sklearn's fitted marker: any public attribute ending in an underscore."""
    return any(name.endswith("_") and not name.startswith("__") for name in vars(estimator))


# --- checkpoint coordinates ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """A checkpoint on the local disk and where it came from.

    Args:
        path: Local file (or directory for a snapshot without ``weights_file``) the build reads.
        source: Provenance without host paths (repo id, filename, revision, snapshot commit).
    """

    path: str
    source: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckpointSpec:
    """Static Hub coordinates of one checkpoint, resolved local-first.

    Args:
        repo_id: Hugging Face repository id.
        filename: The file inside the repository (``kind="file"``) when it is fixed.
        filename_param: Hyperparameter naming the file instead of ``filename``. Its value is a
            string, a mapping from variant to file, or a tuple with one file per variant in
            ``variant_order`` (the TabICL ``(classifier, regressor)`` form).
        revision: Pinned commit or branch; ``None`` resolves the cache's default branch pointer.
        kind: ``"file"`` resolves one file through ``resolve_hf_file``; ``"snapshot"`` resolves the
            repository snapshot through ``resolve_hf_snapshot`` and appends ``subfolder`` and
            ``weights_file`` to the snapshot directory.
        subfolder: Folder inside the repository (both kinds).
        required_files: Snapshot paths that must exist for a cached snapshot to count as complete;
            they double as the download patterns.
        weights_file: The file inside the snapshot that becomes the key's checkpoint (state_dict
            mode needs one file).
        user_path_param: Hyperparameter that may carry a user checkpoint path. An existing path is
            shared under that path; a value that is not an existing path disables sharing. Wrappers
            that never share user checkpoints list the parameter in ``disable_when`` instead.
        default_filename: Default for ``filename_param`` when the overlaid hyperparameters do not
            carry it: a string, a per-variant mapping, or a callable of the concrete model class.
        variant_order: Variant names a tuple-valued ``filename_param`` is indexed by.
    """

    repo_id: str
    filename: str | None = None
    filename_param: str | None = None
    revision: str | None = None
    kind: Literal["file", "snapshot"] = "file"
    subfolder: str | None = None
    required_files: tuple[str, ...] = ()
    weights_file: str | None = None
    user_path_param: str | None = None
    default_filename: str | Mapping[str, str] | Callable[[type], Any] | None = None
    variant_order: tuple[str, ...] = ("classifier", "regressor")

    def _pick(self, value: Any, variant: str) -> str | None:
        """One filename for ``variant`` from a string, a per-variant mapping or a per-variant tuple."""
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            if variant not in self.variant_order:
                return None
            index = self.variant_order.index(variant)
            return value[index] if index < len(value) else None
        if hasattr(value, "get"):
            return value.get(variant)
        raise TypeError(f"{self.filename_param!r} must be a string, a mapping or a tuple per variant, got {value!r}")

    def filename_for(self, *, hyperparameters: Mapping[str, Any], variant: str, cls: type | None = None) -> str | None:
        """The repository file a fit of ``variant`` reads under the overlaid ``hyperparameters``; ``None`` if none."""
        if self.filename is not None:
            return self.filename
        value = hyperparameters.get(self.filename_param) if self.filename_param else None
        if value is None and self.default_filename is not None:
            value = self.default_filename(cls) if callable(self.default_filename) else self.default_filename
        return self._pick(value, variant)

    def _user_path(self, hyperparameters: Mapping[str, Any]) -> ResolvedCheckpoint | None | Literal[False]:
        """The user checkpoint when ``user_path_param`` is set: a checkpoint, ``None`` (unshareable) or False (unset)."""
        if self.user_path_param is None or hyperparameters.get(self.user_path_param) is None:
            return False
        candidate = str(hyperparameters[self.user_path_param])
        if not os.path.exists(candidate):
            return None
        if os.path.isdir(candidate) and self.weights_file is not None:
            candidate = os.path.join(candidate, self.weights_file)
            if not os.path.exists(candidate):
                return None
        return ResolvedCheckpoint(path=str(Path(candidate).resolve()), source={"user_path": Path(candidate).name})

    def resolve(
        self,
        *,
        hyperparameters: Mapping[str, Any],
        variant: str,
        allow_download: bool,
        cls: type | None = None,
    ) -> ResolvedCheckpoint | None:
        """Local-first resolution of the checkpoint for ``variant``; ``None`` when the configuration has none.

        Raises:
            tabarena.models.prefetch.WeightsUnavailableError: The file is not cached and
                ``allow_download`` is False.
        """
        user = self._user_path(hyperparameters)
        if user is not False:
            return user
        if self.kind == "file":
            filename = self.filename_for(hyperparameters=hyperparameters, variant=variant, cls=cls)
            if filename is None:
                return None
            path = _hub.resolve_hf_file(
                self.repo_id, filename, revision=self.revision, subfolder=self.subfolder, allow_download=allow_download
            )
            full_name = f"{self.subfolder}/{filename}" if self.subfolder else filename
            return ResolvedCheckpoint(
                path=path, source=hf_checkpoint_source(repo_id=self.repo_id, filename=full_name, revision=self.revision)
            )
        if self.kind == "snapshot":
            patterns = list(self.required_files) or ([f"{self.subfolder}/*"] if self.subfolder else None)
            snapshot = _hub.resolve_hf_snapshot(
                self.repo_id,
                revision=self.revision,
                allow_patterns=patterns,
                required_files=self.required_files or None,
                allow_download=allow_download,
            )
            path = Path(snapshot)
            if self.subfolder:
                path = path / self.subfolder
            if self.weights_file:
                path = path / self.weights_file
            relative = str(path.relative_to(snapshot)) if path != Path(snapshot) else None
            return ResolvedCheckpoint(
                path=str(path),
                source=hf_checkpoint_source(
                    repo_id=self.repo_id, filename=relative, revision=self.revision, path=snapshot
                ),
            )
        raise ValueError(f"unknown checkpoint kind {self.kind!r}")

    def prefetch(self, *, cls: type, variant: str, choices: Mapping[str, tuple[Any, ...]]) -> list[str]:
        """Resolve (downloading when needed) the default checkpoint of ``variant`` and every alternative in ``choices``.

        ``choices`` maps ``filename_param`` to the search space's checkpoint values; other keys are
        ignored. Returns the distinct local paths.
        """
        spec = getattr(cls, "shared_weights_spec", None)
        defaults = dict(spec.resolved_default_params(cls)) if spec is not None else {}
        candidates: list[Mapping[str, Any]] = [defaults]
        if self.filename_param is not None:
            candidates.extend(
                {**defaults, self.filename_param: choice} for choice in choices.get(self.filename_param, ())
            )
        paths: list[str] = []
        for hyperparameters in candidates:
            resolved = self.resolve(hyperparameters=hyperparameters, variant=variant, allow_download=True, cls=cls)
            if resolved is not None and resolved.path not in paths:
                paths.append(resolved.path)
        return paths


# --- the spec ------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SharedWeightsSpec:
    """Everything the mixin needs to know about one library's shared network; see the module docstring.

    Frozen with identity hashing (``eq=False``) so mapping and callable fields are allowed; a spec is
    never used as a dictionary key. Concrete wrappers derive a variant with ``dataclasses.replace``.

    Args:
        library: ``WeightsKey.library``.
        mode: ``"module"`` shares the built network by reference; ``"state_dict"`` caches a CPU
            state dict that every child copies into its own module.
        checkpoint: Static coordinates, one for all variants or one per variant; ``None`` when the
            wrapper overrides ``_resolve_shared_checkpoint``.
        variant: ``WeightsKey.variant`` from the problem type: a constant, a mapping (``None`` for
            a problem type disables sharing there) or a callable of the problem type.
        default_params: Checkpoint-relevant defaults overlaid on the user hyperparameters before the
            key is derived; a callable receives the concrete class.
        dtype: ``WeightsKey.dtype``: a constant or a function of (overlaid hyperparameters, device
            type). A callable returning ``None`` means the network cannot run on that device and
            the key is ``None``.
        flag_params: Hyperparameters copied verbatim into the key's flags because the built module
            stores and reads them.
        key_flags: Extra computed flags from (overlaid hyperparameters, device type).
        disable_when: Per-configuration opt-outs on the overlaid hyperparameters: a parameter name
            (disables when truthy) or a predicate (disables when True).
        unshareable_examples: Hyperparameter dicts that must yield no key; the convention test
            asserts each one. Required when ``disable_when`` contains a predicate.
        device_param: Hyperparameter overriding the allocated device in the key (``"auto"`` and
            ``None`` keep the allocation).
        device_from_params: Replaces the ``device_param`` rule when it is not a plain override; a
            function of (overlaid hyperparameters, allocated device type) returning the key's device
            type or ``None`` to disable sharing.
        cache_device: state_dict mode only: the device the payload is cached on (``"cpu"``).
        state_dict_format: state_dict mode: how the default reader parses the file (``"auto"``
            dispatches on the suffix; a Hub blob path has none, so declare it).
        pin_memory: state_dict mode: pin the host tensors when CUDA is available.
        network_attr: Dotted path on the library estimator holding the shared object (module mode).
        detach_attr: Path detached from the pickled copy; defaults to ``network_attr``. ``"library"``
            means the library's own ``__getstate__`` already drops the network.
        freeze_parameters: Whether the loader calls ``requires_grad_(False)`` on the built modules.
            False where frozen parameters select other CPU kernels and move float32 predictions.
        download_param: Library hyperparameter that must also permit a Hub download on a cache miss.
        seam: How a payload reaches the estimator on attach: ``"attr"`` sets ``network_attr``;
            ``"load_model"`` drives a ``derive_shared_estimator`` class.
        post_attach_calls: Zero-argument estimator methods run after an attach or a device change
            (``seam="load_model"``), guarded by ``hasattr`` and by the estimator being fitted.
        device_attrs: Estimator device fields the mixin rewrites on every device change and load.
        owned_tensor_attrs: Fitted tensors the estimator stores on the fit device; moved to the CPU
            in the pickled copy and back to the current device on attach.
        flag_attrs: Key flags whose value the estimator also stores, written after a device swap.
        estimator_move: The library estimator's own device-move method, used only for an estimator
            that owns its network; a shared payload is never moved.
        checkpoint_choices: Search-space alternatives of ``filename_param`` values that the default
            ``prefetch_weights`` resolves in addition to the defaults. Ignored by the key.
    """

    library: str
    mode: Literal["module", "state_dict"] = "module"
    checkpoint: CheckpointSpec | Mapping[str, CheckpointSpec] | None = None
    variant: str | Mapping[str, str | None] | Callable[[str], str | None] = field(
        default_factory=lambda: dict(DEFAULT_VARIANTS)
    )
    default_params: Mapping[str, Any] | Callable[[type], Mapping[str, Any]] = field(default_factory=dict)
    dtype: str | Callable[[dict, str], str | None] = "float32"
    flag_params: tuple[str, ...] = ()
    key_flags: Callable[[dict, str], Mapping[str, Any]] | None = None
    disable_when: tuple[str | Callable[[dict], bool], ...] = ()
    unshareable_examples: tuple[Mapping[str, Any], ...] = ()
    device_param: str | None = None
    device_from_params: Callable[[dict, str], str | None] | None = None
    cache_device: str | None = None
    state_dict_format: Literal["safetensors", "torch", "auto"] = "auto"
    pin_memory: bool = True
    network_attr: str = "model_"
    detach_attr: str | None = None
    freeze_parameters: bool = True
    download_param: str | None = None
    seam: Literal["attr", "load_model"] = "attr"
    post_attach_calls: tuple[str, ...] = ()
    device_attrs: tuple[DeviceAttr, ...] = ()
    owned_tensor_attrs: tuple[str, ...] = ()
    flag_attrs: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    estimator_move: str | None = None
    checkpoint_choices: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)

    def variant_for(self, problem_type: str) -> str | None:
        """The key's variant for ``problem_type``; ``None`` disables sharing for that task."""
        if isinstance(self.variant, str):
            return self.variant
        if callable(self.variant):
            return self.variant(problem_type)
        return self.variant.get(problem_type)

    def variants(self) -> tuple[str, ...]:
        """Every variant that can carry a checkpoint, for prefetching."""
        if self.checkpoint is not None and not isinstance(self.checkpoint, CheckpointSpec):
            return tuple(self.checkpoint.keys())
        found: list[str] = []
        for problem_type in PROBLEM_TYPES:
            variant = self.variant_for(problem_type)
            if variant is not None and variant not in found:
                found.append(variant)
        return tuple(found)

    def checkpoint_for(self, variant: str) -> CheckpointSpec | None:
        """The static checkpoint coordinates of ``variant``, or ``None``."""
        if self.checkpoint is None or isinstance(self.checkpoint, CheckpointSpec):
            return self.checkpoint
        return self.checkpoint.get(variant)

    def resolved_default_params(self, cls: type) -> Mapping[str, Any]:
        """``default_params`` for the concrete class ``cls``."""
        return self.default_params(cls) if callable(self.default_params) else self.default_params

    def overlay(self, cls: type, hyperparameters: Mapping[str, Any]) -> dict[str, Any]:
        """The defaults overlaid with the user hyperparameters (user values win)."""
        return {**self.resolved_default_params(cls), **hyperparameters}

    def can_share(self, hyperparameters: Mapping[str, Any]) -> bool:
        """Evaluate ``disable_when`` on already-overlaid hyperparameters."""
        for rule in self.disable_when:
            if isinstance(rule, str):
                if hyperparameters.get(rule):
                    return False
            elif rule(dict(hyperparameters)):
                return False
        return True

    def dtype_for(self, hyperparameters: Mapping[str, Any], device_type: str) -> str | None:
        """The key's dtype for a configuration on ``device_type``; ``None`` means it cannot run there."""
        if callable(self.dtype):
            return self.dtype(dict(hyperparameters), device_type)
        return self.dtype

    def flags_for(self, hyperparameters: Mapping[str, Any], device_type: str) -> dict[str, Any]:
        """The raw (unstringified) key flags of a configuration on ``device_type``."""
        flags = {name: hyperparameters[name] for name in self.flag_params if name in hyperparameters}
        if self.key_flags is not None:
            flags.update(self.key_flags(dict(hyperparameters), device_type))
        return flags

    def has_predicates(self) -> bool:
        return any(not isinstance(rule, str) for rule in self.disable_when)


def _validate_spec(spec: SharedWeightsSpec, cls: type) -> None:
    """Raise ``TypeError`` for a spec the mixin cannot run; called from ``__init_subclass__``."""
    where = f"{cls.__name__}.shared_weights_spec"
    if not isinstance(spec, SharedWeightsSpec):
        raise TypeError(f"{where} must be a SharedWeightsSpec, got {type(spec).__name__}")
    if spec.mode not in ("module", "state_dict"):
        raise TypeError(f"{where}: mode must be 'module' or 'state_dict', got {spec.mode!r}")
    if spec.mode == "state_dict" and spec.cache_device != "cpu":
        raise TypeError(f"{where}: state_dict mode caches on the CPU; declare cache_device='cpu'")
    if spec.mode == "module" and not spec.network_attr:
        raise TypeError(f"{where}: module mode needs a non-empty network_attr")
    if spec.seam not in ("attr", "load_model"):
        raise TypeError(f"{where}: seam must be 'attr' or 'load_model', got {spec.seam!r}")
    if spec.has_predicates() and not spec.unshareable_examples:
        raise TypeError(f"{where}: disable_when has a predicate; declare at least one unshareable_examples entry")
    for path, kind in spec.device_attrs:
        if kind not in ("torch", "str") or not path:
            raise TypeError(f"{where}: device_attrs entry {(path, kind)!r} must be ('<path>', 'torch'|'str')")
    if spec.checkpoint is not None and not isinstance(spec.checkpoint, CheckpointSpec):
        if not all(isinstance(value, CheckpointSpec) for value in spec.checkpoint.values()):
            raise TypeError(f"{where}: checkpoint must be a CheckpointSpec or a mapping of variant to CheckpointSpec")
    if spec.state_dict_format not in ("auto", "safetensors", "torch"):
        raise TypeError(f"{where}: state_dict_format must be 'auto', 'safetensors' or 'torch'")


# --- the mixin -----------------------------------------------------------------------------------


@dataclass
class _KeyMemo:
    """What a key was derived from, so a loader can re-resolve a moved checkpoint and report provenance."""

    source: dict[str, Any]
    problem_type: str
    variant: str
    hyperparameters: dict[str, Any]


class SharedWeightsModelMixin:
    """Shares a wrapper's pretrained network through the process-wide registry; see the module docstring.

    Inherit it as the first base and declare ``shared_weights_spec``. ``shared_weights_spec = None``
    (the bare mixin, or a base that inherits it without sharing) makes every classmethod a
    passthrough, so the mixin can sit in an MRO without effect.
    """

    shared_weights_spec: ClassVar[SharedWeightsSpec | None] = None

    #: Parallel Ray fold workers cannot share process memory, so every mixin user fits folds one after
    #: another; ``AbstractModel._get_default_ag_args_ensemble`` merges this over the MRO and a wrapper's
    #: own extra (for example ``{"refit_folds": True}``) still wins.
    _default_ag_args_ensemble_extra: ClassVar[dict] = {"fold_fitting_strategy": "sequential_local"}

    _shared_weights_memo: ClassVar[dict[WeightsKey, _KeyMemo]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        spec = getattr(cls, "shared_weights_spec", None)
        if spec is None:
            return
        _validate_spec(spec, cls)
        cls._shared_weights_memo = {}
        registered = cls.__dict__.get("ag_key") not in _UNREGISTERED_AG_KEYS
        if not registered:
            return
        if "class_settings_cls" not in cls.__dict__:
            cls.class_settings_cls = SharedWeightsClassSettings
        if spec.mode == "module":
            own = getattr(cls._build_shared_weights, "__func__", None)
            base = SharedWeightsModelMixin.__dict__["_build_shared_weights"].__func__
            if own is base:
                raise TypeError(f"{cls.__name__} shares in module mode and must implement _build_shared_weights(key)")

    # --- class level ------------------------------------------------------------------------------

    @classmethod
    def _class_tags(cls) -> dict:
        spec = getattr(cls, "shared_weights_spec", None)
        if spec is None or spec.mode != "module":
            return {}
        # The pickled child is weightless with its fitted tensors on the CPU, so there is nothing to
        # move before a save; a CPU round trip would also move a shared module for every holder.
        return {"can_set_device": True, "set_device_on_save_to": None, "set_device_on_load": True}

    @classmethod
    def _can_share_params(cls, hyperparameters: Mapping[str, Any]) -> bool:
        """Evaluate ``spec.disable_when`` on already-overlaid hyperparameters."""
        spec = getattr(cls, "shared_weights_spec", None)
        return spec is not None and spec.can_share(hyperparameters)

    @classmethod
    def _shared_device_type(
        cls, spec: SharedWeightsSpec, hyperparameters: Mapping[str, Any], device: str
    ) -> str | None:
        """The key's device type from the allocated ``device`` and the configuration; ``None`` disables sharing."""
        if spec.mode == "state_dict":
            return spec.cache_device
        allocated = normalize_device(device)
        if spec.device_from_params is not None:
            requested = spec.device_from_params(dict(hyperparameters), allocated)
            return None if requested is None else normalize_device(requested)
        if spec.device_param is not None:
            requested = hyperparameters.get(spec.device_param)
            if requested is None or requested == "auto":
                return allocated
            if isinstance(requested, (list, tuple)):
                return None
            return normalize_device(requested)
        return allocated

    @classmethod
    def _key_components(
        cls, spec: SharedWeightsSpec, *, problem_type: str, hyperparameters: Mapping[str, Any], device: str
    ) -> tuple[str, str, str] | None:
        """``(variant, device type, dtype)`` of a configuration, or ``None`` when any rule disables sharing."""
        supported = getattr(cls, "_supported_problem_types", None)
        if supported is not None and problem_type not in supported:
            return None
        if not spec.can_share(hyperparameters):
            return None
        variant = spec.variant_for(problem_type)
        device_type = None if variant is None else cls._shared_device_type(spec, hyperparameters, device)
        dtype = None if device_type is None else spec.dtype_for(hyperparameters, device_type)
        if variant is None or device_type is None or dtype is None:
            return None
        return variant, device_type, dtype

    @classmethod
    def shared_weights_key(
        cls,
        *,
        problem_type: str,
        hyperparameters: Mapping[str, Any],
        device: str,
        allow_download: bool = True,
    ) -> WeightsKey | None:
        """The registry key a fit of this configuration uses; ``None`` when it cannot share.

        The single entry point for the warm-up (layer 5 of ``warmup_model_cls``) and for
        ``_acquire_shared_weights``. ``ag_args*`` keys are stripped, ``spec.default_params`` is
        overlaid, then ``disable_when``, the variant, the device rule, the dtype and the checkpoint
        resolution decide; any of them can yield ``None``. A problem type the class does not support
        yields ``None`` too. The resolved checkpoint's provenance is memoized for
        :meth:`shared_weights_source`.

        Raises:
            tabarena.models.prefetch.WeightsUnavailableError: The checkpoint is not cached and
                ``allow_download`` is False.
        """
        spec = getattr(cls, "shared_weights_spec", None)
        if spec is None:
            return None
        hps = spec.overlay(cls, _strip_ag_args(hyperparameters))
        components = cls._key_components(spec, problem_type=problem_type, hyperparameters=hps, device=device)
        if components is None:
            return None
        variant, device_type, dtype = components
        resolved = cls._resolve_shared_checkpoint(
            problem_type=problem_type, variant=variant, hyperparameters=hps, allow_download=allow_download
        )
        if resolved is None:
            return None
        key = make_key(
            spec.library, resolved.path, variant, device_type, dtype=dtype, **spec.flags_for(hps, device_type)
        )
        cls._shared_weights_memo[key] = _KeyMemo(
            source=dict(resolved.source), problem_type=problem_type, variant=variant, hyperparameters=dict(hps)
        )
        return key

    @classmethod
    def shared_weights_source(cls, key: WeightsKey) -> dict[str, Any]:
        """The checkpoint provenance memoized by the last :meth:`shared_weights_key` derivation of ``key``; ``{}`` if unknown."""
        memo = cls._memo_for(key)
        return {} if memo is None else dict(memo.source)

    @classmethod
    def _memo_for(cls, key: WeightsKey) -> _KeyMemo | None:
        """The memo of ``key`` or of a key differing only in device, dtype or flags."""
        memo = cls._shared_weights_memo.get(key)
        if memo is not None:
            return memo
        for other, candidate in cls._shared_weights_memo.items():
            if other.checkpoint == key.checkpoint and other.variant == key.variant and other.library == key.library:
                return candidate
        return None

    @classmethod
    def _resolve_shared_checkpoint(
        cls,
        *,
        problem_type: str,
        variant: str,
        hyperparameters: Mapping[str, Any],
        allow_download: bool,
        stage: str = "fit",
    ) -> ResolvedCheckpoint | None:
        """The local checkpoint for ``variant`` under the overlaid ``hyperparameters``; ``None`` when there is none.

        The default resolves ``spec.checkpoint`` for the variant. Override where the coordinates
        come from the library (its cache directory, a manifest, a variant registry); this hook and
        ``_build_shared_weights`` are the only ones that may import the library.
        """
        spec = cls.shared_weights_spec
        checkpoint = spec.checkpoint_for(variant)
        if checkpoint is None:
            raise TypeError(
                f"{cls.__name__} declares no CheckpointSpec for variant {variant!r}; implement _resolve_shared_checkpoint"
            )
        return checkpoint.resolve(
            hyperparameters=hyperparameters, variant=variant, allow_download=allow_download, cls=cls
        )

    @classmethod
    def _load_shared_weights(cls, key: WeightsKey, *, allow_download: bool | None = None) -> Any:
        """Registry loader: build the payload for ``key`` (the warm-up primes with ``partial(cls._load_shared_weights, key)``).

        ``allow_download=None`` applies the fetch policy's fit-stage default. A key whose checkpoint
        path no longer exists is re-resolved from the derivation memo (download only when allowed)
        and built from the re-resolved file; the registry entry keeps the key it was asked for. The
        payload's modules are put in eval mode and, with ``spec.freeze_parameters``, frozen.
        """
        spec = getattr(cls, "shared_weights_spec", None)
        if spec is None:
            raise TypeError(f"{cls.__name__} declares no shared_weights_spec and cannot load shared weights")
        if allow_download is None:
            allow_download = fetch_allowed(True, stage="fit")
        build_key = key
        if not os.path.exists(key.checkpoint):
            memo = cls._memo_for(key)
            resolved = None
            if memo is not None:
                try:
                    resolved = cls._resolve_shared_checkpoint(
                        problem_type=memo.problem_type,
                        variant=memo.variant,
                        hyperparameters=memo.hyperparameters,
                        allow_download=allow_download,
                        stage="load",
                    )
                except _hub.WeightsUnavailableError as exc:
                    raise PretrainedWeightsUnavailableError(
                        unavailable_message(model_name=cls.__name__, stage="load", location=str(exc))
                    ) from exc
            if resolved is None or not os.path.exists(resolved.path):
                raise PretrainedWeightsUnavailableError(
                    unavailable_message(model_name=cls.__name__, stage="load", location=key.checkpoint)
                )
            build_key = key.replace(checkpoint=resolved.path)
        payload = cls._build_shared_weights(build_key)
        return _finalize_payload(payload, freeze=spec.freeze_parameters)

    @classmethod
    def _build_shared_weights(cls, key: WeightsKey) -> Any:
        """Build the payload for one key: read ``key.checkpoint`` on ``key.device`` with ``key.dtype`` and the flags.

        Mandatory in module mode (return the module or the small container the estimator accepts,
        built with the library's own steps; never read task data). state_dict mode reads the file
        into a CPU state dict by default.
        """
        spec = cls.shared_weights_spec
        if spec.mode == "state_dict":
            return load_state_dict_file(
                key.checkpoint, spec.cache_device or "cpu", pin_memory=spec.pin_memory, format=spec.state_dict_format
            )
        raise NotImplementedError(f"{cls.__name__} must implement _build_shared_weights(key) for module mode")

    @classmethod
    def prefetch_weights(cls) -> list[str]:
        """Make every declared checkpoint present locally; returns the resolved local paths.

        Resolves, for every variant of ``spec.checkpoint``, the default checkpoint and every
        ``spec.checkpoint_choices`` alternative with downloads allowed. ``[]`` without a spec or
        static coordinates; wrappers whose coordinates live in the library override this.
        """
        spec = getattr(cls, "shared_weights_spec", None)
        if spec is None or spec.checkpoint is None:
            return []
        paths: list[str] = []
        for variant in spec.variants():
            checkpoint = spec.checkpoint_for(variant)
            if checkpoint is None:
                continue
            for path in checkpoint.prefetch(cls=cls, variant=variant, choices=spec.checkpoint_choices):
                if path not in paths:
                    paths.append(path)
        return paths

    # --- instance level ---------------------------------------------------------------------------

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._shared_key: WeightsKey | None = None
        self._present_before_fit: bool | None = None
        self._checkpoint_source: dict[str, Any] | None = None
        self._shared_hps: dict[str, Any] | None = None

    def __setstate__(self, state: dict) -> None:
        for name in _SHARED_FIELDS:
            state.setdefault(name, None)
        self.__dict__.update(state)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        spec = self.shared_weights_spec
        if spec is None or spec.mode != "module" or self._shared_key is None or not self._network_attached():
            return state
        estimator = self._detach_for_pickle(self.model)
        estimator = move_tensor_attrs(estimator, spec.owned_tensor_attrs, "cpu", copy_holder=True)
        state["model"] = estimator
        return state

    def _shares_network(self) -> bool:
        """Whether this model takes its network from the registry instead of the library loader."""
        spec = self.shared_weights_spec
        if spec is None or self.aux_params.save_pretrained_weights:
            return False
        settings = type(self).get_class_settings()
        if settings is not None and not getattr(settings, "share_weights", True):
            return False
        return spec.can_share(spec.overlay(type(self), _strip_ag_args(self._get_model_params())))

    def _acquire_shared_weights(self, *, device: str, stage: LoadedBy = "fit") -> tuple[WeightsKey | None, Any | None]:
        """The one call a wrapper's ``_fit`` adds: ``(key, payload)`` from the registry, or ``(None, None)``.

        Reads the model's own hyperparameters, applies the fetch policy (and ``spec.download_param``),
        derives the key, records ``present_before_fit`` and the provenance and takes the registry
        object (a hit after the warm-up, one build otherwise).

        Raises:
            PretrainedWeightsUnavailableError: The checkpoint is not cached and fetching is forbidden.
        """
        spec = self.shared_weights_spec
        if spec is None or not self._shares_network():
            return None, None
        hps = _strip_ag_args(self._get_model_params())
        policy_stage = "fit" if stage in ("fit", "warmup") else "load"
        allow = fetch_allowed(self.aux_params.fetch_pretrained_weights, stage=policy_stage)
        if spec.download_param is not None:
            allow = allow and bool(hps.get(spec.download_param, True))
        try:
            key = type(self).shared_weights_key(
                problem_type=self.problem_type, hyperparameters=hps, device=device, allow_download=allow
            )
        except _hub.WeightsUnavailableError as exc:
            raise PretrainedWeightsUnavailableError(
                unavailable_message(model_name=self.name, stage=policy_stage, location=str(exc))
            ) from exc
        if key is None:
            return None, None
        self._present_before_fit = contains(key)
        self._checkpoint_source = type(self).shared_weights_source(key)
        self._shared_hps = spec.overlay(type(self), hps)
        payload = get_or_load(
            key,
            partial(type(self)._load_shared_weights, key, allow_download=allow),
            stage=stage,
            source=self._checkpoint_source,
        )
        self._shared_key = key
        return key, payload

    def _network_attached(self) -> bool:
        """Whether the estimator currently holds a network at ``spec.network_attr`` (an empty list counts as detached)."""
        if self.model is None:
            return False
        value = get_by_path(self.model, self.shared_weights_spec.network_attr)
        if isinstance(value, (list, tuple)):
            return bool(value)
        return value is not None

    def _attach_shared_weights(self, payload: Any, device: str) -> None:
        """Point the fitted estimator at the registry object ``payload`` living on device type ``device``.

        Never moves the payload and never imports the library. ``seam="attr"`` sets
        ``spec.network_attr``; ``seam="load_model"`` applies the device fields, hands the payload to
        the derived estimator, runs its ``_load_model`` and then ``spec.post_attach_calls``. Both
        then apply the device bookkeeping (device fields, flag fields, owned tensors).
        """
        spec = self.shared_weights_spec
        device_type = normalize_device(device)
        check_payload_device(payload, device_type)
        if spec.seam == "attr":
            if not set_by_path(self.model, spec.network_attr, payload):
                raise RuntimeError(f"{self.name}: cannot set {spec.network_attr!r} on {type(self.model).__name__}")
        else:
            if self._shared_key is None:
                raise RuntimeError(f"{self.name}: attaching through the load_model seam needs a shared key")
            apply_device_attrs(self.model, spec.device_attrs, device_type)
            self.model.use_shared_weights(self._key_for_device(device_type), payload, type(self)._load_shared_weights)
            self.model._load_model()
            if _looks_fitted(self.model):
                for name in spec.post_attach_calls:
                    method = getattr(self.model, name, None)
                    if method is not None:
                        method()
        self._apply_device_bookkeeping(device_type)

    def _detach_for_pickle(self, estimator: Any) -> Any:
        """A shallow copy of ``estimator`` without the shared object; the live estimator keeps it.

        The default detaches ``spec.detach_attr`` (``spec.network_attr`` when unset) and passes the
        estimator through for ``detach_attr="library"``. ``__getstate__`` moves
        ``spec.owned_tensor_attrs`` to the CPU on the result afterwards, so an override need not.
        """
        spec = self.shared_weights_spec
        if spec.detach_attr == "library":
            return estimator
        return detach_by_path(estimator, spec.detach_attr or spec.network_attr)

    def _key_for_device(self, device_type: str) -> WeightsKey:
        """The fit key re-derived for ``device_type``: device replaced, dtype and flags recomputed from the fit hyperparameters."""
        if self._shared_key is None:
            raise RuntimeError(f"{self.name} has no shared key")
        spec = self.shared_weights_spec
        device_type = normalize_device(device_type)
        if self._shared_hps is None:
            return self._shared_key.replace(device=device_type)
        dtype = spec.dtype_for(self._shared_hps, device_type)
        if dtype is None:
            raise RuntimeError(f"{self.name} cannot run on {device_type!r}: no shareable weights dtype for that device")
        return self._shared_key.replace(
            device=device_type, dtype=dtype, flags=spec.flags_for(self._shared_hps, device_type)
        )

    def _load_policy_allows(self) -> bool:
        return fetch_allowed(self.aux_params.fetch_pretrained_weights, stage="load")

    def _ensure_network(self, device: str | None = None) -> None:
        """Reattach the registry object to a weightless estimator (module mode; no-op when attached or in state_dict mode).

        The key is re-derived for ``device`` (the recorded device by default; the CPU when CUDA is
        recorded but absent), a moved checkpoint is re-resolved honouring the load-stage fetch
        policy, and the registry answers with the fitting process's object or one build.

        Raises:
            RuntimeError: The estimator is detached and no shared key is recorded.
        """
        spec = self.shared_weights_spec
        if spec is None or spec.mode != "module" or self.model is None or self._network_attached():
            return
        if self._shared_key is None:
            raise RuntimeError(f"{self.name}: the estimator has no network and no shared key to take one from")
        device_type = normalize_device(device or getattr(self, "device", None) or self.get_device())
        if device_type == "cuda" and not _cuda_available():
            device_type = "cpu"
        key = self._key_for_device(device_type)
        allow = self._load_policy_allows()
        if not os.path.exists(key.checkpoint):
            try:
                resolved = type(self)._resolve_shared_checkpoint(
                    problem_type=self.problem_type,
                    variant=key.variant,
                    hyperparameters=self._shared_hps or {},
                    allow_download=allow,
                    stage="load",
                )
            except _hub.WeightsUnavailableError as exc:
                raise PretrainedWeightsUnavailableError(
                    unavailable_message(model_name=self.name, stage="load", location=str(exc))
                ) from exc
            if resolved is None:
                raise PretrainedWeightsUnavailableError(
                    unavailable_message(model_name=self.name, stage="load", location=key.checkpoint)
                )
            key = key.replace(checkpoint=resolved.path)
            self._checkpoint_source = dict(resolved.source)
        payload = get_or_load(
            key,
            partial(type(self)._load_shared_weights, key, allow_download=allow),
            stage="load",
            source=self._checkpoint_source,
        )
        self._shared_key = key
        self._attach_shared_weights(payload, device_type)

    def save(self, path: str | None = None, verbose: bool = True) -> str:
        """Save through the parent; an estimator that owns its network takes the CPU round trip AutoGluon's tag used to give it."""
        spec = self.shared_weights_spec
        device = getattr(self, "device", None)
        round_trip = (
            spec is not None
            and spec.mode == "module"
            and self._shared_key is None
            and self.model is not None
            and device is not None
            and normalize_device(device) != "cpu"
            and hasattr(self, "set_device")
            and self.is_fit()
        )
        if not round_trip:
            return super().save(path=path, verbose=verbose)
        self.set_device("cpu")
        try:
            return super().save(path=path, verbose=verbose)
        finally:
            self.set_device(device)

    def predict_proba(self, X, **kwargs):
        self._ensure_network()
        return super().predict_proba(X, **kwargs)

    def _set_device(self, device: str) -> None:
        """Module mode: swap registry entries per device type, never ``.to()`` a shared payload; otherwise defer to the parent."""
        spec = self.shared_weights_spec
        if spec is None or spec.mode != "module":
            parent = super()
            if hasattr(parent, "_set_device"):
                parent._set_device(device)
            return
        if self.model is None:
            return
        device_type = normalize_device(device)
        if not self._network_attached():
            if self._shared_key is None:
                return
            self._ensure_network(device_type)
        elif self._shared_key is None:
            self._move_owned_network(device)
        else:
            # Read through ``_shared_modules`` so a wrapper whose network sits behind a nested estimator
            # (the TabPFN many-class wrapper) reports the attached device, not ``network_attr``'s absence.
            current = list(self._shared_modules())
            if payload_device(current) != device_type:
                key = self._key_for_device(device_type)
                payload = get_or_load(
                    key,
                    partial(type(self)._load_shared_weights, key, allow_download=self._load_policy_allows()),
                    stage="load",
                    source=self._checkpoint_source,
                )
                self._shared_key = key
                self._attach_shared_weights(payload, device_type)
            else:
                self._shared_key = self._key_for_device(device_type)
        self._apply_device_bookkeeping(device_type)
        self._after_device_change(device_type)

    def _apply_device_bookkeeping(self, device_type: str) -> None:
        """Write ``device_attrs``, ``flag_attrs`` and move ``owned_tensor_attrs`` for ``device_type``."""
        spec = self.shared_weights_spec
        if self.model is None:
            return
        apply_device_attrs(self.model, spec.device_attrs, device_type)
        if spec.flag_attrs and self._shared_hps is not None:
            flags = spec.flags_for(self._shared_hps, device_type)
            for name, paths in spec.flag_attrs.items():
                if name in flags:
                    for path in paths:
                        set_by_path(self.model, path, flags[name])
        move_tensor_attrs(self.model, spec.owned_tensor_attrs, device_type, copy_holder=False)

    def _move_owned_network(self, device: str) -> None:
        """Move an estimator that owns its network in place: ``spec.estimator_move`` or ``.to(device)`` on the module."""
        spec = self.shared_weights_spec
        if spec.estimator_move:
            getattr(self.model, spec.estimator_move)(device)
            return
        for module in payload_modules(get_by_path(self.model, spec.network_attr)):
            module.to(device)
        move_tensor_attrs(self.model, spec.owned_tensor_attrs, device, copy_holder=False)

    def _after_device_change(self, device_type: str) -> None:
        """Bookkeeping ``spec.device_attrs`` cannot express (engine caches, a rebuilt predictor); no-op by default."""

    def get_device(self) -> str:
        """Device type of the attached network (through ``_shared_modules``), else of the first ``device_attrs`` field, else the recorded device."""
        spec = self.shared_weights_spec
        if spec is None or spec.mode != "module":
            parent = super()
            if hasattr(parent, "get_device"):
                try:
                    return parent.get_device()
                except NotImplementedError:
                    pass
            return normalize_device(getattr(self, "device", None))
        if self.model is not None:
            if self._network_attached():
                return payload_device(list(self._shared_modules()))
            for path, _kind in spec.device_attrs:
                value = get_by_path(self.model, path)
                if value is not None and not isinstance(value, (list, tuple)) and value != "auto":
                    return normalize_device(value)
        return normalize_device(getattr(self, "device", None))

    def _shared_modules(self) -> Iterable[torch.nn.Module]:
        """The modules ``prepare_for_inference`` puts in eval mode and ``_get_memory_size`` counts."""
        if self.model is None:
            return []
        return payload_modules(get_by_path(self.model, self.shared_weights_spec.network_attr))

    def _prepare_owned_for_inference(self) -> None:
        """Untimed preparation of an estimator that owns its network (built lazily by some libraries); no-op by default."""

    def prepare_for_inference(self) -> None:
        """Untimed and idempotent: reattach the network, put it in eval mode, synchronize a CUDA device.

        Model-only and data-free; the data-dependent first-call work stays in the timed predict.
        """
        spec = self.shared_weights_spec
        if spec is None or self.model is None:
            return
        if self._shared_key is None:
            self._prepare_owned_for_inference()
        if spec.mode == "module":
            self._ensure_network()
        elif hasattr(self, "suggest_device_infer") and hasattr(self, "set_device"):
            target = normalize_device(self.suggest_device_infer())
            if normalize_device(self.get_device()) != target:
                self.set_device(target)
        modules = list(self._shared_modules())
        for module in modules:
            module.eval()
        if modules and payload_device(modules) == "cuda":
            import torch

            torch.cuda.synchronize()

    def _get_memory_size(self) -> int:
        spec = self.shared_weights_spec
        if spec is not None and spec.mode == "module" and self._shared_key is not None and self._network_attached():
            return self._get_pickled_size() + tensor_bytes(list(self._shared_modules()))
        return super()._get_memory_size()

    def get_info(self, include_feature_metadata: bool = True) -> dict:
        info = super().get_info(include_feature_metadata=include_feature_metadata)
        info["shared_weights"] = self._shared_weights_info()
        return info

    def _shared_weights_info(self) -> dict[str, Any] | None:
        """``info["shared_weights"]``: the key without host paths plus load provenance; ``None`` for an unshared fit."""
        key = self._shared_key
        if key is None:
            return None
        return {
            **key.to_dict(),
            "checkpoint": Path(key.checkpoint).name,
            "loaded_by": loaded_by(key),
            "present_before_fit": self._present_before_fit,
            "checkpoint_source": dict(self._checkpoint_source or {}),
        }


def _finalize_payload(payload: Any, *, freeze: bool) -> Any:
    """Eval mode (and, with ``freeze``, no gradients) on every module of a freshly built payload."""
    for module in payload_modules(payload):
        module.eval()
        if freeze:
            module.requires_grad_(False)
    return payload


__all__ = [
    "DEFAULT_VARIANTS",
    "CheckpointSpec",
    "ResolvedCheckpoint",
    "SharedWeightsModelMixin",
    "SharedWeightsSpec",
]
