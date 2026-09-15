"""Library-side helpers for wrappers that share pretrained weights through the registry.

:mod:`tabarena.models._shared_weights_model` owns the wrapper side (key derivation, the weightless
pickle, device swaps). This module holds the pieces that touch the library estimator or its payload:

* :func:`derive_shared_estimator` for libraries whose estimator builds its network inside one
  method (``_load_model`` in tabicl, tabldm, the vendored TabSwift, OrionMSP): the derived class
  attaches the registry payload there and falls through to the library without a key.
* :class:`SharedStateDictEstimatorMixin` and :func:`build_from_state_dict` for fine-tuning
  estimators that copy a cached CPU state dict into their own module.
* Payload walkers (:func:`payload_modules`, :func:`payload_device`, :func:`check_payload_device`)
  that accept a module, a state dict or a small dataclass container holding modules.
* Dotted attribute-path helpers (:func:`get_by_path`, :func:`set_by_path`, :func:`detach_by_path`,
  :func:`apply_device_attrs`, :func:`move_tensor_attrs`) that the mixin's default hooks run over
  ``SharedWeightsSpec.network_attr``, ``device_attrs`` and ``owned_tensor_attrs``.
* :func:`hf_checkpoint_source` for provenance without host paths and
  :func:`check_signature_matches`, the drift guard for constructor replicas.

Torch and every model library are imported inside the functions, so importing this module from a
wrapper's ``model.py`` stays cheap and works without the optional extras.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import sys
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from tabarena.models._weights import get_or_load, normalize_device, shallow_copy
from tabarena.models.prefetch import commit_from_snapshot_path

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    import torch

    from tabarena.models._weights import WeightsKey

#: A dotted path on the library estimator and how the device is spelled there.
DeviceAttr = tuple[str, Literal["torch", "str"]]

#: Attribute set on every class :func:`derive_shared_estimator` creates; names the library base.
DERIVED_MARKER = "_tabarena_derived_from"


# --- dotted attribute paths -----------------------------------------------------------------------


def _split(path: str) -> list[str]:
    parts = [part for part in path.split(".") if part]
    if not parts:
        raise ValueError("an attribute path must not be empty")
    return parts


def get_by_path(obj: Any, path: str, default: Any = None) -> Any:
    """``obj.a.b.c`` for ``path="a.b.c"``; ``default`` when any step is missing or ``None``."""
    current = obj
    for part in _split(path):
        if current is None:
            return default
        current = getattr(current, part, None)
    return default if current is None else current


def set_by_path(obj: Any, path: str, value: Any) -> bool:
    """Set ``obj.a.b.c = value``; False (nothing set) when an intermediate object is missing."""
    parts = _split(path)
    parent = obj if len(parts) == 1 else get_by_path(obj, ".".join(parts[:-1]))
    if parent is None:
        return False
    setattr(parent, parts[-1], value)
    return True


def _copy_along_path(root: Any, parts: list[str]) -> tuple[Any, Any] | None:
    """Shallow-copy every object from ``root`` down to the parent of the leaf named by ``parts``.

    Returns ``(new_root, new_parent)`` or ``None`` when an intermediate object is missing. The
    copies share every attribute with the originals, so setting the leaf on ``new_parent`` leaves
    the live objects untouched.
    """
    new_root = shallow_copy(root)
    parent = new_root
    for part in parts[:-1]:
        child = getattr(parent, part, None)
        if child is None:
            return None
        child_copy = shallow_copy(child)
        setattr(parent, part, child_copy)
        parent = child_copy
    return new_root, parent


def detach_by_path(obj: Any, path: str) -> Any:
    """A shallow copy of ``obj`` (and of every object along ``path``) with the leaf set to ``None``.

    ``obj`` itself keeps the leaf. When an intermediate object is missing the copy is returned
    unchanged, so a path that exists only after fit is safe before it.
    """
    parts = _split(path)
    copied = _copy_along_path(obj, parts)
    if copied is None:
        return shallow_copy(obj)
    new_root, parent = copied
    setattr(parent, parts[-1], None)
    return new_root


def apply_device_attrs(obj: Any, attrs: tuple[DeviceAttr, ...], device_type: str) -> None:
    """Write ``device_type`` to every ``(path, kind)`` of ``attrs`` that has an existing parent.

    ``kind="torch"`` writes ``torch.device(device_type)``, ``kind="str"`` the type string. A path
    whose intermediate object is missing (an ``inference_config_`` that exists only after fit) is
    skipped.
    """
    if not attrs:
        return
    torch_device = None
    for path, kind in attrs:
        if kind == "torch":
            if torch_device is None:
                import torch

                torch_device = torch.device(device_type)
            set_by_path(obj, path, torch_device)
        elif kind == "str":
            set_by_path(obj, path, device_type)
        else:
            raise ValueError(f"unknown device attribute kind {kind!r} for {path!r}; expected 'torch' or 'str'")


def move_tensor_attrs(obj: Any, paths: tuple[str, ...], device: str, *, copy_holder: bool) -> Any:
    """Move the tensor at every existing path of ``paths`` to ``device`` with ``detach().to(device)``.

    With ``copy_holder`` the tensors are replaced on shallow copies along each path and the new root
    is returned (the live objects keep their tensors; this is the pickle path). Without it they are
    replaced in place and ``obj`` is returned. Paths holding no tensor are skipped.
    """
    if not paths:
        return obj
    import torch

    root = obj
    for path in paths:
        tensor = get_by_path(root, path)
        if not isinstance(tensor, torch.Tensor):
            continue
        moved = tensor.detach().to(device)
        if copy_holder:
            parts = _split(path)
            copied = _copy_along_path(root, parts)
            if copied is None:
                continue
            root, parent = copied
            setattr(parent, parts[-1], moved)
        else:
            set_by_path(root, path, moved)
    return root


# --- payload walkers -----------------------------------------------------------------------------


def _walk_payload(payload: Any, depth: int) -> Iterator[Any]:
    """Yield the modules and tensors reachable from ``payload`` through containers."""
    import torch

    if payload is None or depth < 0:
        return
    if isinstance(payload, (torch.nn.Module, torch.Tensor)):
        yield payload
    elif isinstance(payload, dict):
        for value in payload.values():
            yield from _walk_payload(value, depth - 1)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            yield from _walk_payload(value, depth - 1)
    elif dataclasses.is_dataclass(payload) and not isinstance(payload, type):
        for field in dataclasses.fields(payload):
            yield from _walk_payload(getattr(payload, field.name, None), depth - 1)


def payload_modules(payload: Any) -> list[torch.nn.Module]:
    """The ``torch.nn.Module`` objects of a payload: a module, a dataclass or dict holding modules."""
    import torch

    return [item for item in _walk_payload(payload, depth=4) if isinstance(item, torch.nn.Module)]


def payload_device(payload: Any) -> str:
    """Device type of the first parameter, buffer or tensor found in ``payload``; ``"cpu"`` when none."""
    import torch

    for item in _walk_payload(payload, depth=4):
        if isinstance(item, torch.Tensor):
            return normalize_device(item.device)
        for tensor in item.parameters():
            return normalize_device(tensor.device)
        for tensor in item.buffers():
            return normalize_device(tensor.device)
    return "cpu"


def check_payload_device(payload: Any, device: str) -> None:
    """Raise ``RuntimeError`` when ``payload`` does not live on device type ``device``."""
    expected = normalize_device(device)
    actual = payload_device(payload)
    if actual != expected:
        raise RuntimeError(
            f"the shared weights live on {actual!r} but the estimator asked for {expected!r}; a shared payload is "
            "never moved with .to(), take the registry entry for the requested device instead"
        )


# --- the _load_model seam ------------------------------------------------------------------------


def estimator_device_type(estimator: Any) -> str | None:
    """Device type an sklearn-style estimator records (``device_`` first, then ``device``); ``None`` if neither."""
    for name in ("device_", "device"):
        value = getattr(estimator, name, None)
        if value is None or value == "auto":
            continue
        if isinstance(value, (list, tuple)):
            return None
        return normalize_device(value)
    return None


def _default_apply(estimator: Any, payload: Any) -> Any:
    """Attach a bare module payload the way most ``_load_model`` hooks end: ``model_`` and ``model_path_``."""
    estimator.model_ = payload
    key = getattr(estimator, "_tabarena_key", None)
    if key is not None:
        estimator.model_path_ = key.checkpoint
    return payload


def derive_shared_estimator(
    library_cls: type,
    *,
    module: str,
    name: str,
    seam: Literal["load_model"] = "load_model",
    load_method: str = "_load_model",
    apply: Callable[[Any, Any], Any] | None = None,
    returns: bool = False,
    cpu_fallback_attr: str | None = "device",
    skip_when: tuple[str, ...] = ("kv_cache",),
) -> type:
    """A subclass of ``library_cls`` whose ``load_method`` attaches the registry payload.

    The wrapper hands the estimator its key, the payload it took from the registry and the loader
    (the wrapper's ``_load_shared_weights`` classmethod) through ``use_shared_weights`` before the
    library's ``fit`` runs ``load_method``. Without a key, or when any attribute in ``skip_when`` is
    truthy, the library's own method runs and the estimator owns its network. After unpickling only
    the key and the loader survive (``__getstate__`` drops the payload): ``load_method`` asks the
    registry for the key on the estimator's current device type, binding that device-replaced key
    into the loader partial, a hit in the benchmark process and one load in a fresh one. A key
    whose checkpoint path no longer exists (another host, a moved cache) falls back to the library's
    own load, so unpickling never fails on a stale path.

    Args:
        library_cls: The library estimator class.
        module: ``__name__`` of the calling module. The class is created under this module and must
            be bound there under ``name``; pickle resolves classes by module and qualified name, and
            a factory-made class living in this helper module would be unpicklable.
        name: The class name and the module-level attribute the caller binds the result to.
        seam: Only ``"load_model"`` exists today.
        load_method: The library method that builds the network.
        apply: ``apply(estimator, payload)`` sets what the library's method would have set. The
            default sets ``model_`` and ``model_path_`` for a bare module payload.
        returns: Whether ``load_method`` returns ``apply``'s result (a ``_build_model_specs`` style
            method) instead of ``None``.
        cpu_fallback_attr: Init parameter holding the requested device; a pickled CUDA value is
            rewritten to ``"cpu"`` on a host without CUDA before the library's ``__setstate__``
            runs. ``None`` disables the fallback.
        skip_when: Estimator attributes that force the library path when truthy.
    """
    if seam != "load_model":
        raise ValueError(f"unknown seam {seam!r}; only 'load_model' exists")
    apply_fn = apply or _default_apply
    library_load = getattr(library_cls, load_method)

    def use_shared_weights(self, key: WeightsKey, payload: Any, loader: Callable[..., Any]) -> None:
        """Make the next ``load_method`` attach ``payload`` (the registry entry for ``key``)."""
        self._tabarena_key = key
        self._tabarena_payload = payload
        self._tabarena_loader = loader

    def load(self, *args: Any, **kwargs: Any) -> Any:
        key = getattr(self, "_tabarena_key", None)
        if key is None or any(getattr(self, attr, None) for attr in skip_when):
            return library_load(self, *args, **kwargs)
        payload = getattr(self, "_tabarena_payload", None)
        device_type = estimator_device_type(self) or key.device
        if payload is None or payload_device(payload) != device_type:
            key = key.replace(device=device_type)
            if not os.path.exists(key.checkpoint):
                self._tabarena_key = None
                self._tabarena_payload = None
                return library_load(self, *args, **kwargs)
            payload = get_or_load(key, partial(self._tabarena_loader, key), stage="load")
            self._tabarena_key = key
        self._tabarena_payload = payload
        result = apply_fn(self, payload)
        return result if returns else None

    def getstate(self) -> dict:
        module_obj = sys.modules.get(type(self).__module__)
        if getattr(module_obj, type(self).__qualname__, None) is not type(self):
            raise RuntimeError(
                f"{type(self).__qualname__} must be bound at module level in {type(self).__module__} "
                "(derive_shared_estimator(..., module=__name__, name=<that name>)) so pickle can address it"
            )
        parent_getstate = getattr(super(derived, self), "__getstate__", None)
        state = parent_getstate() if parent_getstate is not None else self.__dict__
        state = dict(state) if state is not None else {}
        state.pop("_tabarena_payload", None)
        return state

    def setstate(self, state: dict) -> None:
        if cpu_fallback_attr is not None:
            requested = state.get(cpu_fallback_attr)
            if requested is not None and not isinstance(requested, (list, tuple)):
                import torch

                if normalize_device(requested) == "cuda" and not torch.cuda.is_available():
                    state[cpu_fallback_attr] = "cpu"
        parent_setstate = getattr(super(derived, self), "__setstate__", None)
        if parent_setstate is not None:
            parent_setstate(state)
        else:
            self.__dict__.update(state)

    namespace = {
        "__module__": module,
        "__qualname__": name,
        "__doc__": f"``{library_cls.__name__}`` whose ``{load_method}`` attaches the shared-weights registry payload.",
        DERIVED_MARKER: library_cls,
        "_tabarena_key": None,
        "_tabarena_payload": None,
        "_tabarena_loader": None,
        "use_shared_weights": use_shared_weights,
        load_method: load,
        "__getstate__": getstate,
        "__setstate__": setstate,
    }
    derived = type(name, (library_cls,), namespace)
    return derived


# --- state_dict mode -----------------------------------------------------------------------------


class SharedStateDictEstimatorMixin:
    """Estimator side of ``mode="state_dict"``: holds the cached CPU state dict for the next build.

    The wrapper calls :meth:`configure_shared_weights` with the registry's state dict before ``fit``
    and with ``None`` afterwards; the estimator's own build copies the tensors into a fresh module
    (see :func:`build_from_state_dict`). The state dict is never pickled with the estimator.
    """

    _shared_state_dict: Mapping[str, torch.Tensor] | None = None

    def configure_shared_weights(self, state_dict: Mapping[str, torch.Tensor] | None) -> None:
        """Use ``state_dict`` for the next build (``None`` restores the library's own checkpoint read)."""
        self._shared_state_dict = state_dict

    def __getstate__(self) -> dict:
        parent_getstate = getattr(super(), "__getstate__", None)
        state = parent_getstate() if parent_getstate is not None else self.__dict__
        state = dict(state) if state is not None else {}
        state.pop("_shared_state_dict", None)
        return state


def build_from_state_dict(
    module_factory: Callable[[], torch.nn.Module],
    state_dict: Mapping[str, torch.Tensor],
    device: str,
) -> torch.nn.Module:
    """Build ``module_factory()`` on the meta device, materialize it on ``device`` and copy ``state_dict`` in.

    ``load_state_dict`` copies tensor by tensor, so the module never aliases the cached tensors and
    fine-tuning it leaves the registry entry untouched.
    """
    import torch

    from tabarena.models._weights import copy_state_dict_into

    with torch.device("meta"):
        module = module_factory()
    module.to_empty(device=device)
    copy_state_dict_into(module, dict(state_dict))
    return module


# --- provenance and drift guards -----------------------------------------------------------------


def hf_checkpoint_source(
    *,
    repo_id: str | None,
    filename: str | None,
    revision: str | None,
    path: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Provenance of a Hub checkpoint without host paths: repo id, filename and the serving revision.

    ``revision`` is reported as given when pinned; otherwise the commit sha is read from ``path``
    when that is a snapshot path (``.../snapshots/<sha>/...``), and, for a ``filename`` resolved
    through the cache's default branch, from the cached snapshot link the Hub client knows
    (a local lookup, no request). ``None`` when neither is available.
    """
    resolved_revision = revision
    if resolved_revision is None and path is not None:
        resolved_revision = commit_from_snapshot_path(path)
    if resolved_revision is None and repo_id is not None and filename is not None:
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(repo_id, filename)
        except Exception:
            cached = None
        if isinstance(cached, str):
            resolved_revision = commit_from_snapshot_path(cached)
    return {"repo_id": repo_id, "filename": filename, "revision": resolved_revision, **extra}


def check_signature_matches(
    library_callable: Callable[..., Any],
    replica_callable: Callable[..., Any],
    *,
    skip: tuple[str, ...] = (),
) -> None:
    """Raise ``TypeError`` when the parameters of a replica differ from the library callable it mirrors.

    Compares parameter names, order, kinds and defaults, ignoring the names in ``skip`` (the
    replica's own additions such as ``model=``). The guard for constructor replicas that exist
    because a library offers no injection seam: a library bump that changes the constructor fails
    here instead of diverging silently.
    """

    def params(fn: Callable[..., Any]) -> list[tuple[str, Any, Any]]:
        return [
            (p.name, p.kind, p.default)
            for p in inspect.signature(fn).parameters.values()
            if p.name not in skip and p.name != "self"
        ]

    expected = params(library_callable)
    actual = params(replica_callable)
    if expected != actual:
        missing = [p[0] for p in expected if p not in actual]
        extra = [p[0] for p in actual if p not in expected]
        raise TypeError(
            f"{getattr(replica_callable, '__qualname__', replica_callable)} drifted from "
            f"{getattr(library_callable, '__qualname__', library_callable)}: library-only or changed {missing}, "
            f"replica-only or changed {extra}"
        )
