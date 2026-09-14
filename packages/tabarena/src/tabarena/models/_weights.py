"""Process-wide registry of pretrained checkpoint weights shared by foundation-model wrappers.

What is shared
    One immutable object per :class:`WeightsKey` per process, where the key names the checkpoint
    file, the estimator variant, the device type, the parameter dtype and the forward-affecting
    construction flags. The object is usually the built ``torch.nn.Module`` (in-context models
    never mutate their weights during a fit) but can be a state dict or a small container that the
    library's estimator accepts as its network. A bagged fit with 8 fold children and a refit child
    otherwise reads the same checkpoint and builds the same network 9 times, inside the timed fit.

Why it is fair
    Pretrained weights are part of the environment, like an imported library or a CUDA context:
    they are identical for every fit of that checkpoint and a served deployment keeps them resident.
    The untimed warm-up (:mod:`tabarena.models.warmup`) primes the registry through :func:`prime`,
    the timed fit takes a registry hit through :func:`get_or_load`, and every result records what
    happened (:meth:`WeightsEntry.to_metadata`, ``info["shared_weights"]`` of each child,
    ``report()`` in the method metadata). A key is derived from the problem type, the allocated
    hardware and the configured hyperparameters only, never from the task's data, so a primed
    entry carries no task-specific state into the fit.

What is never shared
    Anything written during a fit: fine-tuned parameters (a fine-tuning wrapper copies the cached
    state dict into a fresh module with :func:`copy_state_dict_into` and trains that copy), fitted
    context, KV caches, compiled graphs and other data-derived state. A wrapper whose configuration mutates the network in
    place (for example a library flag that compiles the module) returns ``None`` from
    ``shared_weights_key`` and loads its own network exactly as before.

Loader contract
    A loader is a zero-argument callable that builds the object for one key: resolve files through
    :func:`tabarena.models.prefetch.resolve_hf_file` or
    :func:`tabarena.models.prefetch.resolve_hf_snapshot` (local-first, so no Hub request lands in
    a timer), build on ``key.device`` with ``key.dtype``, call ``eval()`` and
    ``requires_grad_(False)`` on modules, and never read task data. The registry runs every loader
    with the Python, NumPy and torch (CPU and CUDA) random states saved and restored, so building a
    network never advances a global random number generator. A loader that raises registers
    nothing and the exception propagates to the caller.

Wrapper convention
    There is no mandatory mixin; each foundation-model wrapper implements the convention in its own
    class body so the pieces stay readable next to the library calls they wrap:

    * ``class_settings_cls = SharedWeightsClassSettings`` so that
      ``TabularPredictor.fit(model_class_settings={"<ag_key>": {"share_weights": False}})``
      switches sharing off for that class alone (AutoGluon resolves the settings owner to the
      nearest class declaring ``class_settings_cls``).
    * ``warmup_modules``: the modules the warm-up imports before the fit.
    * ``_default_shared_weights_params()``: the checkpoint-relevant literals of
      ``_set_default_params``, overlaid on the user hyperparameters before the key is derived.
    * ``shared_weights_key(*, problem_type, hyperparameters, device) -> WeightsKey | None``: the
      single source of truth for warm-up and fit; ``None`` disables sharing for that configuration.
    * ``_load_shared_weights(key)``: the classmethod loader.
    * ``_shares_network()``: False when ``ag.save_pretrained_weights`` is set, when the class
      settings disable sharing, or when the configuration is unshareable.
    * ``_ensure_network(device=None)``: reattaches the registry object to a weightless estimator.
    * ``__getstate__``: pickles the estimator without the shared object (see :func:`detach_attr`)
      only when the fit shared its network, so the refit child's save inside the timed fit no
      longer serializes the checkpoint.
    * ``_class_tags`` with ``set_device_on_save_to=None``: no CPU round-trip before a weightless
      save.
    * ``_set_device``: swaps the registry entry for the new device type and never calls ``.to()``
      on a shared module.
    * ``prepare_for_inference()``: untimed and idempotent; ``_ensure_network()``, ``eval()``, a
      CUDA synchronize.
    * ``_get_memory_size()``: ``_get_pickled_size() + tensor_bytes(network)``.
    * ``get_info()["shared_weights"]``: the key as a dict, ``loaded_by``, ``present_before_fit``
      and ``checkpoint_source``.

This module imports torch only inside the functions that need it, so importing it from a wrapper's
``model.py`` keeps ``import tabarena.models`` cheap.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import os
import random
import sys
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from autogluon.core.models.abstract._class_settings import ClassSettings

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    import torch

logger = logging.getLogger(__name__)

#: Which stage of a job first loaded a registry entry; the provenance the metadata reports.
LoadedBy = Literal["warmup", "fit", "load", "predict"]

#: Environment variable overriding the default registry capacity.
CAPACITY_ENV_VAR = "TABARENA_SHARED_WEIGHTS_CAPACITY"
DEFAULT_CAPACITY = 2


def normalize_device(device: str | torch.device | None) -> str:
    """The device type only, for example ``"cuda"`` for ``"cuda:0"``; ``None`` means ``"cpu"``.

    TabArena runs one GPU per job, so the index carries no information and wrappers that spell
    the device ``"cuda"`` share the entry with wrappers that spell it ``"cuda:0"``.
    """
    if device is None:
        return "cpu"
    return str(device).strip().lower().split(":")[0]


@dataclass(frozen=True)
class WeightsKey:
    """Identity of one immutable network in the process registry.

    Args:
        library: Short library name, for example ``"tabpfn"``, ``"tabicl"``, ``"mitra_v2"``.
        checkpoint: Resolved local path (the Hugging Face blob path after symlink resolution) or
            a stable opaque id for sources that are not a single file.
        variant: Estimator discriminator, for example ``"classifier"`` or ``"regressor"``.
        device: Device type from :func:`normalize_device`.
        dtype: Parameter dtype of the stored network, for example ``"float32"`` or ``"bfloat16"``.
        flags: Forward-affecting construction knobs as sorted ``(name, value)`` string pairs.
    """

    library: str
    checkpoint: str
    variant: str
    device: str
    dtype: str = "float32"
    flags: tuple[tuple[str, str], ...] = ()

    def replace(self, **changes: Any) -> WeightsKey:
        """A copy with ``changes`` applied; ``device`` is normalized, ``flags`` re-sorted."""
        if "device" in changes:
            changes["device"] = normalize_device(changes["device"])
        if "flags" in changes:
            changes["flags"] = _normalize_flags(changes["flags"])
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form with named fields; ``flags`` becomes a dict."""
        return {
            "library": self.library,
            "checkpoint": self.checkpoint,
            "variant": self.variant,
            "device": self.device,
            "dtype": self.dtype,
            "flags": dict(self.flags),
        }

    def short(self) -> str:
        """Compact log label: library, variant, device, dtype, checkpoint basename and flags."""
        label = f"{self.library}/{self.variant}@{self.device}:{self.dtype}[{Path(self.checkpoint).name}]"
        if self.flags:
            label += "{" + ",".join(f"{name}={value}" for name, value in self.flags) + "}"
        return label


def _normalize_flags(flags: Any) -> tuple[tuple[str, str], ...]:
    """Sorted ``(name, value-as-str)`` pairs from a mapping or an iterable of pairs."""
    items = flags.items() if hasattr(flags, "items") else flags
    return tuple(sorted((str(name), str(value)) for name, value in items))


def make_key(
    library: str,
    checkpoint: str | Path,
    variant: str,
    device: str | torch.device | None,
    *,
    dtype: str = "float32",
    **flags: Any,
) -> WeightsKey:
    """Build a :class:`WeightsKey`, resolving a path checkpoint and normalizing device and flags.

    A ``checkpoint`` that names an existing path is resolved to its absolute target (symlinks
    followed, so a Hugging Face snapshot link and its blob agree); any other string is kept as an
    opaque stable id.
    """
    if isinstance(checkpoint, Path) or os.path.exists(str(checkpoint)):
        checkpoint = str(Path(checkpoint).resolve())
    return WeightsKey(
        library=library,
        checkpoint=str(checkpoint),
        variant=variant,
        device=normalize_device(device),
        dtype=dtype,
        flags=_normalize_flags(flags),
    )


@dataclass
class WeightsEntry:
    """One registered object with its load provenance.

    Args:
        key: The registry key.
        value: The shared object (a module, a state dict or a small container).
        loaded_by: Stage that ran the loader.
        load_time_s: Wall-clock seconds the loader took.
        n_bytes: Tensor bytes of ``value`` when they could be counted, else ``None``.
        source: Where the checkpoint came from (repo id, filename, revision, snapshot sha) for the
            uploaded metadata, which must not depend on absolute host paths.
        hits: Registry hits since the entry was loaded or the stats were reset.
    """

    key: WeightsKey
    value: Any
    loaded_by: LoadedBy
    load_time_s: float
    n_bytes: int | None
    source: dict[str, Any] = field(default_factory=dict)
    hits: int = 0

    def to_metadata(self) -> dict[str, Any]:
        """JSON-able record without the object reference, for uploaded results.

        ``checkpoint`` is reduced to its basename (for a Hugging Face blob that is the content
        sha256, a stable identifier) so the record carries no absolute host path; ``source`` holds
        the repo id and filename. :meth:`WeightsKey.to_dict` keeps the absolute path and is meant
        for logs and in-process use only.
        """
        return {
            **self.key.to_dict(),
            "checkpoint": Path(self.key.checkpoint).name,
            "loaded_by": self.loaded_by,
            "load_time_s": self.load_time_s,
            "n_bytes": self.n_bytes,
            "source": dict(self.source),
            "hits": self.hits,
        }


_ENTRIES: OrderedDict[WeightsKey, WeightsEntry] = OrderedDict()
_LOCK = threading.RLock()
_CAPACITY: int | None = None


def _fresh_stats() -> dict[str, int]:
    return {"hits": 0, "misses": 0, "primes": 0, "evictions": 0, "evicted_primed": 0}


_STATS: dict[str, int] = _fresh_stats()


def capacity() -> int:
    """Entries the registry keeps; ``DEFAULT_CAPACITY`` unless ``CAPACITY_ENV_VAR`` overrides it."""
    global _CAPACITY
    with _LOCK:
        if _CAPACITY is None:
            raw = os.environ.get(CAPACITY_ENV_VAR)
            try:
                _CAPACITY = DEFAULT_CAPACITY if raw is None else max(0, int(raw))
            except ValueError:
                logger.warning("Ignoring %s=%r (not an integer); using %d.", CAPACITY_ENV_VAR, raw, DEFAULT_CAPACITY)
                _CAPACITY = DEFAULT_CAPACITY
        return _CAPACITY


def set_capacity(n: int) -> None:
    """Set the capacity and evict least recently used entries beyond it; ``0`` stores nothing."""
    global _CAPACITY
    with _LOCK:
        _CAPACITY = max(0, int(n))
        _evict_to(_CAPACITY)


def _evict_to(n: int) -> None:
    while len(_ENTRIES) > n:
        _, entry = _ENTRIES.popitem(last=False)
        _STATS["evictions"] += 1
        if entry.loaded_by == "warmup":
            _STATS["evicted_primed"] += 1
            logger.warning(
                "Shared weights %s primed by the warm-up were evicted (capacity %d); the next fit of that "
                "checkpoint loads inside the timer. Raise %s or release unused entries.",
                entry.key.short(),
                n,
                CAPACITY_ENV_VAR,
            )
        else:
            logger.info(
                "Shared weights %s (loaded by %s) evicted (capacity %d).", entry.key.short(), entry.loaded_by, n
            )


def _warn_if_primed_elsewhere(key: WeightsKey) -> None:
    """Warn when the warm-up primed the same checkpoint on another device or dtype than the fit wants."""
    for other in _ENTRIES.values():
        if (
            other.loaded_by == "warmup"
            and other.key.library == key.library
            and other.key.checkpoint == key.checkpoint
            and other.key.variant == key.variant
        ):
            logger.warning(
                "The warm-up primed %s but the fit needs %s; the primed entry is unused and the fit loads inside "
                "the timer. The warm-up and the fit must derive the same key.",
                other.key.short(),
                key.short(),
            )


@contextmanager
def _rng_guard(*, cuda: bool) -> Iterator[None]:
    """Save and restore the Python, NumPy and torch random states around a loader.

    The torch generators are forked with ``torch.random.fork_rng`` over the CPU and, when ``cuda``
    is requested and available, every CUDA device. Torch is imported here when it is importable so
    the fork also covers a loader that imports torch itself.
    """
    py_state = random.getstate()
    np_module = sys.modules.get("numpy")
    np_state = np_module.random.get_state() if np_module is not None else None
    try:
        import torch
    except ImportError:
        torch = None
    try:
        if torch is None:
            yield
        else:
            devices = list(range(torch.cuda.device_count())) if cuda and torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                yield
    finally:
        random.setstate(py_state)
        if np_state is not None:
            np_module.random.set_state(np_state)


def _run_loader(key: WeightsKey, loader: Callable[[], Any]) -> tuple[Any, float]:
    """Run ``loader`` under the RNG guard; returns the value and the load time in seconds."""
    torch_module = sys.modules.get("torch")
    cuda_initialized = bool(torch_module is not None and torch_module.cuda.is_initialized())
    start = time.perf_counter()
    with _rng_guard(cuda=key.device == "cuda" or cuda_initialized):
        value = loader()
    return value, time.perf_counter() - start


def get_or_load(
    key: WeightsKey,
    loader: Callable[[], Any],
    *,
    stage: LoadedBy = "fit",
    source: dict[str, Any] | None = None,
) -> Any:
    """The registered object for ``key``; on a miss run ``loader`` under the RNG guard and register it.

    The lock is held while the loader runs, so concurrent callers of the same key share one load.
    When ``capacity()`` is ``0`` the value is returned without being stored. A raising loader
    registers nothing and its exception propagates.

    Args:
        key: Registry key.
        loader: Zero-argument callable building the object (see the module's loader contract).
        stage: Stage recorded as the entry's ``loaded_by`` on a miss.
        source: Provenance recorded on the entry (repo id, filename, revision, snapshot sha).
    """
    with _LOCK:
        entry = _ENTRIES.get(key)
        if entry is not None:
            _ENTRIES.move_to_end(key)
            entry.hits += 1
            _STATS["hits"] += 1
            logger.debug("Shared weights hit for %s (stage %s).", key.short(), stage)
            return entry.value
        return _load_entry(key, loader, stage=stage, source=source).value


def _load_entry(
    key: WeightsKey,
    loader: Callable[[], Any],
    *,
    stage: LoadedBy,
    source: dict[str, Any] | None,
) -> WeightsEntry:
    """Miss path shared by :func:`get_or_load` and :func:`prime`; the caller holds the lock.

    Runs ``loader`` under the RNG guard, counts the miss and builds the entry with its real load
    time, byte estimate and provenance. The entry is stored (and the LRU trimmed) only when
    ``capacity()`` is positive; it is returned either way, so a transient entry still reports what
    the loader did.
    """
    if stage != "warmup":
        _warn_if_primed_elsewhere(key)
    value, load_time_s = _run_loader(key, loader)
    _STATS["misses"] += 1
    entry = WeightsEntry(
        key=key,
        value=value,
        loaded_by=stage,
        load_time_s=load_time_s,
        n_bytes=_estimate_bytes(value),
        source=dict(source or {}),
    )
    if capacity() > 0:
        _ENTRIES[key] = entry
        _evict_to(capacity())
    logger.info("Shared weights %s loaded by %s in %.2fs.", key.short(), stage, load_time_s)
    return entry


def prime(key: WeightsKey, loader: Callable[[], Any], *, source: dict[str, Any] | None = None) -> WeightsEntry:
    """Warm-up entry point: load ``key`` with ``loaded_by="warmup"`` unless it is already present.

    Idempotent: a present key is returned untouched (no hit is counted). A raising loader
    registers nothing and re-raises. With ``capacity()`` ``0`` the returned entry is not stored
    but still carries the loader's load time, byte estimate and ``source``.
    """
    with _LOCK:
        existing = _ENTRIES.get(key)
        if existing is not None:
            return existing
        entry = _load_entry(key, loader, stage="warmup", source=source)
        _STATS["primes"] += 1
        return entry


def contains(key: WeightsKey) -> bool:
    """Whether ``key`` is registered (no LRU touch)."""
    with _LOCK:
        return key in _ENTRIES


def loaded_by(key: WeightsKey) -> LoadedBy | None:
    """Stage that loaded ``key``, or ``None`` when it is not registered."""
    with _LOCK:
        entry = _ENTRIES.get(key)
        return None if entry is None else entry.loaded_by


def peek(key: WeightsKey) -> WeightsEntry | None:
    """The entry for ``key`` without touching the LRU order or the hit counters."""
    with _LOCK:
        return _ENTRIES.get(key)


def release(*, keys: Iterable[WeightsKey] | None = None) -> int:
    """Drop the registry's references to all entries (or to ``keys``); returns how many were dropped.

    Live estimators keep their own reference, so the memory is freed once the last of them is
    gone. When something was dropped, the garbage collector runs and, if CUDA is initialized, the
    caching allocator returns freed blocks to the device.
    """
    with _LOCK:
        if keys is None:
            dropped = len(_ENTRIES)
            _ENTRIES.clear()
        else:
            dropped = 0
            for key in list(keys):
                if _ENTRIES.pop(key, None) is not None:
                    dropped += 1
    if dropped:
        _free_memory()
    return dropped


def release_if_library_changes(library: str) -> int:
    """Release every entry of a library other than ``library``; returns how many were dropped.

    Meant for in-process sweeps that fit one model class after another: a SLURM job fits one
    method per process and never needs it.
    """
    with _LOCK:
        stale = [key for key in _ENTRIES if key.library != library]
    return release(keys=stale) if stale else 0


def _free_memory() -> None:
    gc.collect()
    torch_module = sys.modules.get("torch")
    if torch_module is not None and torch_module.cuda.is_initialized():
        torch_module.cuda.empty_cache()


def report() -> dict[str, Any]:
    """JSON-able snapshot: ``capacity``, ``stats``, ``evicted_primed`` and every entry's metadata."""
    with _LOCK:
        return {
            "capacity": capacity(),
            "stats": dict(_STATS),
            "evicted_primed": _STATS["evicted_primed"],
            "entries": [entry.to_metadata() for entry in _ENTRIES.values()],
        }


def reset_stats() -> None:
    """Zero the counters (including each entry's ``hits``); the entries themselves stay registered."""
    with _LOCK:
        _STATS.clear()
        _STATS.update(_fresh_stats())
        for entry in _ENTRIES.values():
            entry.hits = 0


def _estimate_bytes(value: Any) -> int | None:
    """Tensor bytes of a module or a mapping of tensors; ``None`` for anything else."""
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return None
    if isinstance(value, torch_module.nn.Module):
        return tensor_bytes(value)
    if isinstance(value, dict) and value and all(isinstance(v, torch_module.Tensor) for v in value.values()):
        return tensor_bytes(value)
    return None


def _iter_tensors(obj: Any) -> Iterator[torch.Tensor]:
    import torch

    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, torch.nn.Module):
        yield from obj.parameters()
        yield from obj.buffers()
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_tensors(value)
    else:
        for item in obj:
            yield from _iter_tensors(item)


def tensor_bytes(obj: torch.nn.Module | dict[str, torch.Tensor] | Iterable[Any]) -> int:
    """Bytes held by the tensors of ``obj``, each underlying storage counted once.

    ``obj`` is a module (parameters and buffers), a state dict, a tensor or an iterable of those.
    Tensors that view the same storage (tied weights, slices of one buffer) are counted once at the
    storage's size, which is what the device actually holds. Tensors without a real storage
    address (a module built under ``torch.device("meta")``, whose storages all report data pointer
    0) are identified by the tensor object instead, so a meta-built module reports the bytes it
    will hold after ``to_empty``.
    """
    seen: set[Any] = set()
    total = 0
    for tensor in _iter_tensors(obj):
        try:
            storage = tensor.untyped_storage()
            data_ptr = storage.data_ptr()
            ident: Any = id(tensor) if data_ptr == 0 else (tensor.device, data_ptr)
            n_bytes = storage.nbytes()
        except (RuntimeError, NotImplementedError, AttributeError):
            ident = id(tensor)
            n_bytes = tensor.numel() * tensor.element_size()
        if ident not in seen:
            seen.add(ident)
            total += n_bytes
    return total


def shallow_copy(obj: Any) -> Any:
    """A new instance of ``type(obj)`` sharing every attribute; ``__init__`` and ``__getstate__`` are not run."""
    cls = type(obj)
    try:
        new = object.__new__(cls)
    except TypeError:  # built-in types such as SimpleNamespace refuse object.__new__
        new = cls.__new__(cls)
    new.__dict__.update(obj.__dict__)
    return new


def detach_attr(obj: Any, name: str) -> Any:
    """A shallow copy of ``obj`` with attribute ``name`` set to ``None``; ``obj`` itself keeps it.

    The building block of a weightless ``__getstate__``: pickle the copy, keep the live estimator.
    """
    copy = shallow_copy(obj)
    setattr(copy, name, None)
    return copy


def load_state_dict_file(
    path: str | Path,
    device: str = "cpu",
    *,
    pin_memory: bool = False,
) -> dict[str, torch.Tensor]:
    """Read a checkpoint file into a state dict on ``device``.

    ``.safetensors`` files go through ``safetensors.torch.load_file``; everything else through
    ``torch.load(map_location=device, weights_only=True)``. With ``pin_memory`` and CUDA available,
    CPU tensors are pinned so a later host-to-device copy runs asynchronously; a state dict cached
    in pinned host memory is what a fine-tuning wrapper copies into each child's own module.
    """
    import torch

    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(path), device=device)
    else:
        state_dict = torch.load(str(path), map_location=device, weights_only=True)
    if pin_memory and torch.cuda.is_available():
        state_dict = {
            name: tensor.pin_memory() if tensor.device.type == "cpu" else tensor for name, tensor in state_dict.items()
        }
    return state_dict


def copy_state_dict_into(module: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Copy ``state_dict`` into ``module``'s own storage; the cached tensors are left untouched.

    ``module`` must own materialized parameters (call ``to_empty(device=...)`` on a module built
    under ``torch.device("meta")`` first); ``load_state_dict`` then copies tensor by tensor, so
    training the module never aliases or mutates the shared cache.
    """
    module.load_state_dict(state_dict, strict=True)


@dataclass(frozen=True)
class SharedWeightsClassSettings(ClassSettings):
    """Process-wide switch for one wrapper class; declare ``class_settings_cls`` on each wrapper.

    ``share_weights=False`` makes every model of that class load its own network through the
    library path and pickle it as before. Set through
    ``TabularPredictor.fit(model_class_settings={"<ag_key>": {"share_weights": False}})``; the
    snapshot travels with each model, so fold workers see the same value. The switch is scoped to
    the class whose body declares ``class_settings_cls``, so each wrapper declares it itself.
    """

    share_weights: bool = True
