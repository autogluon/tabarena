"""Keep a fitted model's context cache out of its pickle.

An in-context model that caches its encoded training rows (a KV cache, an encoded context) holds a
fitted state that grows with every training row, often by tens of KB per row. AutoGluon pickles a
model inside the timed fit, in memory to size it (``_get_pickled_size``) and to disk to save it, so a
cache that sits in the pickle is copied on every dataset: it inflates the measured training time
everywhere and doubles the RAM on the large tables (Seldon ran a 177 GB node out of memory on
GiveMeSomeCredit, autogluon/tabarena#701). :class:`ContextCacheFileMixin` writes the cache with
``torch.save`` beside ``model.pkl``, which streams the tensors without a copy, maps it back on load,
and counts its bytes for the size estimate without pickling it.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar


def tensor_nbytes(obj: Any) -> int:
    """Bytes of the distinct tensor storages reachable from ``obj``.

    Walks mappings, lists, tuples and object attributes, and counts each storage once, so a cache
    whose entries are views of one batched tensor is not counted twice.
    """
    import torch

    storages: dict[int, int] = {}
    seen: set[int] = set()

    def walk(value: Any) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if torch.is_tensor(value):
            storage = value.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
        elif isinstance(value, Mapping):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                walk(item)
        elif hasattr(value, "__dict__") and not isinstance(value, type):
            for item in vars(value).values():
                walk(item)

    walk(obj)
    return sum(storages.values())


class ContextCacheFileMixin:
    """Save a model's context cache in its own file next to ``model.pkl``, not inside the pickle.

    Put it before the AutoGluon base class (``class MyModel(ContextCacheFileMixin, AbstractTorchModel)``)
    and implement :meth:`_get_context_cache` and :meth:`_set_context_cache` for wherever the fitted
    model keeps its cache. Override :meth:`_context_cache_nbytes` when the cache type reports its own
    size and :meth:`_prepare_loaded_context_cache` to pin a CPU-offloaded cache for transfers.
    ``tabarena.tools.audit_pickle`` measures how much the pickle grows per training row, which is the
    check this mixin makes pass.
    """

    #: The cache file inside the model's directory.
    context_cache_file_name: ClassVar[str] = "context_cache.pt"

    def _get_context_cache(self) -> Any:
        """The fitted cache, or ``None`` when the model has none (not fitted, or caching is off)."""
        raise NotImplementedError

    def _set_context_cache(self, cache: Any) -> None:
        """Put ``cache`` (possibly ``None``) back where :meth:`_get_context_cache` reads it."""
        raise NotImplementedError

    def _context_cache_nbytes(self, cache: Any) -> int:
        """The cache's size in bytes, for AutoGluon's memory estimate."""
        return tensor_nbytes(cache)

    def _prepare_loaded_context_cache(self, cache: Any) -> Any:
        """The cache as the loaded model should hold it; the default keeps it as loaded."""
        return cache

    @contextlib.contextmanager
    def _without_context_cache(self) -> Iterator[Any]:
        """Detach the cache while ``self`` is pickled, so the pickle never copies it."""
        cache = self._get_context_cache()
        if cache is not None:
            self._set_context_cache(None)
        try:
            yield cache
        finally:
            if cache is not None:
                self._set_context_cache(cache)

    def save(self, path: str | None = None, verbose: bool = True) -> str:
        """Pickle the model without the cache and write the cache with ``torch.save``."""
        with self._without_context_cache() as cache:
            path = super().save(path=path, verbose=verbose)
        cache_path = os.path.join(path, self.context_cache_file_name)
        if cache is None:
            if os.path.exists(cache_path):
                os.remove(cache_path)
            return path
        import torch

        # Written beside the file and swapped in: a cache loaded on the CPU stays memory-mapped from this file, and
        # truncating it under the mapping kills the process with SIGBUS (refit_full re-saves a loaded model).
        torch.save(cache, cache_path + ".tmp")
        os.replace(cache_path + ".tmp", cache_path)
        return path

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True):
        """Load the model, then its cache memory-mapped from the file :meth:`save` wrote."""
        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        cache_path = os.path.join(path, cls.context_cache_file_name)
        if os.path.exists(cache_path):
            import torch

            # The cache file is as trusted as model.pkl.
            cache = torch.load(cache_path, mmap=True, weights_only=False)
            model._set_context_cache(model._prepare_loaded_context_cache(cache))
        return model

    def _get_pickled_size(self) -> int:
        """The pickle size without the cache plus the cache's bytes."""
        with self._without_context_cache() as cache:
            size = super()._get_pickled_size()
        return size + (self._context_cache_nbytes(cache) if cache is not None else 0)
