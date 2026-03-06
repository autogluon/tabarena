from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from autogluon.common import TabularDataset

T = TypeVar("T")


class CacheMode(str, Enum):
    """How to interact with the on-disk cache."""
    USE_IF_EXISTS = "use_if_exists"        # load if exists else compute+save
    OVERWRITE = "overwrite"                # always compute+save (ignore existing)
    LOAD_ONLY = "load_only"                # require cache; error if missing


@dataclass(frozen=True)
class DiskCache(Generic[T]):
    """
    Generic on-disk cache for a single artifact.

    Parameters
    ----------
    path
        Where the artifact is stored.
    load_fn
        Callable that loads from `path` and returns the artifact.
    save_fn
        Callable that saves `obj` to `path`.
    """
    path: Path
    load_fn: Callable[[Path], T]
    save_fn: Callable[[Path, T], None]

    def get(
        self,
        compute_fn: Callable[[], T],
        mode: CacheMode = CacheMode.USE_IF_EXISTS,
        *,
        mkdir: bool = True,
    ) -> T:
        """
        Load from cache or compute + save, depending on `mode`.
        """
        path = Path(self.path)

        if mode == CacheMode.LOAD_ONLY:
            if not path.exists():
                raise FileNotFoundError(f"Cache missing: {path}")
            return self.load_fn(path)

        if mode == CacheMode.USE_IF_EXISTS and path.exists():
            return self.load_fn(path)

        # OVERWRITE, or USE_IF_EXISTS but missing
        obj = compute_fn()
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.save_fn(path, obj)
        # Optionally re-load to ensure round-trip consistency (mirrors your manual pattern)
        return self.load_fn(path)


# ---- Optional: convenience helper if you don't want to instantiate DiskCache each time ----

def cached(
    *,
    path: Path,
    compute_fn: Callable[[], T],
    load_fn: Callable[[Path], T],
    save_fn: Callable[[Path, T], None],
    mode: CacheMode = CacheMode.USE_IF_EXISTS,
    mkdir: bool = True,
) -> T:
    return DiskCache[T](path=path, load_fn=load_fn, save_fn=save_fn).get(
        compute_fn=compute_fn,
        mode=mode,
        mkdir=mkdir,
    )


def cached_parquet_df(
    *,
    path: Path,
    compute_fn: Callable[[], Any],
    mode: CacheMode = CacheMode.USE_IF_EXISTS,
) -> Any:
    return cached(
        path=path,
        compute_fn=compute_fn,
        load_fn=lambda p: TabularDataset.load(path=p),
        save_fn=lambda p, df: TabularDataset.save(path=p, df=df),
        mode=mode,
    )
