"""Read each task's label file once when several method artifacts are loaded together.

Every method's processed artifact carries its own copy of the per-task label files
(``label-val.csv.zip`` / ``label-test.csv.zip``), so loading a collection of ``n`` methods
parses each label file ``n`` times. That parsing dominated the load time of a BeyondArena
collection, and because pandas' CSV parser holds the GIL it also kept the thread pool that
loads the methods from scaling. :class:`LabelFileCache` keys the parsed frames by
``(dataset, fold, split)`` and reuses a frame when another artifact's file has the same
content, established from the CRC-32 and sizes stored in the zip's local file header (one
30-byte read instead of a full parse). Files that are not single-entry zips with a header
CRC bypass the cache and are parsed normally.

The cache is scoped with :func:`shared_label_files`; :meth:`ZeroshotSimulatorContext.load_groundtruth`
picks up the active one, so the eight call layers between a context's ``load_repo`` and the
label read need no extra parameter.
"""

from __future__ import annotations

import struct
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterator

_LOCAL_FILE_HEADER = struct.Struct("<IHHHHHIIIHH")
_LOCAL_FILE_HEADER_SIGNATURE = 0x04034B50
_FLAG_DATA_DESCRIPTOR = 0x8


def zip_entry_signature(path: str | Path) -> tuple[int, int, int] | None:
    """``(crc32, compressed_size, uncompressed_size)`` of a zip's first entry from its local
    file header, or ``None`` when the file is not a zip or the header carries no CRC (data
    descriptor flag set, as streamed writers do).
    """
    with open(path, "rb") as f:
        header = f.read(_LOCAL_FILE_HEADER.size)
    if len(header) < _LOCAL_FILE_HEADER.size:
        return None
    signature, _, flags, _, _, _, crc32, compressed_size, uncompressed_size, _, _ = _LOCAL_FILE_HEADER.unpack(header)
    if signature != _LOCAL_FILE_HEADER_SIGNATURE or flags & _FLAG_DATA_DESCRIPTOR:
        return None
    return crc32, compressed_size, uncompressed_size


class LabelFileCache:
    """Parsed label frames keyed by ``(dataset, fold, split)``, reused across artifact dirs
    whose files have the same zip content signature. Thread-safe; concurrent misses for the
    same key both parse and the first result is kept.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, int, str], tuple[tuple[int, int, int], pd.DataFrame]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def read(self, path: str | Path, *, dataset: str, fold: int, split: str) -> pd.DataFrame:
        signature = zip_entry_signature(path)
        key = (dataset, fold, split)
        if signature is not None:
            with self._lock:
                entry = self._entries.get(key)
            if entry is not None and entry[0] == signature:
                self.hits += 1
                return entry[1]
        df = pd.read_csv(path, index_col=0)
        self.misses += 1
        if signature is not None:
            with self._lock:
                self._entries.setdefault(key, (signature, df))
        return df


_active_cache: LabelFileCache | None = None
_active_lock = threading.Lock()


def get_active_label_cache() -> LabelFileCache | None:
    return _active_cache


@contextmanager
def shared_label_files() -> Iterator[LabelFileCache]:
    """Make every ``load_groundtruth`` inside the block read through one :class:`LabelFileCache`.

    Threads started inside the block (e.g. a method-loading pool) see the same cache. Blocks
    do not nest: the inner block's cache replaces the outer one until it exits.
    """
    global _active_cache
    cache = LabelFileCache()
    with _active_lock:
        previous = _active_cache
        _active_cache = cache
    try:
        yield cache
    finally:
        with _active_lock:
            _active_cache = previous
