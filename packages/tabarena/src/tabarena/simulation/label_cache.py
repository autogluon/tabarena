"""Read each task's labels once when several method artifacts are loaded together.

Every method's processed artifact carries its own copy of the per-task label files, so loading
a collection of ``n`` methods read each task's labels ``n`` times. :class:`LabelFileCache` keys
the loaded arrays by ``(dataset, fold)`` and reuses them when another artifact's file has the
same content signature. The cache is scoped with :func:`shared_label_files`;
:meth:`ZeroshotSimulatorContext.load_groundtruth` picks up the active one, so the call layers
between a context's ``load_repo`` and the label read need no extra parameter.
"""

from __future__ import annotations

import logging
import struct
import threading
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tabarena.simulation.label_files import (
    LABELS_FILENAME,
    LEGACY_LABEL_FILENAMES,
    decode_labels,
    read_labels_dat_bytes,
    read_task_labels,
    write_labels_dat,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


logger = logging.getLogger(__name__)

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
    """Label arrays keyed by ``(dataset, fold)``, reused across artifact dirs whose files have the
    same content signature. Thread-safe; concurrent misses for the same key both read and the
    first result is kept.

    The signature of a ``labels.dat`` is the CRC-32 of its bytes (the file is read in full anyway,
    the read is what costs); for the legacy zipped-CSV pair it is the CRC-32 and sizes from the two
    zip headers, so a repeated legacy read skips the parse.
    """

    def __init__(self, generate: bool = True) -> None:
        self._entries: dict[tuple[str, int], tuple[object, tuple[np.ndarray, np.ndarray]]] = {}
        self._lock = threading.Lock()
        self.generate = generate
        self.hits = 0
        self.misses = 0

    def _lookup(self, key, signature):
        with self._lock:
            entry = self._entries.get(key)
        if entry is not None and entry[0] == signature:
            self.hits += 1
            return entry[1]
        return None

    @staticmethod
    def _generate(task_dir: Path, labels: tuple[np.ndarray, np.ndarray]) -> None:
        try:
            write_labels_dat(task_dir, *labels)
        except OSError as e:
            logger.debug("could not write %s in %s: %s", LABELS_FILENAME, task_dir, e)

    def _store(self, key, signature, labels):
        self.misses += 1
        with self._lock:
            self._entries.setdefault(key, (signature, labels))
        return labels

    def share(
        self, *, dataset: str, fold: int, labels_val: np.ndarray, labels_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the cached arrays for this task when their content matches ``labels_val`` /
        ``labels_test`` (sizes, dtypes and CRC-32 of the bytes), else cache and return the given ones.
        Lets artifacts loaded together hold one array per task instead of a copy each.
        """
        signature = (
            "arrays",
            labels_val.dtype.str,
            labels_val.size,
            zlib.crc32(np.ascontiguousarray(labels_val)),
            labels_test.dtype.str,
            labels_test.size,
            zlib.crc32(np.ascontiguousarray(labels_test)),
        )
        key = (dataset, fold)
        cached = self._lookup(key, signature)
        if cached is not None:
            return cached
        return self._store(key, signature, (labels_val, labels_test))

    def read_task(self, task_dir: str | Path, *, dataset: str, fold: int) -> tuple[np.ndarray, np.ndarray]:
        """``(labels_val, labels_test)`` of one task directory (see :func:`read_task_labels`)."""
        task_dir = Path(task_dir)
        key = (dataset, fold)
        try:
            blob = read_labels_dat_bytes(task_dir)
        except FileNotFoundError:
            signature = ("legacy", *(zip_entry_signature(task_dir / name) for name in LEGACY_LABEL_FILENAMES))
            if None not in signature[1:]:
                cached = self._lookup(key, signature)
                if cached is not None:
                    if self.generate:
                        # The legacy files match the cached labels, so this directory can be
                        # converted from them without a parse; otherwise only the first
                        # artifact of a collection would be converted per load.
                        self._generate(task_dir, cached)
                    return cached
            labels = read_task_labels(task_dir, generate=self.generate)
            return self._store(key, signature, labels)
        signature = ("dat", len(blob), zlib.crc32(blob))
        cached = self._lookup(key, signature)
        if cached is not None:
            return cached
        return self._store(key, signature, decode_labels(blob))


_active_cache: LabelFileCache | None = None
_active_lock = threading.Lock()


def get_active_label_cache() -> LabelFileCache | None:
    return _active_cache


@contextmanager
def shared_label_files(generate: bool = True) -> Iterator[LabelFileCache]:
    """Make every ``load_groundtruth`` inside the block read through one :class:`LabelFileCache`.

    Threads started inside the block (e.g. a method-loading pool) see the same cache. Blocks
    do not nest: the inner block's cache replaces the outer one until it exits. ``generate``
    is forwarded to :func:`read_task_labels` for legacy artifacts.
    """
    global _active_cache
    cache = LabelFileCache(generate=generate)
    with _active_lock:
        previous = _active_cache
        _active_cache = cache
    try:
        yield cache
    finally:
        with _active_lock:
            _active_cache = previous
