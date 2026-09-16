"""On-disk format of a task's labels: one ``labels.dat`` per ``<dataset>/<fold>`` directory.

Layout: a little-endian uint32 header length, a UTF-8 JSON header
``{"version": 1, "val": {"n": ..., "dtype": ...}, "test": {"n": ..., "dtype": ...}}`` and then the raw
validation labels followed by the raw test labels, each as a contiguous array of the header's
numpy dtype string. Integer labels are stored in the narrowest signed integer type that holds
them; floats keep their dtype. Only the label values are stored, in row order; the original row
ids that the earlier zipped-CSV files carried as their index are not.

The earlier layout, ``label-val.csv.zip`` and ``label-test.csv.zip`` per task, stays readable
through :func:`read_legacy_labels`; :func:`read_task_labels` prefers ``labels.dat`` and falls back
to it, writing ``labels.dat`` next to the legacy files when asked so later loads take the fast
path. The raw format loads without parsing (a single read and two array views per task), which
is what made it 4x faster than the zipped CSVs on the benchmark artifacts at a smaller size.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LABELS_FILENAME = "labels.dat"
LEGACY_LABEL_FILENAMES = ("label-val.csv.zip", "label-test.csv.zip")
_HEADER_LEN = struct.Struct("<I")
_VERSION = 1


def narrow_int_dtype(values: np.ndarray) -> np.dtype:
    """The narrowest signed integer dtype that holds every value (int8 for class ids)."""
    if values.size == 0:
        return np.dtype(np.int8)
    lo, hi = int(values.min()), int(values.max())
    for dtype in (np.int8, np.int16, np.int32, np.int64):
        info = np.iinfo(dtype)
        if info.min <= lo and hi <= info.max:
            return np.dtype(dtype)
    raise ValueError(f"label values out of int64 range: [{lo}, {hi}]")


def as_label_array(labels) -> np.ndarray:
    """Normalize a label container (Series, single-column DataFrame or array) to a 1-D array,
    narrowing integer labels. This is the in-memory form :class:`GroundTruth` keeps.
    """
    if isinstance(labels, np.ndarray) and labels.ndim == 1 and labels.flags.c_contiguous:
        # Already in the canonical form: return the same object, so arrays shared between
        # repositories (see ``merge_ground_truth``) stay shared through a GroundTruth constructor.
        if labels.dtype.kind not in "iu" or labels.dtype == narrow_int_dtype(labels):
            return labels
    values = np.asarray(labels.to_numpy() if hasattr(labels, "to_numpy") else labels).reshape(-1)
    if values.dtype.kind in "iu":
        values = values.astype(narrow_int_dtype(values), copy=False)
    return np.ascontiguousarray(values)


def widen_labels(values: np.ndarray) -> np.ndarray:
    """Integer labels as int64 (what the zipped-CSV files produced through pandas); other dtypes unchanged."""
    if values.dtype.kind in "iu":
        return values.astype(np.int64, copy=False)
    return values


def encode_labels(labels_val: np.ndarray, labels_test: np.ndarray) -> bytes:
    labels_val, labels_test = as_label_array(labels_val), as_label_array(labels_test)
    for name, values in (("val", labels_val), ("test", labels_test)):
        if values.dtype.kind not in "iufb":
            raise TypeError(f"labels.dat stores numeric or bool labels; {name} labels have dtype {values.dtype}")
    header = {
        "version": _VERSION,
        "val": {"n": int(labels_val.size), "dtype": labels_val.dtype.str},
        "test": {"n": int(labels_test.size), "dtype": labels_test.dtype.str},
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return b"".join([_HEADER_LEN.pack(len(header_bytes)), header_bytes, labels_val.tobytes(), labels_test.tobytes()])


def decode_labels(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`encode_labels`; the arrays are read-only views into ``blob``."""
    (header_len,) = _HEADER_LEN.unpack_from(blob, 0)
    start = _HEADER_LEN.size
    header = json.loads(blob[start : start + header_len].decode("utf-8"))
    if header.get("version") != _VERSION:
        raise ValueError(f"unsupported labels.dat version {header.get('version')!r}")
    offset = start + header_len
    arrays = []
    for split in ("val", "test"):
        dtype = np.dtype(header[split]["dtype"])
        n = int(header[split]["n"])
        arrays.append(np.frombuffer(blob, dtype=dtype, count=n, offset=offset))
        offset += n * dtype.itemsize
    if offset != len(blob):
        raise ValueError(f"labels.dat has {len(blob)} bytes, header describes {offset}")
    return arrays[0], arrays[1]


def write_labels_dat(task_dir: str | Path, labels_val: np.ndarray, labels_test: np.ndarray) -> Path:
    """Write ``labels.dat`` atomically (temporary file + rename, so concurrent writers and readers
    never see a partial file).
    """
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / LABELS_FILENAME
    tmp_path = task_dir / f"{LABELS_FILENAME}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        with open(tmp_path, "wb") as f:
            f.write(encode_labels(labels_val, labels_test))
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return path


def read_labels_dat_bytes(task_dir: str | Path) -> bytes:
    with open(Path(task_dir) / LABELS_FILENAME, "rb") as f:
        return f.read()


def read_legacy_labels(task_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the zipped-CSV pair of the earlier layout as label arrays (row ids dropped)."""
    task_dir = Path(task_dir)
    val = pd.read_csv(task_dir / LEGACY_LABEL_FILENAMES[0], index_col=0)
    test = pd.read_csv(task_dir / LEGACY_LABEL_FILENAMES[1], index_col=0)
    return as_label_array(val), as_label_array(test)


def read_task_labels(task_dir: str | Path, generate: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Labels of one task: ``labels.dat`` when present, else the legacy zipped CSVs.

    Only the fast path's file is probed; the legacy files are consulted when it is missing.
    With ``generate``, a legacy read also writes ``labels.dat`` next to the legacy files so the
    next load takes the fast path; a directory that cannot be written (read-only install or
    container) is left as is.
    """
    try:
        blob = read_labels_dat_bytes(task_dir)
    except FileNotFoundError:
        labels_val, labels_test = read_legacy_labels(task_dir)
        if generate:
            try:
                write_labels_dat(task_dir, labels_val, labels_test)
            except OSError as e:
                logger.debug("could not write %s in %s: %s", LABELS_FILENAME, task_dir, e)
        return labels_val, labels_test
    return decode_labels(blob)
