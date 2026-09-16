"""Per-dataset task file: one ``tasks.dat`` per ``<dataset>`` directory holding every fold's
prediction metadata (what ``metadata.json`` held per task) and labels (what ``labels.dat`` held).

Layout: a little-endian uint32 header length, a UTF-8 JSON header::

    {"version": 1, "dataset": ..., "folds": {"<fold>": {"models": [...], "pred_val_shape": [...],
     "pred_test_shape": [...], "dtype": ..., "labels": {"val": {"offset", "n", "dtype"},
     "test": {...}}}}}

and then the raw label arrays of all folds. The prediction memmaps stay per task
(``<dataset>/<fold>/pred-{val,test}.dat``): they are read lazily and are the bulk of an artifact.

Why per dataset: after the labels moved to raw arrays, loading an artifact was bounded by the
number of files opened, not bytes, and every task cost two opens (``metadata.json`` and
``labels.dat``) on a network filesystem, about 6x more when its caches are cold. This layout
opens one file per dataset (142 instead of 7444 on the BeyondArena artifacts).

Writers keep producing per-task files (``to_dir_task_data``: disjoint directories, so parallel
workers can write task slices into one tree) and :func:`consolidate_task_data` folds them into
``tasks.dat`` when the artifact is finalized (``write_processed_context``). Readers prefer
``tasks.dat``, read per-task files for folds it lacks, and generate it when every expected fold is
available (see :func:`load_dataset_tasks`).
"""

from __future__ import annotations

import json
import logging
import os
import struct
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tabarena.simulation.label_files import LABELS_FILENAME, as_label_array, read_task_labels

if TYPE_CHECKING:
    from tabarena.simulation.label_cache import LabelFileCache

logger = logging.getLogger(__name__)

TASKS_FILENAME = "tasks.dat"
METADATA_FILENAME = "metadata.json"
_HEADER_LEN = struct.Struct("<I")
_VERSION = 1
_METADATA_KEYS = ("models", "pred_val_shape", "pred_test_shape", "dtype")


@dataclass
class TaskData:
    """One task's prediction metadata (``models``, ``pred_val_shape``, ``pred_test_shape``,
    ``dtype``) and its label arrays.
    """

    metadata: dict
    labels_val: np.ndarray
    labels_test: np.ndarray


def encode_dataset_tasks(dataset: str, tasks: dict[int, TaskData]) -> bytes:
    folds_header: dict[str, dict] = {}
    chunks: list[bytes] = []
    offset = 0
    for fold in sorted(tasks):
        task = tasks[fold]
        entry = {key: task.metadata[key] for key in _METADATA_KEYS}
        labels = {}
        for split, values in (("val", task.labels_val), ("test", task.labels_test)):
            values = as_label_array(values)
            if values.dtype.kind not in "iufb":
                raise TypeError(f"{TASKS_FILENAME} stores numeric or bool labels; got {values.dtype}")
            labels[split] = {"offset": offset, "n": int(values.size), "dtype": values.dtype.str}
            chunks.append(values.tobytes())
            offset += values.nbytes
        entry["labels"] = labels
        folds_header[str(int(fold))] = entry
    header = json.dumps({"version": _VERSION, "dataset": dataset, "folds": folds_header}, separators=(",", ":"))
    header_bytes = header.encode("utf-8")
    return b"".join([_HEADER_LEN.pack(len(header_bytes)), header_bytes, *chunks])


def decode_dataset_tasks(blob: bytes) -> tuple[str, dict[int, TaskData]]:
    """Inverse of :func:`encode_dataset_tasks`; label arrays are read-only views into ``blob``."""
    (header_len,) = _HEADER_LEN.unpack_from(blob, 0)
    start = _HEADER_LEN.size
    header = json.loads(blob[start : start + header_len].decode("utf-8"))
    if header.get("version") != _VERSION:
        raise ValueError(f"unsupported {TASKS_FILENAME} version {header.get('version')!r}")
    base = start + header_len
    tasks: dict[int, TaskData] = {}
    for fold_str, entry in header["folds"].items():
        arrays = []
        for split in ("val", "test"):
            spec = entry["labels"][split]
            dtype = np.dtype(spec["dtype"])
            arrays.append(np.frombuffer(blob, dtype=dtype, count=int(spec["n"]), offset=base + int(spec["offset"])))
        metadata = {key: entry[key] for key in _METADATA_KEYS}
        tasks[int(fold_str)] = TaskData(metadata=metadata, labels_val=arrays[0], labels_test=arrays[1])
    return header["dataset"], tasks


def write_tasks_dat(dataset_dir: str | Path, dataset: str, tasks: dict[int, TaskData]) -> Path:
    """Write ``tasks.dat`` atomically (temporary file + rename)."""
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / TASKS_FILENAME
    tmp_path = dataset_dir / f"{TASKS_FILENAME}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        with open(tmp_path, "wb") as f:
            f.write(encode_dataset_tasks(dataset, tasks))
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return path


def read_tasks_dat(dataset_dir: str | Path) -> tuple[str, dict[int, TaskData]]:
    with open(Path(dataset_dir) / TASKS_FILENAME, "rb") as f:
        return decode_dataset_tasks(f.read())


def read_task_dir(task_dir: str | Path, generate_labels: bool = False) -> TaskData:
    """One task from its per-task files: ``metadata.json`` plus ``labels.dat`` or the legacy
    zipped-CSV pair (see :func:`~tabarena.simulation.label_files.read_task_labels`).
    """
    task_dir = Path(task_dir)
    with open(task_dir / METADATA_FILENAME) as f:
        metadata = json.load(f)
    labels_val, labels_test = read_task_labels(task_dir, generate=generate_labels)
    return TaskData(
        metadata={key: metadata[key] for key in _METADATA_KEYS}, labels_val=labels_val, labels_test=labels_test
    )


def load_dataset_tasks(dataset_dir: str | Path, folds: list[int] | None, generate: bool = True) -> dict[int, TaskData]:
    """All requested folds of one dataset directory.

    ``tasks.dat`` is read when present; folds it does not contain (and every fold when it is
    missing) come from the per-task files. ``folds=None`` means whatever ``tasks.dat`` holds, or
    the fold directories present on disk. With ``generate``, a ``tasks.dat`` that was missing or
    incomplete is (re)written once every requested fold is available, so the next load opens one
    file; a directory that cannot be written is left as is.
    """
    dataset_dir = Path(dataset_dir)
    tasks: dict[int, TaskData] = {}
    complete_on_disk = False
    try:
        _, tasks = read_tasks_dat(dataset_dir)
        complete_on_disk = True
    except FileNotFoundError:
        pass
    if folds is None:
        folds = sorted(tasks) if tasks else sorted(int(p.name) for p in dataset_dir.iterdir() if p.name.isdigit())
    missing = [fold for fold in folds if fold not in tasks]
    for fold in missing:
        tasks[fold] = read_task_dir(dataset_dir / str(fold))
        complete_on_disk = False
    if generate and missing and not complete_on_disk:
        try:
            write_tasks_dat(dataset_dir, dataset_dir.name, {fold: tasks[fold] for fold in folds})
        except OSError as e:
            logger.debug("could not write %s in %s: %s", TASKS_FILENAME, dataset_dir, e)
    return {fold: tasks[fold] for fold in folds}


def load_task_data(
    entries: dict[Path, list[int] | None],
    *,
    n_threads: int = 1,
    generate: bool = True,
    label_cache: LabelFileCache | None = None,
) -> dict[tuple[str, int], TaskData]:
    """``load_dataset_tasks`` over ``entries`` (dataset directory -> folds), on ``n_threads``
    threads, keyed by ``(dataset, fold)``. Labels go through ``label_cache`` when given, so
    artifacts loaded together share one array per task (see ``LabelFileCache.share``).
    """

    def one(item):
        dataset_dir, folds = item
        return dataset_dir.name, load_dataset_tasks(dataset_dir, folds, generate=generate)

    items = list(entries.items())
    if n_threads <= 1 or len(items) <= 1:
        results = [one(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=min(n_threads, len(items))) as executor:
            results = list(executor.map(one, items))
    out: dict[tuple[str, int], TaskData] = {}
    for dataset, tasks in results:
        for fold, task in tasks.items():
            if label_cache is not None:
                task.labels_val, task.labels_test = label_cache.share(
                    dataset=dataset, fold=fold, labels_val=task.labels_val, labels_test=task.labels_test
                )
            out[(dataset, fold)] = task
    return out


def consolidate_task_data(model_predictions_dir: str | Path, remove_task_files: bool = True) -> int:
    """Fold every dataset directory's per-task ``metadata.json`` + labels into ``tasks.dat``.

    Called when an artifact is finalized. With ``remove_task_files`` the per-task
    ``metadata.json`` and ``labels.dat`` are deleted afterwards (the prediction memmaps stay).
    Returns the number of datasets consolidated.
    """
    root = Path(model_predictions_dir)
    n = 0
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        fold_dirs = sorted(
            (p for p in dataset_dir.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)
        )
        tasks = {int(p.name): read_task_dir(p) for p in fold_dirs if (p / METADATA_FILENAME).exists()}
        if not tasks:
            continue
        write_tasks_dat(dataset_dir, dataset_dir.name, tasks)
        n += 1
        if remove_task_files:
            for fold in tasks:
                (dataset_dir / str(fold) / METADATA_FILENAME).unlink(missing_ok=True)
                (dataset_dir / str(fold) / LABELS_FILENAME).unlink(missing_ok=True)
    return n
