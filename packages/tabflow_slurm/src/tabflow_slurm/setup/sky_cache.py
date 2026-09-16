"""Seed a bucket with the datasets a run needs, so SkyPilot workers never fetch from OpenML or the Hub.

The head node materializes every task of a run into its own caches during ``setup`` (OpenML tasks
into the OpenML cache, data-foundry tasks into ``tabarena_tasks/`` plus their text-embedding
caches). This module copies exactly those files into a static, shared prefix in the run's bucket,
laid out like ``CacheConfig.from_root``::

    <dataset_cache_uri>/openml/org/openml/www/tasks/<task id>/        task.xml, datasplits.*
    <dataset_cache_uri>/openml/org/openml/www/datasets/<dataset id>/  description.xml, features.xml, dataset_<id>.pq, ...
    <dataset_cache_uri>/openml/tabarena_tasks/<slug>.pkl              a portable data-foundry task
    <dataset_cache_uri>/tabarena/text_cache/<embedding>/<slug>_cache.parquet

A ``cache_manifest.json`` in the launch's queue maps each dataset to its entries. The worker pulls
them into its ``CACHE_ROOT`` (the same layout) before an item runs, so the runner's
``--materialize_tasks`` finds everything cached and downloads nothing. Datasets are immutable, so
the prefix is shared across users and runs: seeding uploads only what is missing and then verifies
that every local file is present remotely. Workers only read the prefix. A curated bucket that this
account cannot write to is used the same way with ``upload=False``: the setup then only verifies.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from tabarena.benchmark.task.metadata.schema import tid_from_task_id_str

if TYPE_CHECKING:
    from tabarena.benchmark.task.metadata import TaskMetadataCollection
    from tabflow_slurm.setup.sky_storage import Storage

_USER_TASK_PREFIX = "UserTask|"
_PORTABLE_USER_TASK_MARKER = b"tabarena-user-task-v1"
_OPENML_WWW = Path("org/openml/www")


@dataclass(frozen=True)
class CacheEntry:
    """One file or directory of a task, as it lives locally and where it goes in the cache prefix."""

    rel: str
    """Path relative to the ``CacheConfig.from_root`` layout (``openml/...`` or ``tabarena/...``)."""
    local: Path
    is_dir: bool


def local_cache_roots() -> tuple[Path, Path]:
    """The head node's ``(openml root, tabarena root)`` as configured in this process."""
    import openml

    from tabarena.loaders import get_tabarena_cache_root

    return Path(openml.config._root_cache_directory).expanduser(), Path(get_tabarena_cache_root()).expanduser()


def dataset_id_from_task_xml(path: Path) -> int:
    """The ``data_set_id`` an OpenML ``task.xml`` names."""
    match = re.search(r"<oml:data_set_id>\s*(\d+)\s*</oml:data_set_id>", path.read_text())
    if match is None:
        raise ValueError(f"{path} names no data_set_id.")
    return int(match.group(1))


def openml_task_entries(tid: int, *, openml_root: Path) -> list[CacheEntry]:
    """The task and dataset directories of OpenML task ``tid`` in the local cache (empty when not cached)."""
    task_dir = openml_root / _OPENML_WWW / "tasks" / str(tid)
    if not (task_dir / "task.xml").exists():
        return []
    entries = [CacheEntry(rel=f"openml/{_OPENML_WWW}/tasks/{tid}", local=task_dir, is_dir=True)]
    dataset_dir = openml_root / _OPENML_WWW / "datasets" / str(dataset_id_from_task_xml(task_dir / "task.xml"))
    if dataset_dir.is_dir():
        entries.append(
            CacheEntry(rel=f"openml/{_OPENML_WWW}/datasets/{dataset_dir.name}", local=dataset_dir, is_dir=True)
        )
    return entries


def is_portable_user_task(path: Path) -> bool:
    """Whether a ``tabarena_tasks/<slug>.pkl`` is the self-contained format (a legacy pickle names head-node paths)."""
    with path.open("rb") as fh:
        return _PORTABLE_USER_TASK_MARKER in fh.read(4096)


def user_task_entries(task_id_str: str, *, openml_root: Path, tabarena_root: Path) -> list[CacheEntry]:
    """The task pickle and text cache of a data-foundry task in the local caches (empty when not cached).

    A legacy pickle (one that points at ``local/datasets/`` of this machine) is skipped with a
    warning; the worker then materializes that dataset from its source as it would without a cache.
    """
    from tabarena.benchmark.preprocessing.text_cache import embedding_id
    from tabarena.benchmark.task.user_task import UserTask

    slug = UserTask.from_task_id_str(task_id_str).slug
    task_path = openml_root / "tabarena_tasks" / f"{slug}.pkl"
    if not task_path.exists():
        return []
    if not is_portable_user_task(task_path):
        warnings.warn(
            f"{task_path} is a legacy task pickle bound to this machine's cache; not seeded. Re-materialize it to "
            "get the portable format, or let the workers download the dataset.",
            stacklevel=2,
        )
        return []
    entries = [CacheEntry(rel=f"openml/tabarena_tasks/{slug}.pkl", local=task_path, is_dir=False)]
    text_cache = tabarena_root / "text_cache" / embedding_id() / f"{slug}_cache.parquet"
    if text_cache.exists():
        entries.append(
            CacheEntry(rel=f"tabarena/text_cache/{embedding_id()}/{text_cache.name}", local=text_cache, is_dir=False)
        )
    return entries


def collect_cache_entries(
    collection: TaskMetadataCollection, *, openml_root: Path, tabarena_root: Path
) -> dict[str, list[CacheEntry]]:
    """Per dataset (``tabarena_task_name``), the cache entries the head node holds for it."""
    by_dataset: dict[str, list[CacheEntry]] = {}
    for ttm in collection:
        dataset = ttm.tabarena_task_name
        if dataset in by_dataset or ttm.task_id_str is None:
            continue
        task_id_str = str(ttm.task_id_str)
        if task_id_str.startswith(_USER_TASK_PREFIX):
            entries = user_task_entries(task_id_str, openml_root=openml_root, tabarena_root=tabarena_root)
        else:
            entries = openml_task_entries(tid_from_task_id_str(task_id_str), openml_root=openml_root)
        by_dataset[dataset] = entries
    return by_dataset


@dataclass
class CacheSeedReport:
    """What seeding did and found."""

    cache_uri: str
    datasets: int = 0
    files: int = 0
    total_bytes: int = 0
    uploaded_files: int = 0
    not_cached_locally: list[str] = field(default_factory=list)
    """Datasets with nothing to seed (not materialized here, or a legacy pickle); their workers download."""
    unverified: list[str] = field(default_factory=list)
    """Entries still missing remotely after seeding (read-only prefix, or a failed upload)."""

    def summary(self) -> str:
        parts = [
            f"dataset cache {self.cache_uri}: {self.datasets} dataset(s), {self.files} file(s), "
            f"{self.total_bytes / 1e6:.0f} MB, {self.uploaded_files} file(s) uploaded now"
        ]
        if self.not_cached_locally:
            parts.append(f"not cached on this node (workers download them): {self.not_cached_locally}")
        if self.unverified:
            parts.append(f"MISSING remotely (workers download them): {self.unverified}")
        return "\n".join(parts)


def seed_dataset_cache(
    entries_by_dataset: dict[str, list[CacheEntry]],
    *,
    storage: Storage,
    cache_uri: str,
    upload: bool = True,
) -> tuple[dict, CacheSeedReport]:
    """Upload what is missing under ``cache_uri`` (when ``upload``), verify, and build the worker manifest.

    Returns ``(manifest, report)``. The manifest lists, per dataset, the entries that are verified
    present remotely: ``{"cache_uri": ..., "datasets": {"<dataset>": ["openml/...", ...]}}``. An entry
    that could not be verified is left out, so the worker falls back to downloading that dataset from
    its source instead of failing on a half-seeded cache.
    """
    cache_uri = cache_uri.rstrip("/")
    report = CacheSeedReport(cache_uri=cache_uri)
    manifest: dict = {"cache_uri": cache_uri, "datasets": {}}
    for dataset, entries in entries_by_dataset.items():
        if not entries:
            report.not_cached_locally.append(dataset)
            continue
        report.datasets += 1
        verified: list[str] = []
        for entry in entries:
            remote = f"{cache_uri}/{entry.rel}"
            if entry.is_dir:
                files = sorted(p for p in entry.local.rglob("*") if p.is_file())
                names = {p.relative_to(entry.local).as_posix() for p in files}
                present = set(_remote_files(storage, remote))
                missing = names - present
                if missing and upload:
                    storage.upload_dir(entry.local, remote)
                    report.uploaded_files += len(missing)
                    present = set(_remote_files(storage, remote))
                report.files += len(files)
                report.total_bytes += sum(p.stat().st_size for p in files)
                ok = names <= present
            else:
                exists = storage.exists(remote)
                if not exists and upload:
                    storage.upload_file(entry.local, remote)
                    report.uploaded_files += 1
                    exists = storage.exists(remote)
                report.files += 1
                report.total_bytes += entry.local.stat().st_size
                ok = exists
            if ok:
                verified.append(entry.rel)
            else:
                report.unverified.append(entry.rel)
        if verified:
            manifest["datasets"][dataset] = verified
    return manifest, report


def _remote_files(storage: Storage, uri: str) -> list[str]:
    """Relative paths of the objects under ``uri`` (OpenML task and dataset directories are flat)."""
    return [name for name in storage.list_names(uri) if not name.endswith("/")]
