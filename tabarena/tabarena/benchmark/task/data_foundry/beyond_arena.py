"""End-to-end glue for running a Data Foundry collection (e.g. ``BeyondArena``).

This module turns "the user picked a collection + some filters" into "runnable
TabArena tasks", while downloading as little as possible:

* :func:`load_reference_metadata` returns a task-metadata DataFrame *without*
  downloading any dataset. It prefers a git-committed reference CSV (shipped as
  package data) so a user can filter by name / dtype / size before fetching
  anything. If no committed CSV exists it regenerates one by downloading the
  whole collection (and caches the result).
* :func:`materialize_task` ensures a *single* task's local OpenML pickle exists,
  downloading + converting its container only when it is not already on disk.
  This is what makes "download only the datasets that survived the filter" work.

Task ids are portable by construction: :attr:`UserTask.task_id` is a deterministic
hash of the (collection-unique) task name, and standardized tasks serialize their
``task_id_str`` *without* a cache path (see :meth:`UserTask.task_id_str`), so the
same reference CSV resolves correctly on any machine via the ambient OpenML cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tabarena.benchmark.task.data_foundry.adapter import (
    DEFAULT_EVAL_METRICS,
    DataFoundryAdapter,
    convert_curated_container_to_user_task,
)
from tabarena.benchmark.task.data_foundry.metadata_cache import get_path_to_metadata_cache

if TYPE_CHECKING:
    from data_foundry.collections import DatasetCollection


def get_beyond_arena_collection() -> DatasetCollection:
    """Return the official ``BeyondArena`` Data Foundry collection.

    Imported lazily so ``tabarena`` works without the optional ``data-foundry``
    dependency installed.
    """
    from data_foundry.collections import BEYOND_ARENA

    return BEYOND_ARENA


def reference_metadata_package_path(collection_name: str) -> Path:
    """Path to the git-committed reference-metadata CSV for ``collection_name``.

    Shipped as package data in the unified sources data directory
    (``benchmark/task/metadata/sources/data/``, shared with the built-in suites).
    The file may not exist — callers fall back to regeneration when it is missing.
    """
    from tabarena.benchmark.task.metadata.sources.base import committed_metadata_path

    return committed_metadata_path(collection_name)


def _generated_metadata_cache_path(collection_name: str) -> Path:
    """Path of the regenerated metadata CSV inside the OpenML metadata cache."""
    return get_path_to_metadata_cache() / f"{collection_name}_tasks_metadata.csv"


def load_reference_metadata(
    *,
    collection: DatasetCollection | None = None,
    cache_dir: str | None = None,
    force_regenerate: bool = False,
) -> pd.DataFrame:
    """Return the reference task-metadata DataFrame for ``collection``.

    Resolution order (unless ``force_regenerate``):

    1. The git-committed CSV (package data) — no dataset downloads.
    2. A previously regenerated CSV in the OpenML metadata cache.
    3. Regenerate by downloading + converting the whole collection, caching the
       result for next time.

    Args:
        collection: Collection to describe. Defaults to ``BeyondArena``.
        cache_dir: Optional data_foundry cache override used only when regenerating.
        force_regenerate: Skip both caches and rebuild from the source collection.

    Returns:
        One row per ``(task, split)`` with all :class:`TabArenaTaskMetadata`
        columns plus ``data_foundry_uri``. ``task_id_str`` uses the portable
        sentinel cache path (see :func:`localize_task_id_str`).
    """
    if collection is None:
        collection = get_beyond_arena_collection()

    if not force_regenerate:
        committed = reference_metadata_package_path(collection.name)
        if committed.exists():
            print(f"Loading committed {collection.name} reference metadata from {committed}.")
            return pd.read_csv(committed)

        cached = _generated_metadata_cache_path(collection.name)
        if cached.exists():
            print(f"Loading regenerated {collection.name} reference metadata from {cached}.")
            return pd.read_csv(cached)

    out_path = generate_reference_metadata(
        collection=collection,
        out_path=_generated_metadata_cache_path(collection.name),
        cache_dir=cache_dir,
        force_download=force_regenerate,
    )
    return pd.read_csv(out_path)


def generate_reference_metadata(
    *,
    collection: DatasetCollection | None = None,
    out_path: str | Path | None = None,
    cache_dir: str | None = None,
    force_download: bool = False,
    evaluation_metrics: dict[str, list[str]] | None = None,
) -> Path:
    """Build the reference-metadata CSV for a collection (maintainer tool).

    Downloads + converts *every* container in the collection (creating each local
    OpenML task pickle as a side effect), then writes the metadata CSV. The CSV is
    machine-independent and clean to commit: standardized ``UserTask`` ids carry no
    cache path (see :meth:`UserTask.task_id_str`).

    Commit the returned file to ``data_foundry/data/<collection.name>_tasks_metadata.csv``
    to enable filter-before-download for everyone.

    Args:
        collection: Collection to materialize. Defaults to ``BeyondArena``.
        out_path: Where to write the CSV. Defaults to the OpenML metadata cache.
        cache_dir: Optional data_foundry cache override (where containers download).
        force_download: Re-fetch every container from its source.
        evaluation_metrics: Override the allowed eval metrics per problem type.

    Returns:
        The path the CSV was written to.
    """
    if collection is None:
        collection = get_beyond_arena_collection()

    adapter = DataFoundryAdapter(collection=collection, evaluation_metrics=evaluation_metrics)
    metadata_df = adapter.to_tabarena_user_tasks(cache_dir=cache_dir, force_download=force_download)

    out_path = Path(out_path) if out_path is not None else _generated_metadata_cache_path(collection.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(metadata_df)} reference-metadata rows to {out_path}.")
    metadata_df.to_csv(out_path, index=False)
    return out_path


def materialize_task(
    *,
    collection: DatasetCollection,
    task_id_str: str,
    data_foundry_uri: str,
    evaluation_metrics: dict[str, list[str]] | None = None,
    cache_dir: str | None = None,
    force_download: bool = False,
) -> str:
    """Ensure one task's local OpenML pickle exists; return its local ``task_id_str``.

    If the pickle already exists on disk (matched by the stable task name) and
    ``force_download`` is ``False``, this is a no-op that just returns the id —
    *no* dataset is downloaded. Otherwise the container is fetched via the
    collection's source (cached by data_foundry) and converted into a local
    OpenML task.

    Args:
        collection: The source collection (used to resolve + download the entry).
        task_id_str: The ``UserTask`` id from the reference metadata. Only its
            task-name segment is used; the pickle is resolved against the ambient
            OpenML cache.
        data_foundry_uri: The collection-entry relative path identifying the
            container to download (``<unique_name>/[versions/]<uuid>``).
        evaluation_metrics: Allowed eval metrics per problem type. Defaults to
            :data:`DEFAULT_EVAL_METRICS`.
        cache_dir: Optional data_foundry cache override.
        force_download: Re-fetch + reconvert even if the pickle already exists.

    Returns:
        The ``task_id_str`` whose resolved pickle path now exists on disk.
    """
    from tabarena.benchmark.task.user_task import UserTask

    if not data_foundry_uri or pd.isna(data_foundry_uri):
        raise ValueError(
            f"Cannot materialize task {task_id_str!r}: missing `data_foundry_uri`. "
            "Reference metadata for a Data Foundry collection must carry it.",
        )

    task = UserTask.from_task_id_str(task_id_str)
    if (not force_download) and task.openml_task_path.exists():
        return task.task_id_str

    if evaluation_metrics is None:
        evaluation_metrics = DEFAULT_EVAL_METRICS

    uuid = Path(data_foundry_uri).name
    container = collection.get_dataset(uuid, cache_dir=cache_dir, force_download=force_download)
    user_task = convert_curated_container_to_user_task(
        container=container,
        evaluation_metrics=evaluation_metrics,
    )
    return user_task.task_id_str
