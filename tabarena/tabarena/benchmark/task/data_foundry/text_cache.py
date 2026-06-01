"""Import the semantic-text embedding cache that ships *inside* a Data Foundry container.

The BeyondArena HF dataset (e.g. ``TabArena/BeyondArena``) ships, inside each dataset container
directory (``<dataset>/<uuid>/``), a ``tabarena_text_cache.parquet`` *extra artifact* alongside the
dataset files. data_foundry downloads the whole container folder in one ``snapshot_download``, so the
text cache arrives on disk together with the dataset — there is no separate text-cache download.

This module copies that bundled parquet into the canonical, slug-keyed tabarena cache
(:func:`~tabarena.benchmark.preprocessing.text_cache.text_cache_path`) so the fit-time loader finds
it like any other cache. The copy is triggered when a container is converted to a TabArena task
(materialization); :func:`ensure_text_cache_for_task` re-derives it from the already-downloaded
container for tasks materialized before the cache was imported.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from data_foundry.collections import DatasetCollection

#: Prefix of the text-cache extra artifact shipped inside each Data Foundry container. The encoder
#: ``<embedding-id>`` is appended (the container can't use a subfolder — extra artifacts are bare
#: files and the snapshot download glob is non-recursive), so the filename carries the same
#: encoder-versioning the tabarena local cache encodes as a subfolder.
CONTAINER_TEXT_CACHE_PREFIX = "tabarena_text_cache"


def container_text_cache_filename(embedding_id: str | None = None) -> str:
    """Filename of the text-cache artifact in a container for the given (or current) encoder."""
    if embedding_id is None:
        from tabarena.benchmark.preprocessing.text_cache import embedding_id as current_embedding_id

        embedding_id = current_embedding_id()
    return f"{CONTAINER_TEXT_CACHE_PREFIX}_{embedding_id}.parquet"


def import_text_cache_from_container(container, task_key: str) -> Path | None:
    """Copy a container's bundled text cache into the canonical tabarena location.

    ``container`` is a (downloaded) ``data_foundry.curation_container.CuratedContainer`` — its
    ``loaded_from_path`` points at the on-disk container dir. Looks for the encoder-versioned
    artifact (``tabarena_text_cache_<embedding-id>.parquet``). Returns the destination path, or
    ``None`` if the container ships no cache for the current encoder. Existing destinations are kept.
    """
    from tabarena.benchmark.preprocessing.text_cache import text_cache_path

    filename = container_text_cache_filename()
    if not container.has_extra_file(filename):
        logger.debug(f"[text-cache] {task_key}: container ships no '{filename}'; skipping.")
        return None
    dst = text_cache_path(task_key)
    if dst.exists():
        logger.debug(f"[text-cache] {task_key}: already present at {dst}; skipping import.")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(container.extra_file_path(filename), dst)
    logger.info(f"[text-cache] {task_key}: imported embeddings from container -> {dst}")
    return dst


def ensure_text_cache_for_task(
    *,
    collection: DatasetCollection,
    data_foundry_uri: str,
    task_key: str,
    cache_dir: str | None = None,
    force_download: bool = False,
) -> Path | None:
    """Ensure ``task_key``'s text cache is in the canonical location, importing it from the container.

    No-op (returns the existing path) if the cache is already present. Otherwise resolves the task's
    container — using the local Data Foundry download if available, else fetching it from the source
    (the same ``snapshot_download`` that brings the dataset) — and copies its bundled text cache.
    """
    from tabarena.benchmark.preprocessing.text_cache import resolve_existing_cache_path

    if not force_download:
        existing = resolve_existing_cache_path(task_key)
        if existing is not None:
            return existing

    uuid = Path(data_foundry_uri).name
    container = collection.get_dataset(uuid, cache_dir=cache_dir, force_download=force_download)
    return import_text_cache_from_container(container, task_key)
