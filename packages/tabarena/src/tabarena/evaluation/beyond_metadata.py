"""Load BeyondArena (data-foundry) task metadata for evaluation, self-contained.

The committed reference CSV shipped with the metadata source
(``benchmark/task/metadata/sources/data/<name>_tasks_metadata.csv``) already carries the
warehouse-level fields inline (``task_type``, ``num_text_cols``, ``num_high_cardinality_cats``,
``num_cols_after_preprocessing``, ``missing_value_fraction``, ``domain``, ``dataset_year``,
``source``), so the loaded :class:`~tabarena.benchmark.task.metadata.TaskMetadataCollection` is
self-contained: its :meth:`~tabarena.benchmark.task.metadata.TaskMetadataCollection.per_dataset_frame`
carries every column the BeyondArena subset predicates need (incl. ``max_train_rows``) — no
warehouse merge needed.
"""

from __future__ import annotations

from pathlib import Path

from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.benchmark.task.metadata.sources.base import (
    InMemoryTaskMetadataSource,
    committed_metadata_path,
)


def load_beyond_task_metadata_collection(source: str | Path = "BeyondArena") -> TaskMetadataCollection:
    """Load the BeyondArena task metadata as a native :class:`TaskMetadataCollection`.

    Args:
        source: Either a registered suite/collection name whose committed CSV ships as package
            data (e.g. ``"BeyondArena"``), or a path to a ``*_tasks_metadata.csv`` reference file.
    """
    data = committed_metadata_path(str(source)) if _looks_like_suite_name(str(source)) else source
    return TaskMetadataCollection(InMemoryTaskMetadataSource(data).load())


def _looks_like_suite_name(source: str) -> bool:
    """True if ``source`` is a bare suite/collection name rather than a CSV path."""
    return not (source.endswith(".csv") or "/" in source or "\\" in source)
