"""TabArena task metadata: schema (data classes), sources, and the TaskMetadataCollection."""

from __future__ import annotations

from tabarena.benchmark.task.metadata.collection import (
    BeyondArenaTaskMetadataCollection,
    TabArenaTaskMetadataCollection,
    TaskMetadataCollection,
    TaskSubset,
)
from tabarena.benchmark.task.metadata.compute import compute_task_metadata
from tabarena.benchmark.task.metadata.schema import (
    GroupLabelTypes,
    SplitIndex,
    SplitMetadata,
    SplitTimeHorizonTypes,
    SplitTimeHorizonUnitTypes,
    TabArenaTaskMetadata,
    ValidationMetadata,
    to_legacy_task_metadata,
)
from tabarena.benchmark.task.metadata.sources import (
    DataFoundryTaskMetadataSource,
    InMemoryTaskMetadataSource,
    OpenMLTaskMetadataSource,
    TabArenaV0pt1TaskMetadataSource,
    TaskMetadataSource,
    resolve_source,
)


def default_task_metadata_collection() -> TaskMetadataCollection:
    """The default native TabArena v0.1 task metadata collection (metadata only, no downloads).

    Matches what ``TabArenaContext(task_metadata="tabarena")`` loads — used as the ``None`` default
    for task metadata across the evaluator / ``compare`` paths.
    """
    return TaskMetadataCollection.from_preset("TabArena-v0.1")


__all__ = [
    "BeyondArenaTaskMetadataCollection",
    "DataFoundryTaskMetadataSource",
    "GroupLabelTypes",
    "InMemoryTaskMetadataSource",
    "OpenMLTaskMetadataSource",
    "SplitIndex",
    "SplitMetadata",
    "SplitTimeHorizonTypes",
    "SplitTimeHorizonUnitTypes",
    "TabArenaTaskMetadata",
    "TabArenaTaskMetadataCollection",
    "TabArenaV0pt1TaskMetadataSource",
    "TaskMetadataCollection",
    "TaskMetadataSource",
    "TaskSubset",
    "ValidationMetadata",
    "compute_task_metadata",
    "default_task_metadata_collection",
    "resolve_source",
    "to_legacy_task_metadata",
]
