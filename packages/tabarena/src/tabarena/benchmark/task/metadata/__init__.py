"""TabArena task metadata: schema (data classes) and bundle (selection/filtering)."""

from __future__ import annotations

from tabarena.benchmark.task.metadata.bundles import (
    BeyondArenaLiteMetadataBundle,
    BeyondArenaMetadataBundle,
    TabArenaMetadataBundle,
    TabArenaV0pt1LiteMetadataBundle,
    TabArenaV0pt1MetadataBundle,
)
from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection
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
    return TabArenaV0pt1MetadataBundle(materialize=False).load_collection()


__all__ = [
    "BeyondArenaLiteMetadataBundle",
    "BeyondArenaMetadataBundle",
    "DataFoundryTaskMetadataSource",
    "GroupLabelTypes",
    "InMemoryTaskMetadataSource",
    "OpenMLTaskMetadataSource",
    "SplitIndex",
    "SplitMetadata",
    "SplitTimeHorizonTypes",
    "SplitTimeHorizonUnitTypes",
    "TabArenaMetadataBundle",
    "TabArenaTaskMetadata",
    "TabArenaV0pt1LiteMetadataBundle",
    "TabArenaV0pt1MetadataBundle",
    "TabArenaV0pt1TaskMetadataSource",
    "TaskMetadataCollection",
    "TaskMetadataSource",
    "ValidationMetadata",
    "default_task_metadata_collection",
    "resolve_source",
    "to_legacy_task_metadata",
]
