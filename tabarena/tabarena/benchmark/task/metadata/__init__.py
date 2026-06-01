"""TabArena task metadata: schema (data classes) and bundle (selection/filtering)."""

from __future__ import annotations

from tabarena.benchmark.task.metadata.bundles import (
    BeyondArenaLiteMetadataBundle,
    BeyondArenaMetadataBundle,
    TabArenaMetadataBundle,
    TabArenaV0pt1LiteMetadataBundle,
    TabArenaV0pt1MetadataBundle,
)
from tabarena.benchmark.task.metadata.schema import (
    GroupLabelTypes,
    SplitIndex,
    SplitMetadata,
    SplitTimeHorizonTypes,
    SplitTimeHorizonUnitTypes,
    TabArenaTaskMetadata,
)
from tabarena.benchmark.task.metadata.sources import (
    DataFoundryTaskMetadataSource,
    InMemoryTaskMetadataSource,
    OpenMLTaskMetadataSource,
    TabArenaV0pt1TaskMetadataSource,
    TaskMetadataSource,
    resolve_source,
)

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
    "TaskMetadataSource",
    "resolve_source",
]
