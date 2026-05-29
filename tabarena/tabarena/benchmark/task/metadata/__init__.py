"""TabArena task metadata: schema (data classes) and bundle (selection/filtering)."""

from __future__ import annotations

from tabarena.benchmark.task.metadata.bundle import TabArenaMetadataBundle
from tabarena.benchmark.task.metadata.schema import (
    GroupLabelTypes,
    SplitIndex,
    SplitMetadata,
    SplitTimeHorizonTypes,
    SplitTimeHorizonUnitTypes,
    TabArenaTaskMetadata,
)

__all__ = [
    "GroupLabelTypes",
    "SplitIndex",
    "SplitMetadata",
    "SplitTimeHorizonTypes",
    "SplitTimeHorizonUnitTypes",
    "TabArenaMetadataBundle",
    "TabArenaTaskMetadata",
]
