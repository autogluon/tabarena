"""Task-selection bundles: the base class plus shipped benchmark-suite presets.

A bundle defines *which* tasks (datasets x splits) to run and how to filter them.
``base`` holds the generic :class:`TabArenaMetadataBundle`; the other modules are
thin presets pinned to a specific benchmark suite.
"""

from __future__ import annotations

from tabarena.benchmark.task.metadata.bundles.base import TabArenaMetadataBundle
from tabarena.benchmark.task.metadata.bundles.beyond_arena import (
    BeyondArenaLiteMetadataBundle,
    BeyondArenaMetadataBundle,
)
from tabarena.benchmark.task.metadata.bundles.tabarena_v0pt1 import (
    TabArenaV0pt1LiteMetadataBundle,
    TabArenaV0pt1MetadataBundle,
)

__all__ = [
    "BeyondArenaLiteMetadataBundle",
    "BeyondArenaMetadataBundle",
    "TabArenaMetadataBundle",
    "TabArenaV0pt1LiteMetadataBundle",
    "TabArenaV0pt1MetadataBundle",
]
