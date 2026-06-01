"""Preset metadata bundles for the TabArena v0.1 benchmark.

Thin :class:`~tabarena.benchmark.task.metadata.bundles.base.TabArenaMetadataBundle`
subclasses that default the source to the ``"TabArena-v0.1"`` suite literal (resolved
to :class:`~tabarena.benchmark.task.metadata.sources.tabarena_v0pt1.TabArenaV0pt1TaskMetadataSource`).
"""

from __future__ import annotations

from dataclasses import dataclass

from tabarena.benchmark.task.metadata.bundles.base import TabArenaMetadataBundle


@dataclass
class TabArenaV0pt1MetadataBundle(TabArenaMetadataBundle):
    """Metadata for full TabArena v0.1 benchmark: 51 datasets, 816 tasks."""

    task_metadata: str = "TabArena-v0.1"


@dataclass
class TabArenaV0pt1LiteMetadataBundle(TabArenaV0pt1MetadataBundle):
    """TabArena v0.1 Lite (first split of each dataset): 51 datasets, 51 tasks."""

    split_indices_to_run: str = "lite"
