"""Task metadata sources + the single ``resolve_source`` entry point.

A :class:`TaskMetadataSource` knows how to load (and optionally materialize) task
metadata. :func:`resolve_source` maps whatever a bundle was configured with — a
source instance, a registered suite literal, or already-available metadata
(DataFrame / CSV path / list) — onto a concrete source. This is the *one* place
that knows about all built-in loaders, so bundles never special-case a suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.benchmark.task.metadata.sources.base import (
    InMemoryTaskMetadataSource,
    TaskMetadataSource,
)
from tabarena.benchmark.task.metadata.sources.data_foundry import DataFoundryTaskMetadataSource
from tabarena.benchmark.task.metadata.sources.openml import OpenMLTaskMetadataSource
from tabarena.benchmark.task.metadata.sources.tabarena_v0pt1 import (
    TABARENA_V0PT1_NAME,
    TabArenaV0pt1TaskMetadataSource,
    load_tabarena_v0_1_task_metadata,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pandas as pd

    from tabarena.benchmark.task.metadata.schema import TabArenaTaskMetadata


def _beyond_arena_source() -> TaskMetadataSource:
    """Build the default source for the official ``BeyondArena`` collection.

    The collection is passed as a lazy factory (+ explicit ``name``) so merely *loading* the
    reference metadata — e.g. constructing a ``BeyondArenaContext`` for offline leaderboard
    compare — does not import the optional ``data-foundry`` dependency; the collection is built
    only when a download is needed (CSV regeneration / materialization).
    """
    from tabarena.benchmark.task.data_foundry import get_beyond_arena_collection

    return DataFoundryTaskMetadataSource(collection_factory=get_beyond_arena_collection, name="BeyondArena")


# Registered suite literals -> factory. Factories are lazy so optional deps
# (e.g. data-foundry for "BeyondArena") are only imported when actually used.
_SOURCE_REGISTRY: dict[str, Callable[[], TaskMetadataSource]] = {
    TABARENA_V0PT1_NAME: TabArenaV0pt1TaskMetadataSource,  # "TabArena-v0.1"
    "BeyondArena": _beyond_arena_source,
}


def resolve_source(
    task_metadata: TaskMetadataSource | pd.DataFrame | list[TabArenaTaskMetadata] | str | Path,
) -> TaskMetadataSource:
    """Resolve a bundle's ``task_metadata`` into a :class:`TaskMetadataSource`.

    Resolution order:

    1. A :class:`TaskMetadataSource` instance is returned unchanged.
    2. A string matching a registered suite literal (e.g. ``"TabArena-v0.1"``,
       ``"BeyondArena"``) builds that suite's source.
    3. Anything else — a DataFrame, a ``list[TabArenaTaskMetadata]``, or a str/Path
       to a CSV file — is wrapped in an :class:`InMemoryTaskMetadataSource`.
    """
    if isinstance(task_metadata, TaskMetadataSource):
        return task_metadata
    if isinstance(task_metadata, str) and task_metadata in _SOURCE_REGISTRY:
        return _SOURCE_REGISTRY[task_metadata]()
    return InMemoryTaskMetadataSource(task_metadata)


__all__ = [
    "DataFoundryTaskMetadataSource",
    "InMemoryTaskMetadataSource",
    "OpenMLTaskMetadataSource",
    "TabArenaV0pt1TaskMetadataSource",
    "TaskMetadataSource",
    "load_tabarena_v0_1_task_metadata",
    "resolve_source",
]
