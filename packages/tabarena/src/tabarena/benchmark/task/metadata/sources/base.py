"""The :class:`TaskMetadataSource` strategy and the in-memory source.

A *source* answers two questions for a bundle, decoupled from any filtering:

* :meth:`TaskMetadataSource.load` — *where does the task metadata come from?*
  (a built-in suite, a DataFrame/CSV the user already has, a Data Foundry
  collection, ...). Returns ``list[TabArenaTaskMetadata]`` without applying any
  selection filters.
* :meth:`TaskMetadataSource.materialize` — *how do we make the (already-filtered)
  tasks runnable?* The default is a no-op; sources backed by a remote collection
  override it to download + convert only the tasks that survived filtering.

This keeps :class:`~tabarena.benchmark.task.metadata.bundles.base.TabArenaMetadataBundle`
free of any per-suite special-casing: the bundle owns filtering, the source owns
loading + materialization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from tabarena.benchmark.task.metadata.schema import TabArenaTaskMetadata


def committed_metadata_dir() -> Path:
    """Directory holding the git-committed reference-metadata CSVs (one per source).

    All sources (built-in suites and Data Foundry collections) share this single
    location: ``benchmark/task/metadata/sources/data/``.
    """
    return Path(__file__).parent / "data"


def committed_metadata_path(name: str) -> Path:
    """Path to the committed ``<name>_tasks_metadata.csv`` reference file.

    ``name`` is the suite literal / collection name, e.g. ``"TabArena-v0.1"`` or
    ``"BeyondArena"``.
    """
    return committed_metadata_dir() / f"{name}_tasks_metadata.csv"


class TaskMetadataSource(ABC):
    """Produces (and optionally materializes) task metadata for a bundle."""

    @abstractmethod
    def load(self) -> list[TabArenaTaskMetadata]:
        """Return the (unfiltered) task metadata this source describes.

        Implementations must not download dataset contents here when it can be
        avoided — loading should be cheap so bundles can filter before any
        expensive materialization (see :meth:`materialize`).
        """

    def materialize(self, task_metadata: list[TabArenaTaskMetadata]) -> None:
        """Ensure the given (already-filtered) tasks are runnable, in place.

        Called by the bundle after filtering, only when the bundle's
        ``materialize`` flag is set. The default does nothing — metadata that is
        already local (in-memory frames, built-in suites) needs no preparation.
        Sources backed by a remote collection override this to fetch + convert
        the tasks and update each ``task_id_str`` accordingly.
        """
        return


class InMemoryTaskMetadataSource(TaskMetadataSource):
    """Wraps task metadata the caller already has: a list, DataFrame, or CSV path.

    A DataFrame (or CSV loaded into one) is parsed row-by-row via
    :meth:`TabArenaTaskMetadata.from_row`. A list is passed through as-is. There
    is nothing to materialize — see :meth:`TaskMetadataSource.materialize`.
    """

    def __init__(self, data: pd.DataFrame | list[TabArenaTaskMetadata] | str | Path) -> None:
        """Store the already-available metadata (list, DataFrame, or CSV path)."""
        self.data = data

    def load(self) -> list[TabArenaTaskMetadata]:
        """Parse the wrapped data into a list of TabArenaTaskMetadata."""
        data = self.data
        if isinstance(data, (str, Path)):
            print(f"Loading task metadata from {data}...")
            data = pd.read_csv(data, index_col=False)
        if isinstance(data, pd.DataFrame):
            data = [TabArenaTaskMetadata.from_row(row) for _, row in data.iterrows()]
        data = list(data)
        assert all(isinstance(x, TabArenaTaskMetadata) for x in data)
        return data
