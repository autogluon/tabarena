"""Preset metadata bundles for the Data Foundry ``BeyondArena`` collection.

Thin :class:`~tabarena.benchmark.task.metadata.bundles.base.TabArenaMetadataBundle`
subclasses that default the source to the ``"BeyondArena"`` suite literal (resolved
to a :class:`~tabarena.benchmark.task.metadata.sources.data_foundry.DataFoundryTaskMetadataSource`
over the official collection).

All the loading / download behavior lives on the (general-purpose) source, so:

* the same filters as any bundle apply (``problem_types_to_run``,
  ``dataset_names_to_run``, dtype / size filters), and ``materialize=False`` skips
  all downloads (inspect / filter only);
* passing ``task_metadata=<DataFrame | list | csv path>`` makes the bundle use your
  own metadata instead (it is *not* special-cased here — the base resolves it);
* to run a *different* collection, use the source directly::

      TabArenaMetadataBundle(task_metadata=DataFoundryTaskMetadataSource(my_collection))

Requires the optional ``data-foundry`` dependency (``tabarena[data-foundry]``).
"""

from __future__ import annotations

from dataclasses import dataclass

from tabarena.benchmark.task.metadata.bundles.base import TabArenaMetadataBundle


@dataclass
class BeyondArenaMetadataBundle(TabArenaMetadataBundle):
    """Tasks sourced from the official Data Foundry ``BeyondArena`` collection."""

    task_metadata: str = "BeyondArena"


@dataclass
class BeyondArenaLiteMetadataBundle(BeyondArenaMetadataBundle):
    """BeyondArena Lite: the first split (``r0f0``) of each dataset."""

    split_indices_to_run: str = "lite"
