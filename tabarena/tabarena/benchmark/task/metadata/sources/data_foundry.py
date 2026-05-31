"""Task metadata source backed by *any* Data Foundry collection.

This is general-purpose: ``BeyondArena`` is just one collection. Point it at any
:class:`data_foundry.collections.DatasetCollection` to get the same
"filter the reference metadata, then download only what survived" workflow.

Requires the optional ``data-foundry`` dependency (``tabarena[data-foundry]``);
all data_foundry imports are deferred to call time so importing this module never
forces that dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.benchmark.task.metadata.schema import TabArenaTaskMetadata
from tabarena.benchmark.task.metadata.sources.base import TaskMetadataSource

if TYPE_CHECKING:
    from data_foundry.collections import DatasetCollection


class DataFoundryTaskMetadataSource(TaskMetadataSource):
    """Load + materialize tasks from a Data Foundry collection.

    :meth:`load` returns the collection's *reference* metadata without downloading
    any dataset (preferring a committed CSV, else regenerating once and caching),
    so a bundle can filter first. :meth:`materialize` then downloads + converts
    only the tasks that survived filtering.
    """

    def __init__(
        self,
        collection: DatasetCollection,
        *,
        cache_dir: str | None = None,
        force_download: bool = False,
        force_regenerate: bool = False,
        evaluation_metrics: dict[str, list[str]] | None = None,
    ) -> None:
        """Initialize the source.

        Args:
            collection: The Data Foundry collection to load tasks from.
            cache_dir: Optional override for the data_foundry download cache.
            force_download: Re-fetch + reconvert each container during
                materialization even if its local task pickle already exists.
            force_regenerate: Ignore the committed / cached reference CSV and
                rebuild it by downloading the whole collection.
            evaluation_metrics: Override the allowed eval metrics per problem type
                (see :data:`tabarena.benchmark.task.data_foundry.DEFAULT_EVAL_METRICS`).
        """
        self.collection = collection
        self.cache_dir = cache_dir
        self.force_download = force_download
        self.force_regenerate = force_regenerate
        self.evaluation_metrics = evaluation_metrics

    def load(self) -> list[TabArenaTaskMetadata]:
        """Load the collection's reference metadata (no dataset downloads)."""
        from tabarena.benchmark.task.data_foundry import load_reference_metadata

        metadata_df = load_reference_metadata(
            collection=self.collection,
            cache_dir=self.cache_dir,
            force_regenerate=self.force_regenerate,
        )
        return [TabArenaTaskMetadata.from_row(row) for _, row in metadata_df.iterrows()]

    def materialize(self, task_metadata: list[TabArenaTaskMetadata]) -> None:
        """Download + convert each task that carries a ``data_foundry_uri``, in place.

        Tasks without a ``data_foundry_uri`` are skipped (e.g. custom tasks mixed
        in via a hand-edited frame) — they are assumed to be locally available.
        """
        to_materialize = [
            ttm for ttm in task_metadata if isinstance(ttm.data_foundry_uri, str) and ttm.data_foundry_uri
        ]
        if not to_materialize:
            return

        from tqdm import tqdm

        from tabarena.benchmark.task.data_foundry import materialize_task

        for ttm in tqdm(to_materialize, desc=f"Materializing {self.collection.name} tasks"):
            ttm.task_id_str = materialize_task(
                collection=self.collection,
                task_id_str=ttm.task_id_str,
                data_foundry_uri=ttm.data_foundry_uri,
                evaluation_metrics=self.evaluation_metrics,
                cache_dir=self.cache_dir,
                force_download=self.force_download,
            )
