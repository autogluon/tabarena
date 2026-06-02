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
        verbose: bool = False,
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
            verbose: Emit per-task ``debug`` logs during materialization (e.g. the
                text-cache "skipping" lines). Off by default to keep the materialize
                progress bar clean.
        """
        self.collection = collection
        self.cache_dir = cache_dir
        self.force_download = force_download
        self.force_regenerate = force_regenerate
        self.evaluation_metrics = evaluation_metrics
        self.verbose = verbose

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

        ``task_metadata`` is typically unrolled to one entry per split, so the same
        dataset (same ``data_foundry_uri``) appears once per split. Materialization
        is a dataset-level operation, so we download/convert each unique dataset only
        once and then propagate the resulting ``task_id_str`` to all of its splits —
        avoiding the long loop of redundant (no-op) per-split calls.
        """
        to_materialize = [
            ttm for ttm in task_metadata if isinstance(ttm.data_foundry_uri, str) and ttm.data_foundry_uri
        ]
        if not to_materialize:
            return

        # Group splits by their unique dataset (data_foundry_uri), preserving order.
        splits_by_uri: dict[str, list[TabArenaTaskMetadata]] = {}
        for ttm in to_materialize:
            splits_by_uri.setdefault(ttm.data_foundry_uri, []).append(ttm)

        from contextlib import suppress

        from tqdm import tqdm

        from tabarena.benchmark.task.data_foundry import materialize_task

        # Silence huggingface_hub's per-file "Fetching N files" bars — the single progress bar below
        # (showing the current dataset) is the clean view; restore HF bars afterwards.
        restore_hf_bars = None
        with suppress(ImportError):
            from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

            disable_progress_bars()
            restore_hf_bars = enable_progress_bars

        print(
            f"Materializing {len(splits_by_uri)} {self.collection.name} dataset(s) "
            f"across {len(to_materialize)} split(s) (datasets + text caches)..."
        )
        bar = tqdm(splits_by_uri.items(), desc=f"Materializing {self.collection.name}", unit="dataset")
        try:
            for data_foundry_uri, splits in bar:
                first = splits[0]
                bar.set_postfix_str(first.tabarena_task_name or first.dataset_name or "")
                task_id_str = materialize_task(
                    collection=self.collection,
                    task_id_str=first.task_id_str,
                    data_foundry_uri=data_foundry_uri,
                    evaluation_metrics=self.evaluation_metrics,
                    cache_dir=self.cache_dir,
                    force_download=self.force_download,
                    verbose=self.verbose,
                )
                # Propagate the resolved local id to every split of this dataset.
                for ttm in splits:
                    ttm.task_id_str = task_id_str
        finally:
            bar.close()
            if restore_hf_bars is not None:
                restore_hf_bars()
