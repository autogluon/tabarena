"""Persist + retrieve the per-collection TabArena task-metadata CSV.

Downstream consumers (e.g. ``tabflow_slurm``) need a stable on-disk manifest that
lists which TabArena tasks exist for a given benchmark suite so workers can iterate
them without re-running the conversion. The contract is:

* The manifest is written to ``<openml_cache>/tabarena_metadata_cache/<suite>_tasks_metadata.csv``.
* Columns mirror :meth:`TabArenaTaskMetadata.to_dataframe` and include a
  ``data_foundry_uri`` column with the collection-entry relative path.

This module keeps that contract intact while delegating download + conversion to
:class:`DataFoundryAdapter` and the underlying ``data_foundry`` collection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import openml

from tabarena.benchmark.task.data_foundry.adapter import DataFoundryAdapter

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from data_foundry.collections import DatasetCollection


def _init_openml_cache(openml_cache: str | Path) -> None:
    print(f"Setting OpenML cache directory to: {openml_cache}")
    openml.config.set_root_cache_directory(root_cache_directory=str(openml_cache))


def get_path_to_metadata_cache() -> Path:
    """Return the directory where benchmark-suite metadata CSVs are stored.

    Resolves to ``<openml.config._root_cache_directory>/tabarena_metadata_cache``.
    The directory is not created here — callers materialize it on write.
    """
    return openml.config._root_cache_directory / "tabarena_metadata_cache"


def prepare_collection_for_tabarena(
    *,
    collection: DatasetCollection,
    benchmark_suite_name: str | None = None,
    cache_dir: str | None = None,
    openml_cache: str | Path | None = None,
    force_download: bool = False,
    evaluation_metrics: dict[str, list[str]] | None = None,
    show_sample: bool = True,
) -> Path:
    """Materialize TabArena UserTasks for ``collection`` and save the metadata CSV.

    Convenience wrapper that:

    1. Optionally points OpenML's root cache at ``openml_cache`` (so the UserTask
       pickles + the metadata CSV land in a controlled location, e.g. a shared
       filesystem).
    2. Runs :class:`DataFoundryAdapter` over the collection — downloading via its
       source if needed (HuggingFace) or just resolving local paths
       (LocalWarehouseSource).
    3. Writes the resulting DataFrame to
       ``<openml_cache>/tabarena_metadata_cache/<suite>_tasks_metadata.csv``.

    Args:
        collection: The :class:`DatasetCollection` to process.
        benchmark_suite_name: Identifier used to name the metadata CSV. Defaults to
            ``collection.name``. Pass an explicit value when one source collection
            backs multiple named suites.
        cache_dir: Optional override for the data_foundry cache (e.g. when
            downloading from HF). Forwarded to ``DatasetCollection.iter_containers``.
        openml_cache: If given, sets OpenML's root cache directory before any
            artifact is written. Affects where ``UserTask`` pickles and the
            metadata CSV land.
        force_download: When ``True``, re-fetch every container from its source.
        evaluation_metrics: Override TabArena's allowed-eval-metric set. See
            :data:`adapter.DEFAULT_EVAL_METRICS`.
        show_sample: When ``True``, prints a 5-row sample of the metadata DataFrame.

    Returns:
        The path to the metadata CSV that was written.
    """
    if openml_cache is not None:
        _init_openml_cache(openml_cache=openml_cache)

    suite_name = benchmark_suite_name if benchmark_suite_name is not None else collection.name
    print(f"Preparing data foundry collection {collection.name!r} for TabArena suite {suite_name!r}...")

    adapter = DataFoundryAdapter(
        collection=collection,
        evaluation_metrics=evaluation_metrics,
    )
    metadata_df: pd.DataFrame = adapter.to_tabarena_user_tasks(
        cache_dir=cache_dir,
        force_download=force_download,
        show_sample=show_sample,
    )

    path_to_metadata = get_path_to_metadata_cache() / f"{suite_name}_tasks_metadata.csv"
    path_to_metadata.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving metadata to {path_to_metadata}")
    metadata_df.to_csv(path_to_metadata, index=False)
    return path_to_metadata


def get_metadata_for_benchmark_suite(
    benchmark_suite_name: str,
    *,
    openml_cache: str | Path | None = None,
) -> Path:
    """Return the metadata CSV path for ``benchmark_suite_name``.

    Args:
        benchmark_suite_name: The suite name used when the manifest was written
            (see :func:`prepare_collection_for_tabarena`).
        openml_cache: If given, sets OpenML's root cache directory before
            resolving the path. Use the same value here as during preparation.

    Returns:
        Path to the existing ``<suite>_tasks_metadata.csv``.

    Raises:
        FileNotFoundError: If no manifest has been written for ``benchmark_suite_name``.
    """
    if openml_cache is not None:
        _init_openml_cache(openml_cache=openml_cache)

    path_to_metadata = get_path_to_metadata_cache() / f"{benchmark_suite_name}_tasks_metadata.csv"
    if not path_to_metadata.exists():
        raise FileNotFoundError(
            f"Metadata file {path_to_metadata} does not exist. "
            f"Call prepare_collection_for_tabarena(...) for suite "
            f"{benchmark_suite_name!r} first.",
        )
    return path_to_metadata
