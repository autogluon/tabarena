"""Load BeyondArena (data-foundry) task metadata for evaluation, self-contained.

The committed reference CSV shipped with the metadata source
(``benchmark/task/metadata/sources/data/<name>_tasks_metadata.csv``) already carries the
warehouse-level fields inline (``task_type``, ``num_text_cols``, ``num_high_cardinality_cats``,
``num_cols_after_preprocessing``, ``missing_value_fraction``, ``domain``, ``dataset_year``,
``source``).

The single collapsed frame is suitable as *both* the ``task_metadata`` (for ``compare``) and the
``data_foundry_metadata`` (for ``subset_tasks_data_foundry`` / ``get_subsets_per_dataset``) — it
carries ``dataset``, ``problem_type``, ``max_train_rows`` and the dtype/size/split columns the
BeyondArena subset predicates need.
"""

from __future__ import annotations

import pandas as pd

from tabarena.benchmark.task.metadata.sources.base import (
    InMemoryTaskMetadataSource,
    committed_metadata_path,
)


def load_beyond_task_metadata(source: str = "BeyondArena") -> pd.DataFrame:
    """Load + collapse BeyondArena task metadata to one row per task.

    Args:
        source: Either a registered suite/collection name whose committed CSV ships as package
            data (e.g. ``"BeyondArena"``), or a path to a ``*_tasks_metadata.csv`` reference file.

    Returns:
        One row per task, carrying the legacy minimal columns (``tid``, ``name``, ``dataset``,
        ``n_samples_train_per_fold``/``n_samples_test_per_fold``) plus ``n_splits`` and
        ``max_train_rows`` (= the per-task maximum training-fold size). Usable as both the eval
        ``task_metadata`` and the ``data_foundry_metadata`` subset-filtering frame.
    """
    # A bare name resolves to the committed package-data CSV; anything else is treated as a path.
    data = committed_metadata_path(source) if _looks_like_suite_name(source) else source
    task_metadata = InMemoryTaskMetadataSource(data).load()

    # One row per (task, split), with the legacy minimal columns re-attached.
    df = pd.concat([tm.to_dataframe(add_old_minimal_metadata=True) for tm in task_metadata])

    # Count splits per task before collapsing — the per-task row carries only one fold's data.
    n_splits_per_tid = df.groupby("tid").size().rename("n_splits")
    # Reduce to one row per task; keep the row with the largest training fold so that
    # ``n_samples_train_per_fold`` (and ``max_train_rows`` below) reflect the per-task maximum.
    ta_task_metadata = (
        df.sort_values("n_samples_train_per_fold", ascending=False).groupby("tid").first().reset_index()
    )
    ta_task_metadata = ta_task_metadata.merge(n_splits_per_tid, on="tid", how="left")
    # The BeyondArena size predicates read ``max_train_rows`` directly off the metadata frame
    # (``subset_tasks_data_foundry`` does not go through ``compare``'s alias path).
    ta_task_metadata["max_train_rows"] = ta_task_metadata["n_samples_train_per_fold"]

    return ta_task_metadata


def _looks_like_suite_name(source: str) -> bool:
    """True if ``source`` is a bare suite/collection name rather than a CSV path."""
    return not (source.endswith(".csv") or "/" in source or "\\" in source)
