"""Tests for the self-contained BeyondArena task-metadata loader/collapse.

Exercises the real committed reference CSV (small: 142 tasks), verifying the collapse to one row
per task, the derived ``max_train_rows``, slug-based ``dataset`` keys, and that every BeyondArena
subset predicate column is present and usable on the collapsed frame.
"""

from __future__ import annotations

import pandas as pd
from tabarena.benchmark.task.metadata.sources.base import (
    InMemoryTaskMetadataSource,
    committed_metadata_path,
)
from tabarena.evaluation.beyond_metadata import load_beyond_task_metadata
from tabarena.evaluation.context.beyond_arena import BeyondArenaContext


def test_collapse_one_row_per_task():
    tm = load_beyond_task_metadata("BeyondArena")
    assert tm["tid"].is_unique
    assert len(tm) == tm["tid"].nunique()
    for col in ["dataset", "problem_type", "max_train_rows", "task_type", "n_splits", "name"]:
        assert col in tm.columns, col


def test_dataset_is_slug_not_legacy_task_id():
    tm = load_beyond_task_metadata("BeyondArena")
    # The committed CSV was migrated to slugs; eval matches result dirs on these.
    assert not tm["dataset"].astype(str).str.startswith("Task-").any()
    assert (tm["dataset"] == tm["tabarena_task_name"]).all()


def test_max_train_rows_is_per_task_max_over_folds():
    per_split = pd.concat(
        t.to_dataframe(add_old_minimal_metadata=True)
        for t in InMemoryTaskMetadataSource(committed_metadata_path("BeyondArena")).load()
    )
    expected = per_split.groupby("tid")["n_samples_train_per_fold"].max()
    tm = load_beyond_task_metadata("BeyondArena").set_index("tid")
    assert (tm["max_train_rows"] == expected.reindex(tm.index)).all()


def test_every_subset_predicate_is_usable_on_collapsed_frame():
    tm = load_beyond_task_metadata("BeyondArena")
    for name, predicate in BeyondArenaContext.SUBSET_PREDICATES.items():
        if name == "lite":
            continue  # "lite" needs a per-fold "fold" column (only on df_results)
        mask = predicate(tm)  # must not KeyError -> column present
        assert mask.dtype == bool
        assert len(mask) == len(tm)


def test_size_buckets_partition_all_tasks():
    tm = load_beyond_task_metadata("BeyondArena")
    preds = BeyondArenaContext.SUBSET_PREDICATES
    covered = preds["tiny"](tm) | preds["small"](tm) | preds["medium"](tm) | preds["large"](tm)
    # Every task falls into exactly one size bucket (buckets are contiguous, non-overlapping).
    assert covered.all()
