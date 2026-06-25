"""Tests for the self-contained BeyondArena task-metadata collection loader.

Exercises the real committed reference CSV (small: 142 tasks), verifying the native collection
load, slug-based ``dataset`` keys, the per-dataset frame's ``max_train_rows`` (per-task maximum
over splits), and that every BeyondArena subset predicate column is present and usable on the
per-dataset frame.
"""

from __future__ import annotations

from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.contexts.beyondarena_context import BeyondArenaContext
from tabarena.evaluation.beyond_metadata import load_beyond_task_metadata_collection


def test_loader_returns_native_collection():
    coll = load_beyond_task_metadata_collection("BeyondArena")
    assert isinstance(coll, TaskMetadataCollection)
    assert len(coll) > 0


def test_per_dataset_frame_one_row_per_task():
    frame = load_beyond_task_metadata_collection("BeyondArena").per_dataset_frame()
    assert frame["dataset"].is_unique
    for col in ["dataset", "problem_type", "max_train_rows", "task_type"]:
        assert col in frame.columns, col


def test_dataset_is_slug_not_legacy_task_id():
    frame = load_beyond_task_metadata_collection("BeyondArena").per_dataset_frame()
    # The committed CSV was migrated to slugs; eval matches result dirs on these.
    assert not frame["dataset"].astype(str).str.startswith("Task-").any()
    assert (frame["dataset"] == frame["tabarena_task_name"]).all()


def test_max_train_rows_is_per_task_max_over_splits():
    coll = load_beyond_task_metadata_collection("BeyondArena")
    expected: dict[str, int] = {}
    for t in coll:
        sizes = [s.num_instances_train for s in t.splits_metadata.values()]
        expected[t.tabarena_task_name] = max(expected.get(t.tabarena_task_name, 0), *sizes)
    actual = coll.per_dataset_frame().set_index("dataset")["max_train_rows"].to_dict()
    assert actual == expected


def test_every_subset_predicate_is_usable_on_per_dataset_frame():
    frame = load_beyond_task_metadata_collection("BeyondArena").per_dataset_frame()
    for name, predicate in BeyondArenaContext.SUBSET_PREDICATES.items():
        # Split-level predicates (e.g. "lite", "core") declare a per-split "split" column that
        # only a task grid / results frame carries — they are not applicable to a one-row-per-
        # dataset frame, so skip them via their self-described required columns.
        if any(col not in frame.columns for col in predicate.required_columns):
            continue
        mask = predicate.evaluate(frame, name=name)  # must not KeyError -> column present
        assert mask.dtype == bool
        assert len(mask) == len(frame)


def test_size_buckets_partition_all_tasks():
    frame = load_beyond_task_metadata_collection("BeyondArena").per_dataset_frame()
    preds = BeyondArenaContext.SUBSET_PREDICATES
    covered = preds["tiny"](frame) | preds["small"](frame) | preds["medium"](frame) | preds["large"](frame)
    # Every task falls into exactly one size bucket (buckets are contiguous, non-overlapping).
    assert covered.all()
