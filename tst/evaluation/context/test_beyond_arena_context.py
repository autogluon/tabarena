"""Tests for `BeyondArenaContext` — native-collection construction + warehouse subset filters."""

from __future__ import annotations

import pytest

from tabarena.evaluation.context.beyond_arena import BeyondArenaContext


@pytest.fixture(scope="module")
def ctx() -> BeyondArenaContext:
    # methods=[] keeps construction light; task_metadata="BeyondArena" loads the committed CSV.
    return BeyondArenaContext(task_metadata="BeyondArena", methods=[])


def test_constructs_from_native_collection(ctx):
    # Regression: the BeyondArena CSV is in the native (per-split) schema and is reconstructed
    # into a TaskMetadataCollection. Previously this raised TypeError because a DataFrame was
    # handed to the (collection-only) base context.
    assert len(ctx.task_metadata_collection) > 0


def test_task_grid_exposes_warehouse_columns(ctx):
    grid = ctx.task_metadata_collection.task_grid()
    for col in ("task_type", "num_cols_after_preprocessing", "num_text_cols", "num_high_cardinality_cats"):
        assert col in grid.columns


def test_warehouse_subset_predicates_run(ctx, tmp_path):
    # "temporal" / "text" need warehouse columns the base task_grid did not expose before.
    temporal = ctx._subset_dataset_fold_repeats(subset="temporal")
    assert all(isinstance(t[0], str) for t in temporal)
    runner = ctx.make_experiment_batch_runner(expname=str(tmp_path), subset="text")
    # every kept split is a real split of the (filtered) collection
    assert set(runner.task_metadata_collection.dataset_fold_repeats()).issubset(
        set(ctx.task_metadata_collection.dataset_fold_repeats()),
    )
