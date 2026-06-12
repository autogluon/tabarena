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


class TestPresets:
    def test_beyond_presets_resolve_offline(self):
        ctx = BeyondArenaContext()
        method_names = {m.method for m in ctx.method_metadata_collection.method_metadata_lst}
        assert {"LinearModel", "RandomForest", "CatBoost", "TA-TabM", "TA-TabPFN-2.6"} <= method_names
        assert all(
            m.artifact_name == "beyond_iid_benchmark_2026" for m in ctx.method_metadata_collection.method_metadata_lst
        )
        assert len(ctx.task_metadata_collection) > 0

    def test_beyond_task_metadata_alias(self):
        ctx = BeyondArenaContext(methods=[], task_metadata="beyond")
        assert len(ctx.task_metadata_collection) > 0

    def test_unknown_methods_preset_raises(self):
        with pytest.raises(ValueError, match="preset"):
            BeyondArenaContext(methods="nope", task_metadata="BeyondArena")


class TestSubsetPredicates:
    def test_all_predicates_evaluate_on_task_grid(self, ctx):
        grid = ctx.task_metadata_collection.task_grid()
        for name, predicate in ctx.subset_predicates.items():
            mask = predicate.evaluate(grid, name=name)
            assert mask.dtype == bool, name
            assert len(mask) == len(grid), name

    def test_task_type_buckets_partition_grid(self, ctx):
        grid = ctx.task_metadata_collection.task_grid()
        preds = ctx.subset_predicates
        combined = preds["iid"](grid) | preds["temporal"](grid) | preds["grouped"](grid)
        assert combined.all()
        assert (preds["random"](grid) == preds["iid"](grid)).all()

    def test_lite_keys_on_split(self, ctx):
        # "lite" == split 0 == (fold 0, repeat 0): one row per dataset on the grid.
        grid = ctx.task_metadata_collection.task_grid()
        mask = ctx.subset_predicates["lite"].evaluate(grid, name="lite")
        assert (grid.loc[mask, "split"] == 0).all()
        assert mask.sum() == grid["dataset"].nunique()

    def test_default_subsets_reference_defined_predicates(self, ctx):
        for subset in ctx._default_subsets:
            for name in subset:
                assert name in ctx.subset_predicates, name
