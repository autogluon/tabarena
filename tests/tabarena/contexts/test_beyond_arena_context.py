"""Tests for `BeyondArenaContext` — native-collection construction + warehouse subset filters."""

from __future__ import annotations

import pytest

from tabarena.contexts import BeyondArenaContext


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
    for col in (
        "task_type",
        "num_cols_after_preprocessing",
        "num_text_cols",
        "num_high_cardinality_cats",
        "group_labels",
    ):
        assert col in grid.columns


def test_warehouse_subset_predicates_run(ctx):
    # "temporal" / "text" need warehouse columns the base task_grid did not expose before.
    temporal = ctx.task_metadata_collection.subset_tasks(subset="temporal", predicates=ctx.subset_predicates)
    assert all(isinstance(t[0], str) for t in temporal.dataset_fold_repeats())
    text = ctx.task_metadata_collection.subset_tasks(subset="text", predicates=ctx.subset_predicates)
    # every kept split is a real split of the unfiltered collection
    assert set(text.dataset_fold_repeats()).issubset(set(ctx.task_metadata_collection.dataset_fold_repeats()))


class TestPresets:
    def test_beyond_presets_resolve_offline(self):
        ctx = BeyondArenaContext()
        method_names = {m.method for m in ctx.method_metadata_collection.method_metadata_lst}
        assert {"LinearModel", "RandomForest", "CatBoost", "TA-TabM", "TA-TabPFN-2.6"} <= method_names
        assert all(m.suite == "beyond_iid_benchmark_2026" for m in ctx.method_metadata_collection.method_metadata_lst)
        assert len(ctx.task_metadata_collection) > 0

    def test_only_beyondarena_preset_supported(self):
        # "BeyondArena" is the single preset name; legacy variants and the base
        # context's "tabarena" are rejected for both methods and task_metadata.
        for bad_methods in ("beyond", "tabarena", "nope"):
            with pytest.raises(ValueError, match="preset"):
                BeyondArenaContext(methods=bad_methods, task_metadata="BeyondArena")
        for bad_task_metadata in ("beyond", "tabarena", "nope"):
            with pytest.raises(ValueError, match="preset"):
                BeyondArenaContext(methods=[], task_metadata=bad_task_metadata)


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

    def test_grouped_label_buckets_partition_grouped(self, ctx):
        grid = ctx.task_metadata_collection.task_grid()
        preds = ctx.subset_predicates
        grouped = preds["grouped"](grid)
        lpg = preds["grouped_lpg"].evaluate(grid, name="grouped_lpg")
        lps = preds["grouped_lps"].evaluate(grid, name="grouped_lps")
        assert not (lpg & lps).any()
        assert (grouped == (lpg | lps)).all()
        assert lpg.any() and lps.any()

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

    def test_core_keeps_only_committed_valid_tasks(self, ctx):
        # "core" == the committed (dataset, split) tasks (first folds_to_use splits per dataset).
        import pandas as pd

        from tabarena.contexts.beyondarena.context import CORE_TASKS_CSV

        grid = ctx.task_metadata_collection.task_grid()
        mask = ctx.subset_predicates["core"].evaluate(grid, name="core")

        committed = pd.read_csv(CORE_TASKS_CSV)
        valid = set(zip(committed["dataset"].astype(str), committed["split"].astype(int), strict=False))
        in_grid = set(zip(grid["dataset"].astype(str), grid["split"].astype(int), strict=False))
        kept = set(zip(grid.loc[mask, "dataset"].astype(str), grid.loc[mask, "split"].astype(int), strict=False))

        assert kept, "core kept no tasks"
        assert kept == (valid & in_grid)

    def test_core_keeps_each_dataset_first_splits(self, ctx):
        # The kept splits of each dataset are a contiguous prefix starting at its lowest split.
        grid = ctx.task_metadata_collection.task_grid()
        mask = ctx.subset_predicates["core"].evaluate(grid, name="core")
        kept = grid.loc[mask]
        for dataset, dataset_grid in grid.groupby("dataset"):
            all_splits = sorted(int(s) for s in dataset_grid["split"].unique())
            kept_splits = sorted(int(s) for s in kept.loc[kept["dataset"] == dataset, "split"])
            assert kept_splits == all_splits[: len(kept_splits)], dataset

    def test_core_composes_with_other_predicates(self, ctx):
        grid = ctx.task_metadata_collection.task_grid()
        preds = ctx.subset_predicates
        core = preds["core"].evaluate(grid, name="core")
        core_and_tiny = core & preds["tiny"].evaluate(grid, name="tiny")
        assert core_and_tiny.sum() <= core.sum()
        assert (core_and_tiny & ~core).sum() == 0
