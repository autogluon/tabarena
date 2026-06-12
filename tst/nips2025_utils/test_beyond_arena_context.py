"""Tests for ``BeyondArenaContext`` (presets, subset predicates, task-grid compatibility).

Everything here is offline: the ``"beyond"`` task preset loads the committed
``BeyondArena_tasks_metadata.csv`` package data and downloads nothing.
"""

from __future__ import annotations

import pytest

from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.nips2025_utils.beyond_arena_context import BeyondArenaContext


def _ctx(**kwargs) -> BeyondArenaContext:
    # methods=[] keeps construction light (no method-metadata deepcopy).
    kwargs.setdefault("methods", [])
    kwargs.setdefault("task_metadata", TaskMetadataCollection.from_preset("BeyondArena-lite"))
    return BeyondArenaContext(**kwargs)


class TestPresets:
    def test_beyond_presets_resolve_offline(self):
        ctx = BeyondArenaContext()
        method_names = {m.method for m in ctx.method_metadata_collection.method_metadata_lst}
        assert {"LinearModel", "RandomForest", "CatBoost", "TA-TabM", "TA-TabPFN-2.6"} <= method_names
        assert all(
            m.artifact_name == "beyond_iid_benchmark_2026" for m in ctx.method_metadata_collection.method_metadata_lst
        )
        assert len(ctx.task_metadata_collection.tasks) > 0

    def test_unknown_presets_raise(self):
        with pytest.raises(ValueError, match="preset"):
            BeyondArenaContext(methods=[], task_metadata="nope")
        with pytest.raises(ValueError, match="preset"):
            BeyondArenaContext(methods="nope", task_metadata=TaskMetadataCollection.from_preset("BeyondArena-lite"))

    def test_plain_collection_accepted(self):
        # No collection subclass / re-wrap needed: the base task_grid carries the
        # warehouse columns the Beyond predicates key on.
        ctx = _ctx()
        grid = ctx.task_metadata_collection.task_grid()
        assert {"task_type", "num_text_cols", "num_cols_after_preprocessing", "num_high_cardinality_cats"} <= set(
            grid.columns
        )


class TestSubsetPredicates:
    def test_all_predicates_evaluate_on_task_grid(self):
        ctx = _ctx()
        grid = ctx.task_metadata_collection.task_grid()
        for name, predicate in ctx.subset_predicates.items():
            mask = predicate.evaluate(grid, name=name)
            assert mask.dtype == bool, name
            assert len(mask) == len(grid), name

    def test_task_type_buckets_partition_grid(self):
        ctx = _ctx()
        grid = ctx.task_metadata_collection.task_grid()
        preds = ctx.subset_predicates
        combined = preds["iid"](grid) | preds["temporal"](grid) | preds["grouped"](grid)
        assert combined.all()
        assert (preds["random"](grid) == preds["iid"](grid)).all()

    def test_lite_keys_on_split(self):
        ctx = _ctx()
        grid = ctx.task_metadata_collection.task_grid()
        mask = ctx.subset_predicates["lite"].evaluate(grid, name="lite")
        assert (grid.loc[mask, "split"] == 0).all()
        # the lite preset has exactly one split per dataset, so lite keeps everything
        assert mask.all()

    def test_default_subsets_reference_defined_predicates(self):
        ctx = _ctx()
        for subset in ctx._default_subsets:
            for name in subset:
                assert name in ctx.subset_predicates, name
