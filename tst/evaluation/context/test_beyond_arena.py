"""Tests for ``BeyondArenaContext`` (collection-native task metadata)."""

from __future__ import annotations

from tabarena.evaluation.context.beyond_arena import BeyondArenaContext


def test_init_resolves_source_name_to_native_collection():
    ctx = BeyondArenaContext()
    assert len(ctx.task_metadata_collection) > 0
    # The committed CSV is self-contained: warehouse + predicate columns inline, no merge step.
    frame = ctx.task_metadata_collection.per_dataset_frame()
    assert {"task_type", "num_text_cols", "num_high_cardinality_cats", "max_train_rows"} <= set(frame.columns)


def test_subset_predicates_cover_beyond_subsets():
    expected = {
        "binary",
        "multiclass",
        "classification",
        "regression",
        "tiny",
        "small",
        "medium",
        "large",
        "random",
        "iid",
        "temporal",
        "grouped",
        "low-dim",
        "high-dim",
        "text",
        "high-cardinality",
        "lite",
    }
    assert expected <= set(BeyondArenaContext.SUBSET_PREDICATES)
