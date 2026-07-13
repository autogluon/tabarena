"""Tests for ``BeyondArenaContext`` (collection-native task metadata)."""

from __future__ import annotations

from tabarena.contexts import BeyondArenaContext


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


def test_subset_shortcuts_resolve_on_task_grid():
    """Every shortcut's expression list (including negated atoms like ``"!large"``) evaluates
    against the task grid and selects a non-empty, strictly smaller slice than the full grid.
    """
    from tabarena.benchmark.task.metadata import BeyondArenaTaskMetadataCollection
    from tabarena.nips2025_utils.compare import _evaluate_subset_expression

    grid = BeyondArenaTaskMetadataCollection().task_grid()
    predicates = BeyondArenaContext.SUBSET_PREDICATES
    for name, expressions in BeyondArenaContext.SUBSET_SHORTCUTS.items():
        selected = grid
        for expression in expressions:
            mask = _evaluate_subset_expression(expression, selected, predicates=predicates)
            selected = selected[mask.values]
        assert 0 < len(selected) < len(grid), name


def test_subset_shortcut_name_matches_negated_expressions():
    assert BeyondArenaContext.subset_shortcut_name(["core", "high-cardinality", "!large"]) == "hc_nolarge"
    # order-insensitive
    assert BeyondArenaContext.subset_shortcut_name(["!large", "core", "lite", "high-cardinality"]) == "hc_nolarge_lite"
    assert BeyondArenaContext.subset_shortcut_name(["core", "high-cardinality"]) == "high_cardinality"
    assert BeyondArenaContext.subset_shortcut_name(["core", "!large"]) is None
