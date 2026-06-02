"""Tests for the relocated ``BeyondArenaContext`` and its idempotent warehouse merge."""

from __future__ import annotations

import pandas as pd

from tabarena.evaluation.context.beyond_arena import BeyondArenaContext, _merge_warehouse_fields


def test_merge_is_noop_when_fields_already_present():
    df = pd.DataFrame({"dataset": ["a"], "task_type": ["random"], "num_text_cols": [1]})
    warehouse = pd.DataFrame({"dataset": ["a"], "task_type": ["WRONG"], "num_text_cols": [99]})
    out = _merge_warehouse_fields(df, warehouse)
    # Existing, non-null values are not overwritten.
    assert out.loc[0, "task_type"] == "random"
    assert out.loc[0, "num_text_cols"] == 1


def test_merge_repopulates_missing_column():
    df = pd.DataFrame({"dataset": ["a"]})
    warehouse = pd.DataFrame({"dataset": ["a"], "task_type": ["random"]})
    out = _merge_warehouse_fields(df, warehouse)
    assert out.loc[0, "task_type"] == "random"


def test_merge_repopulates_all_null_column():
    df = pd.DataFrame({"dataset": ["a"], "task_type": [None]})
    warehouse = pd.DataFrame({"dataset": ["a"], "task_type": ["random"]})
    out = _merge_warehouse_fields(df, warehouse)
    assert out.loc[0, "task_type"] == "random"


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
