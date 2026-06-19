"""Tests for `SubsetPredicate` — self-describing subset filters (required-column validation)."""

from __future__ import annotations

import pandas as pd
import pytest

from tabarena.nips2025_utils.compare import _evaluate_subset_expression
from tabarena.nips2025_utils.subset_predicate import SubsetPredicate, tasks_in_frame


def test_evaluate_runs_when_required_columns_present():
    pred = SubsetPredicate(lambda df: df["x"] > 0, ("x",))
    assert pred.evaluate(pd.DataFrame({"x": [-1, 1, 2]})).tolist() == [False, True, True]


def test_evaluate_reports_missing_required_column():
    pred = SubsetPredicate(lambda df: df["max_train_rows"] <= 100, ("max_train_rows",))
    with pytest.raises(ValueError, match="max_train_rows"):
        pred.evaluate(pd.DataFrame({"problem_type": ["binary"]}), name="small")


def test_callable_is_backward_compatible():
    # A SubsetPredicate is callable, so legacy `predicate(df)` call sites keep working.
    pred = SubsetPredicate(lambda df: df["x"] > 0, ("x",))
    assert pred(pd.DataFrame({"x": [1, -1]})).tolist() == [True, False]


def test_subset_expression_surfaces_missing_column():
    predicates = {"small": SubsetPredicate(lambda df: df["max_train_rows"] <= 100, ("max_train_rows",))}
    with pytest.raises(ValueError, match="max_train_rows"):
        _evaluate_subset_expression("small", pd.DataFrame({"problem_type": ["binary"]}), predicates=predicates)


class TestTasksInFrame:
    """`tasks_in_frame` — a data-dependent (dataset, split) membership predicate."""

    @staticmethod
    def _grid() -> pd.DataFrame:
        # dataset "a" has splits 0,1,2; dataset "b" has splits 0,1.
        return pd.DataFrame(
            {
                "dataset": ["a", "a", "a", "b", "b"],
                "split": [0, 1, 2, 0, 1],
            },
        )

    def test_keeps_only_listed_dataset_split_pairs(self):
        # "first 2 splits of a, first 1 split of b"
        valid = pd.DataFrame({"dataset": ["a", "a", "b"], "split": [0, 1, 0]})
        mask = tasks_in_frame(valid).evaluate(self._grid())
        assert mask.tolist() == [True, True, False, True, False]

    def test_required_columns_are_dataset_and_split(self):
        pred = tasks_in_frame(pd.DataFrame({"dataset": ["a"], "split": [0]}))
        assert pred.required_columns == ("dataset", "split")
        with pytest.raises(ValueError, match="split"):
            pred.evaluate(pd.DataFrame({"dataset": ["a"]}))

    def test_custom_source_columns(self):
        valid = pd.DataFrame({"ds": ["a"], "s": [2]})
        mask = tasks_in_frame(valid, dataset_col="ds", split_col="s").evaluate(self._grid())
        assert mask.tolist() == [False, False, True, False, False]
