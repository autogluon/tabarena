"""Tests for `SubsetPredicate` — self-describing subset filters (required-column validation)."""

from __future__ import annotations

import pandas as pd
import pytest

from tabarena.nips2025_utils.compare import _evaluate_subset_expression
from tabarena.nips2025_utils.subset_predicate import SubsetPredicate


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
