from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from tabarena.utils.wrapper_utils import (
    gpu_cell_budget,
    root_handlers_preserved,
    rows_within_budget,
    stratified_row_subsample,
    univariate_top_columns,
)


def test_root_handlers_preserved_removes_added_handler():
    root = logging.getLogger()
    before = list(root.handlers)
    added = logging.StreamHandler()
    with root_handlers_preserved():
        root.addHandler(added)
        assert added in root.handlers
    assert root.handlers == before


def test_root_handlers_preserved_keeps_existing_handlers():
    root = logging.getLogger()
    existing = logging.NullHandler()
    root.addHandler(existing)
    try:
        with root_handlers_preserved():
            logging.info("module-level logging installs a root handler when none exists")
        assert existing in root.handlers
    finally:
        root.removeHandler(existing)


def test_rows_within_budget_keeps_small_tables_and_caps_large_ones():
    assert rows_within_budget(1_000, 10, cell_budget=100_000) == 1_000
    assert rows_within_budget(42_423, 500, cell_budget=14_300_000) == 28_600
    assert rows_within_budget(96_064, 1_799, cell_budget=14_300_000, min_rows=5_000) == 7_948
    # the minimum wins over the budget but never exceeds the table
    assert rows_within_budget(3_000, 10_000, cell_budget=1_000, min_rows=5_000) == 3_000
    assert rows_within_budget(20_000, 10_000, cell_budget=1_000, min_rows=5_000) == 5_000


def test_stratified_subsample_keeps_every_class_in_proportion():
    y = np.array([0] * 900 + [1] * 90 + [2] * 10)
    keep = stratified_row_subsample(y, 100, classification=True, seed=0)
    assert len(keep) == 100 and len(set(keep)) == 100
    assert np.all(np.diff(keep) > 0)  # sorted
    counts = np.bincount(y[keep], minlength=3)
    assert counts[2] >= 1 and 85 <= counts[0] <= 91 and 8 <= counts[1] <= 10


def test_subsample_is_identity_when_enough_rows_and_uniform_for_regression():
    y = np.arange(50, dtype=float)
    assert list(stratified_row_subsample(y, 50, classification=False, seed=0)) == list(range(50))
    keep = stratified_row_subsample(y, 20, classification=False, seed=1)
    assert len(keep) == 20 and np.all(np.diff(keep) > 0)


def test_univariate_top_columns_keeps_informative_columns_in_frame_order():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, size=n)
    X = pd.DataFrame(
        {
            "noise_a": rng.normal(size=n),
            "signal": y + rng.normal(scale=0.1, size=n),
            "constant": np.ones(n),
            "cat": pd.Categorical(np.where(y == 1, "yes", "no")),
            "noise_b": rng.normal(size=n),
        }
    )
    X.loc[X.index[::7], "signal"] = np.nan  # missing values take the median, the ranking still runs
    assert univariate_top_columns(X, y, 5, classification=True) == list(X.columns)
    assert univariate_top_columns(X, y, 2, classification=True) == ["signal", "cat"]
    y_reg = 3 * X["noise_b"].to_numpy() + rng.normal(scale=0.1, size=n)
    assert univariate_top_columns(X, y_reg, 1, classification=False) == ["noise_b"]


def test_gpu_cell_budget_without_cuda_is_none():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        assert gpu_cell_budget(bytes_per_cell=4700, safety=0.7) > 0
    else:
        assert gpu_cell_budget(bytes_per_cell=4700, safety=0.7) is None
