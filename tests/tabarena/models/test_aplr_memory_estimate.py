from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabarena.models.aplr.model import APLRModel

GB = 1024**3


def _frame(n_rows: int, *, n_num: int = 0, cat_levels: tuple[int, ...] = (), n_missing_num: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols: dict[str, object] = {f"x{i}": rng.standard_normal(n_rows) for i in range(n_num)}
    for j, k in enumerate(cat_levels):
        cols[f"c{j}"] = pd.Categorical((np.arange(n_rows) % k).astype(str))
    for i in range(n_missing_num):
        values = rng.standard_normal(n_rows)
        values[::7] = np.nan
        cols[f"m{i}"] = values
    return pd.DataFrame(cols)


def _estimate(X: pd.DataFrame, *, problem_type: str, num_classes: int | None) -> int:
    return APLRModel.estimate_memory_usage_static(X=X, problem_type=problem_type, num_classes=num_classes)


def test_static_estimate_is_registered():
    """AutoGluon only budgets fold parallelism through the estimate if the class advertises it."""
    assert APLRModel.can_estimate_memory_usage_static()


@pytest.mark.parametrize(
    ("n_rows", "n_num", "cat_levels", "n_missing_num", "measured_peak_gb"),
    [
        # Amazon_employee_access fold-0 child fit: 8 categoricals, 8067 one-hot columns.
        (19_115, 0, (7_000, 1_000, 67), 0, 7.85),
        # kddcup09_appetency fold-0 child fit: 178 numerics with NaNs, 30 categoricals, 17641 levels.
        (29_166, 0, (8_820, 8_821), 178, 26.36),
    ],
)
def test_binary_estimate_brackets_measured_peaks(n_rows, n_num, cat_levels, n_missing_num, measured_peak_gb):
    """The estimate must cover the measured single-fold peaks without exceeding twice them."""
    X = _frame(n_rows, n_num=n_num, cat_levels=cat_levels, n_missing_num=n_missing_num)
    estimate_gb = _estimate(X, problem_type="binary", num_classes=2) / GB
    assert measured_peak_gb <= estimate_gb <= 2 * measured_peak_gb


def test_one_hot_width_not_frame_size_drives_estimate():
    """A narrow frame with one high-cardinality categorical outweighs a wide numeric one."""
    wide_numeric = _frame(20_000, n_num=200)
    narrow_categorical = _frame(20_000, cat_levels=(10_000,))
    baseline = APLRModel._MEMORY_BASELINE_BYTES
    dense_numeric = _estimate(wide_numeric, problem_type="regression", num_classes=None) - baseline
    dense_categorical = _estimate(narrow_categorical, problem_type="binary", num_classes=2) - baseline
    assert dense_categorical > 10 * dense_numeric


@pytest.mark.parametrize(
    ("num_classes", "measured_peak_gb"),
    [
        # SDSS17 quarter subsample (11382 rows, 9 numerics, 4249 one-hot columns), 3 and 6 classes.
        (3, 5.30),
        (6, 9.04),
    ],
)
def test_multiclass_estimate_brackets_measured_peaks(num_classes, measured_peak_gb):
    X = _frame(11_382, n_num=9, cat_levels=(4_000, 249))
    estimate_gb = _estimate(X, problem_type="multiclass", num_classes=num_classes) / GB
    assert measured_peak_gb <= estimate_gb <= 2 * measured_peak_gb


def test_multiclass_grows_with_number_of_classes():
    """One logit model per class stays resident: the dense part grows linearly with the class count."""
    X = _frame(45_530, n_num=9, cat_levels=(6_721,))
    baseline = APLRModel._MEMORY_BASELINE_BYTES
    binary = _estimate(X, problem_type="binary", num_classes=2) - baseline
    three_class = _estimate(X, problem_type="multiclass", num_classes=3) - baseline
    six_class = _estimate(X, problem_type="multiclass", num_classes=6) - baseline
    assert binary < three_class < six_class
    assert six_class - three_class == pytest.approx(3 * APLRModel._MEMORY_COPIES_PER_CLASS * 45_530 * 6_730 * 8)


def test_missing_numeric_values_add_indicator_columns():
    complete = _frame(10_000, n_num=50)
    with_missing = _frame(10_000, n_missing_num=50)
    assert _estimate(with_missing, problem_type="regression", num_classes=None) > _estimate(
        complete, problem_type="regression", num_classes=None
    )
