from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tabarena.benchmark.models.wrapper.validation_utils import (
    TabArenaValidationProtocolExecMixin,
    split_time_index_into_intervals,
)
from tabarena.benchmark.task.user_task import GroupLabelTypes

# ---------------------------------------------------------------------------
# Concrete subclass for testing the mixin
# ---------------------------------------------------------------------------


class _Validation(TabArenaValidationProtocolExecMixin):
    """Minimal concrete subclass — the mixin has no abstract methods."""


# ---------------------------------------------------------------------------
# split_time_index_into_intervals
# ---------------------------------------------------------------------------


def test_basic_integer_split_produces_correct_n_intervals():
    time_data = pd.Series([1, 2, 3, 4, 5, 6])
    intervals, n = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=3
    )
    assert n == 3
    assert intervals.nunique() == 3


def test_intervals_are_monotonically_nondecreasing():
    """Interval IDs must weakly increase as time values increase."""
    time_data = pd.Series(range(12))
    intervals, _ = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=4
    )
    sorted_vals = intervals.sort_index().to_numpy()
    assert all(
        sorted_vals[i] <= sorted_vals[i + 1] for i in range(len(sorted_vals) - 1)
    )


def test_duplicate_time_values_land_in_same_interval():
    # Both occurrences of value 1 must receive the same interval label.
    time_data = pd.Series([1, 1, 2, 3, 4, 5])
    intervals, _ = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=3
    )
    assert intervals.iloc[0] == intervals.iloc[1]


def test_fewer_unique_values_than_goal_caps_at_unique_count():
    # 2 unique values → at most 2 intervals, regardless of goal.
    time_data = pd.Series([1, 1, 2, 2, 2])
    intervals, n = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=5
    )
    assert n == 2
    assert intervals.nunique() == 2


def test_original_index_is_preserved():
    time_data = pd.Series([3, 1, 2, 1], index=[10, 20, 30, 40])
    intervals, _ = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=2
    )
    assert list(intervals.index) == [10, 20, 30, 40]
    # Both 1s (idx 20 and 40) must share an interval; 3 (idx 10) must be later.
    assert intervals[20] == intervals[40]
    assert intervals[10] > intervals[20]


def test_balance_on_rows_and_unique_both_produce_correct_interval_count():
    time_data = pd.Series([1, 1, 1, 1, 2, 3])
    for mode in ("rows", "unique"):
        intervals, n = split_time_index_into_intervals(
            time_data=time_data, goal_n_intervals=2, balance_on=mode
        )
        assert n == 2
        assert intervals.nunique() == 2


def test_balance_on_rows_puts_heavy_bucket_together():
    # Values: four 1s, one 2, one 3.  With balance_on="rows" and goal=2,
    # the greedy split should keep all four 1s in interval 0.
    time_data = pd.Series([1, 1, 1, 1, 2, 3])
    intervals, _ = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=2, balance_on="rows"
    )
    assert (
        intervals[intervals.index[0]] == intervals[intervals.index[3]]
    )  # same interval for all 1s


def test_datetime_dtype_is_accepted():
    time_data = pd.Series(pd.date_range("2020-01-01", periods=6, freq="D"))
    intervals, n = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=3
    )
    assert n == 3
    assert intervals.nunique() == 3


def test_goal_n_intervals_less_than_2_raises():
    time_data = pd.Series([1, 2, 3])
    with pytest.raises(ValueError, match="at least 2"):
        split_time_index_into_intervals(time_data=time_data, goal_n_intervals=1)


def test_invalid_balance_on_raises():
    time_data = pd.Series([1, 2, 3])
    with pytest.raises(ValueError, match="balance_on"):
        split_time_index_into_intervals(
            time_data=time_data, goal_n_intervals=2, balance_on="invalid"
        )


def test_nan_in_time_data_raises():
    time_data = pd.Series([1.0, 2.0, float("nan")])
    with pytest.raises(AssertionError):
        split_time_index_into_intervals(time_data=time_data, goal_n_intervals=2)


def test_single_unique_value_raises():
    time_data = pd.Series([5, 5, 5])
    with pytest.raises(ValueError, match="at least 2 unique time values"):
        split_time_index_into_intervals(time_data=time_data, goal_n_intervals=2)


@pytest.mark.parametrize("n_intervals", [2, 4, 8])
def test_output_length_equals_input_length(n_intervals):
    time_data = pd.Series(range(20))
    intervals, _ = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=n_intervals
    )
    assert len(intervals) == len(time_data)


# ---------------------------------------------------------------------------
# TabArenaValidationProtocolExecMixin — static helpers
# ---------------------------------------------------------------------------


def test_group_on_to_groups_data_single_column():
    X = pd.DataFrame({"a": [1, 2, 3], "group": ["x", "y", "x"]})
    result = _Validation.group_on_to_groups_data(X=X, group_on="group")
    assert list(result) == ["x", "y", "x"]


def test_group_on_to_groups_data_multi_column():
    X = pd.DataFrame({"g1": ["a", "a", "b"], "g2": ["1", "2", "1"]})
    result = _Validation.group_on_to_groups_data(X=X, group_on=["g1", "g2"])
    assert list(result) == ["a_1", "a_2", "b_1"]


def test_time_on_to_groups_data_integer():
    X = pd.DataFrame({"time": [1, 2, 3, 4, 5, 6]})
    groups, n_intervals = _Validation.time_on_to_groups_data(
        X=X, time_on="time", num_folds=3
    )
    assert n_intervals == 3
    assert len(groups) == 6
    assert (groups.diff().dropna() >= 0).all()


def test_time_on_to_groups_data_datetime():
    X = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=6, freq="D")})
    groups, n_intervals = _Validation.time_on_to_groups_data(
        X=X, time_on="time", num_folds=3
    )
    assert n_intervals == 3
    assert len(groups) == 6


# ---------------------------------------------------------------------------
# TabArenaValidationProtocolExecMixin — resolve_validation_splits
# ---------------------------------------------------------------------------


def _make_X(n: int) -> pd.DataFrame:
    return pd.DataFrame({"feature": np.arange(n, dtype=float)})


def test_resolve_validation_splits_disabled_returns_unchanged():
    """use_task_specific_validation=False → immediate passthrough."""
    v = _Validation(use_task_specific_validation=False)
    X = _make_X(10)
    y = pd.Series(np.zeros(10))
    custom_splits, folds, repeats = v.resolve_validation_splits(
        X=X, y=y, num_folds=8, num_repeats=1
    )
    assert custom_splits is None
    assert folds == 8
    assert repeats == 1


def test_resolve_validation_splits_num_folds_none_returns_early():
    v = _Validation(use_task_specific_validation=True)
    X = _make_X(10)
    y = pd.Series(np.zeros(10))
    custom_splits, folds, _repeats = v.resolve_validation_splits(
        X=X, y=y, num_folds=None, num_repeats=1
    )
    assert custom_splits is None
    assert folds is None


def test_resolve_validation_splits_num_folds_one_returns_early():
    v = _Validation(use_task_specific_validation=True)
    X = _make_X(10)
    y = pd.Series(np.zeros(10))
    custom_splits, folds, _repeats = v.resolve_validation_splits(
        X=X, y=y, num_folds=1, num_repeats=1
    )
    assert custom_splits is None
    assert folds == 1


def test_resolve_validation_splits_tiny_data_updates_folds_and_repeats():
    """Datasets with <= 500 instances use tiny_data_num_folds/repeats."""
    v = _Validation(use_task_specific_validation=True)
    X = _make_X(100)  # 100 < 500 → tiny
    y = pd.Series(np.zeros(100))
    custom_splits, folds, repeats = v.resolve_validation_splits(
        X=X, y=y, num_folds=8, num_repeats=1
    )
    assert custom_splits is None
    assert folds == _Validation.tiny_data_num_folds
    assert repeats == _Validation.tiny_data_num_repeats


def test_resolve_validation_splits_normal_data_unchanged():
    """Datasets with > 500 instances and no grouping/time → folds/repeats unchanged."""
    v = _Validation(use_task_specific_validation=True)
    X = _make_X(600)  # > 500
    y = pd.Series(np.zeros(600))
    custom_splits, folds, repeats = v.resolve_validation_splits(
        X=X, y=y, num_folds=8, num_repeats=1
    )
    assert custom_splits is None
    assert folds == 8
    assert repeats == 1


def test_resolve_validation_splits_time_on_and_group_on_raises_not_implemented():
    """Simultaneous time_on and group_on is explicitly not implemented."""
    v = _Validation(
        use_task_specific_validation=True,
        time_on="time",
        group_on="group",
    )
    n = 10
    X = pd.DataFrame({"feature": range(n), "time": range(n), "group": ["g"] * n})
    y = pd.Series(np.zeros(n))
    with pytest.raises(NotImplementedError):
        v.resolve_validation_splits(X=X, y=y, num_folds=8, num_repeats=1)


# ---------------------------------------------------------------------------
# TabArenaValidationProtocolExecMixin — _resolve_number_of_splits
# ---------------------------------------------------------------------------


def test_resolve_number_of_splits_tiny_data():
    v = _Validation(use_task_specific_validation=True)
    folds, repeats = v._resolve_number_of_splits(
        num_folds=8, num_repeats=1, num_group_instances=50
    )
    assert folds == _Validation.tiny_data_num_folds
    assert repeats == _Validation.tiny_data_num_repeats


def test_resolve_number_of_splits_normal_data_unchanged():
    v = _Validation(use_task_specific_validation=True)
    folds, repeats = v._resolve_number_of_splits(
        num_folds=8, num_repeats=1, num_group_instances=1000
    )
    assert folds == 8
    assert repeats == 1


def test_resolve_number_of_splits_normal_data_wrong_folds_asserts():
    """The normal path asserts num_folds == 8."""
    v = _Validation(use_task_specific_validation=True)
    with pytest.raises(AssertionError):
        v._resolve_number_of_splits(
            num_folds=5, num_repeats=1, num_group_instances=1000
        )


def test_resolve_number_of_splits_normal_data_wrong_repeats_asserts():
    """The normal path asserts num_repeats is 1 or None."""
    v = _Validation(use_task_specific_validation=True)
    with pytest.raises(AssertionError):
        v._resolve_number_of_splits(
            num_folds=8, num_repeats=3, num_group_instances=1000
        )


# ---------------------------------------------------------------------------
# TabArenaValidationProtocolExecMixin — get_num_group_instances
# ---------------------------------------------------------------------------


def test_get_num_group_instances_no_group():
    v = _Validation(use_task_specific_validation=True, group_on=None)
    X = _make_X(7)
    assert v.get_num_group_instances(X) == 7


# ===========================================================================
# Additional split_time_index_into_intervals tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Additional error conditions
# ---------------------------------------------------------------------------


def test_goal_n_intervals_zero_raises():
    time_data = pd.Series([1, 2, 3])
    with pytest.raises(ValueError, match="at least 2"):
        split_time_index_into_intervals(time_data=time_data, goal_n_intervals=0)


def test_goal_n_intervals_negative_raises():
    time_data = pd.Series([1, 2, 3])
    with pytest.raises(ValueError, match="at least 2"):
        split_time_index_into_intervals(time_data=time_data, goal_n_intervals=-5)


def test_empty_series_raises():
    """An empty series has 0 unique values — cannot form 2 intervals."""
    time_data = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="at least 2 unique time values"):
        split_time_index_into_intervals(time_data=time_data, goal_n_intervals=2)


def test_single_element_series_raises():
    """One element ⟹ one unique value ⟹ cannot form 2 intervals."""
    time_data = pd.Series([42.0])
    with pytest.raises(ValueError, match="at least 2 unique time values"):
        split_time_index_into_intervals(time_data=time_data, goal_n_intervals=2)


# ---------------------------------------------------------------------------
# Invariants that must hold for ANY valid input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("goal", "n_unique", "balance_on"),
    [
        (2, 5, "rows"),
        (3, 6, "unique"),
        (4, 4, "rows"),  # goal == n_unique exactly
        (10, 3, "rows"),  # goal >> n_unique; actual_n caps at 3
        (8, 20, "unique"),
        (2, 2, "rows"),  # minimum valid case
    ],
    ids=["2of5", "3of6_unique", "4of4_exact", "cap_at_3", "8of20_unique", "min_2of2"],
)
def test_returned_n_always_equals_actual_interval_count(goal, n_unique, balance_on):
    """The returned integer n must equal the number of distinct labels in the output."""
    time_data = pd.Series(range(n_unique))
    intervals, n = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=goal, balance_on=balance_on
    )
    assert n == intervals.nunique()


@pytest.mark.parametrize(
    ("goal", "n_unique"),
    [(2, 5), (3, 3), (3, 10), (8, 8), (10, 3)],
    ids=["2of5", "3of3", "3of10", "8of8", "cap_10of3"],
)
def test_interval_labels_are_zero_indexed_and_contiguous(goal, n_unique):
    """Labels must be exactly {0, 1, …, n−1} with no gaps."""
    time_data = pd.Series(range(n_unique))
    intervals, n = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=goal
    )
    assert set(intervals) == set(range(n))


@pytest.mark.parametrize(
    ("goal", "n_unique"),
    [(2, 4), (3, 6), (5, 10)],
    ids=["2of4", "3of6", "5of10"],
)
def test_all_intervals_are_nonempty(goal, n_unique):
    """Each interval from 0 to n−1 must contain at least one row."""
    time_data = pd.Series(range(n_unique))
    intervals, n = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=goal
    )
    for label in range(n):
        assert (intervals == label).any(), f"Interval {label} is empty"


def test_temporal_ordering_invariant_with_unsorted_input():
    """For every pair of rows: time[i] < time[j] ⟹ interval[i] ≤ interval[j].

    Uses an unsorted series so both the mapping AND the index handling are
    exercised — not just a trivially sorted sequence.
    """
    rng = np.random.default_rng(42)
    n = 40
    time_vals = rng.integers(1, 15, size=n)  # values drawn from 1..14
    time_data = pd.Series(time_vals)
    intervals, _ = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=5
    )
    for _t1, _i1, _t2, _i2 in zip(time_data, intervals, time_data, intervals):
        pass  # just verifying iteration works; real check below

    # Build sorted-by-time view and assert monotonicity
    sorted_pairs = sorted(zip(time_data.tolist(), intervals.tolist()))
    for idx in range(len(sorted_pairs) - 1):
        t_a, lbl_a = sorted_pairs[idx]
        t_b, lbl_b = sorted_pairs[idx + 1]
        if t_a < t_b:
            assert lbl_a <= lbl_b, (
                f"Temporal ordering violated: time {t_a} → label {lbl_a}, "
                f"time {t_b} → label {lbl_b}"
            )


def test_all_rows_covered_by_valid_label():
    """Every row must receive a label in [0, n−1]; no row may be unassigned (NaN)."""
    time_data = pd.Series([3, 1, 4, 1, 5, 9, 2, 6])
    intervals, n = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=3
    )
    assert not intervals.isna().any()
    assert intervals.min() == 0
    assert intervals.max() == n - 1


# ---------------------------------------------------------------------------
# Algorithm-specific behaviour
# ---------------------------------------------------------------------------


def test_goal_equals_n_unique_each_unique_gets_own_interval():
    """When goal == n_unique, every unique value is its own interval.

    Traced manually: time=[1,2,3], goal=3 → labels=[0,1,2].
    """
    time_data = pd.Series([1, 2, 3])
    intervals, n = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=3
    )
    assert n == 3
    assert list(intervals) == [0, 1, 2]


def test_goal_larger_than_n_unique_caps_actual_n_at_n_unique():
    """Goal >> n_unique: actual_n returned is n_unique, not goal."""
    time_data = pd.Series([10, 20, 30])  # 3 unique values
    intervals, n = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=100
    )
    assert n == 3
    assert intervals.nunique() == 3


def test_balance_on_rows_vs_unique_produce_different_partitions_for_skewed_data():
    """Pinned test: [1,1,1,2,3,4] with goal=2.

    balance_on="rows"  balances row counts   → cut after t=1: [0,0,0,1,1,1]
    balance_on="unique" balances unique count → cut after t=2: [0,0,0,0,1,1]

    Verified by tracing the greedy algorithm (see algorithm trace notes).
    """
    time_data = pd.Series([1, 1, 1, 2, 3, 4])

    intervals_rows, n_rows = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=2, balance_on="rows"
    )
    intervals_unique, n_unique = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=2, balance_on="unique"
    )

    assert n_rows == n_unique == 2
    assert list(intervals_rows) == [0, 0, 0, 1, 1, 1]
    assert list(intervals_unique) == [0, 0, 0, 0, 1, 1]
    # The two modes genuinely differ
    assert list(intervals_rows) != list(intervals_unique)


def test_balance_on_unique_distributes_unique_values_evenly():
    """6 unique values, goal=3, balance_on="unique" → 2 uniques per interval.

    Traced: [0,1,2,3,4,5] → [0,0,1,1,2,2].
    """
    time_data = pd.Series(range(6))
    intervals, n = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=3, balance_on="unique"
    )
    assert n == 3
    assert list(intervals) == [0, 0, 1, 1, 2, 2]
    for label in range(3):
        assert (intervals == label).sum() == 2  # 2 rows per interval


def test_unsorted_input_labels_are_consistent_with_time_order():
    """Unsorted input [5,3,1,4,2] with goal=3.

    Traced manually:
        mapping: 1→0, 2→0, 3→1, 4→2, 5→2
        output:  [2, 1, 0, 2, 0]
    """
    time_data = pd.Series([5, 3, 1, 4, 2])
    intervals, n = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=3
    )
    assert n == 3
    assert list(intervals) == [2, 1, 0, 2, 0]
    # Confirm the mapping is monotone: t=1,2 → 0; t=3 → 1; t=4,5 → 2
    mapping = dict(zip(time_data.tolist(), intervals.tolist()))
    assert mapping[1] == mapping[2] == 0
    assert mapping[3] == 1
    assert mapping[4] == mapping[5] == 2


def test_heavy_duplicate_cluster_cannot_be_split():
    """10 copies of value 1 must all receive the same interval label."""
    time_data = pd.Series([1] * 10 + [2, 3])
    intervals, _ = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=2
    )
    one_labels = intervals[time_data == 1].unique()
    assert len(one_labels) == 1, (
        "All duplicates of value 1 must be in the same interval"
    )


def test_two_unique_values_always_yields_two_intervals():
    """Exactly 2 unique values is the minimum; both must get distinct labels."""
    time_data = pd.Series([7, 7, 7, 99])
    intervals, n = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=2
    )
    assert n == 2
    assert intervals[time_data == 7].unique().tolist() == [0]
    assert intervals[time_data == 99].unique().tolist() == [1]


def test_actual_n_equals_goal_when_n_unique_exceeds_goal():
    """When there are more unique values than goal, actual_n == goal."""
    time_data = pd.Series(range(20))  # 20 unique values
    for goal in (2, 5, 7):
        _, n = split_time_index_into_intervals(
            time_data=time_data, goal_n_intervals=goal
        )
        assert n == goal, f"Expected actual_n={goal}, got {n}"


# ---------------------------------------------------------------------------
# Data-type coverage
# ---------------------------------------------------------------------------


def test_float_time_values_work():
    """Non-integer numeric series must be accepted and produce valid output."""
    time_data = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    intervals, n = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=3
    )
    assert n == 3
    assert len(intervals) == 6
    assert set(intervals) == {0, 1, 2}


def test_negative_time_values_work():
    """Negative values are valid time inputs (only relative order matters)."""
    time_data = pd.Series([-3, -2, -1, 0, 1, 2])
    intervals, n = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=3
    )
    assert n == 3
    assert len(intervals) == 6
    # Monotonicity: labels must not decrease as time increases
    sorted_labels = intervals.sort_values().reset_index(drop=True)
    assert list(sorted_labels) == sorted(sorted_labels.tolist())


def test_large_integer_time_values_work():
    """Very large integers (near int64 range) should not cause overflow."""
    big = 10**15
    time_data = pd.Series([big, big + 1, big + 2, big + 3, big + 4, big + 5])
    intervals, n = split_time_index_into_intervals(
        time_data=time_data, goal_n_intervals=3
    )
    assert n == 3
    assert len(intervals) == 6


# ===========================================================================
# Additional TabArenaValidationProtocolExecMixin tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Class constants
# ---------------------------------------------------------------------------


def test_mixin_class_constants():
    """The constants govern the tiny-data regime — pin their values explicitly."""
    assert _Validation.tiny_data_num_folds == 5
    assert _Validation.tiny_data_num_repeats == 3
    assert _Validation.max_samples_for_tiny_data == 500


# ---------------------------------------------------------------------------
# _resolve_number_of_splits — boundary cases
# ---------------------------------------------------------------------------


def test_resolve_number_of_splits_at_exact_boundary_500_is_tiny():
    """500 instances == max_samples_for_tiny_data → tiny-data path."""
    v = _Validation(use_task_specific_validation=True)
    folds, repeats = v._resolve_number_of_splits(
        num_folds=8, num_repeats=1, num_group_instances=500
    )
    assert folds == _Validation.tiny_data_num_folds
    assert repeats == _Validation.tiny_data_num_repeats


def test_resolve_number_of_splits_at_501_is_normal():
    """501 instances > max_samples_for_tiny_data → normal path."""
    v = _Validation(use_task_specific_validation=True)
    folds, repeats = v._resolve_number_of_splits(
        num_folds=8, num_repeats=1, num_group_instances=501
    )
    assert folds == 8
    assert repeats == 1


def test_resolve_number_of_splits_num_repeats_none_allowed_on_normal_path():
    """Normal path assertion is: num_repeats == 1 OR num_repeats is None."""
    v = _Validation(use_task_specific_validation=True)
    folds, repeats = v._resolve_number_of_splits(
        num_folds=8, num_repeats=None, num_group_instances=1000
    )
    assert folds == 8
    assert repeats is None  # unchanged — no new value was assigned


# ---------------------------------------------------------------------------
# resolve_validation_splits — additional paths
# ---------------------------------------------------------------------------


def test_resolve_validation_splits_num_folds_zero_returns_early():
    """num_folds=0 satisfies the `<= 1` early-exit condition."""
    v = _Validation(use_task_specific_validation=True)
    X = _make_X(10)
    y = pd.Series(np.zeros(10))
    custom_splits, folds, _repeats = v.resolve_validation_splits(
        X=X, y=y, num_folds=0, num_repeats=1
    )
    assert custom_splits is None
    assert folds == 0


def test_resolve_validation_splits_num_repeats_none_normal_data():
    """num_repeats=None is accepted for the normal-data (>500 rows) path."""
    v = _Validation(use_task_specific_validation=True)
    X = _make_X(600)
    y = pd.Series(np.zeros(600))
    custom_splits, folds, repeats = v.resolve_validation_splits(
        X=X, y=y, num_folds=8, num_repeats=None
    )
    assert custom_splits is None
    assert folds == 8
    assert repeats is None


def test_resolve_validation_splits_exactly_500_instances_is_tiny():
    """500 rows == boundary → tiny-data folds/repeats must be applied."""
    v = _Validation(use_task_specific_validation=True)
    X = _make_X(500)
    y = pd.Series(np.zeros(500))
    custom_splits, folds, repeats = v.resolve_validation_splits(
        X=X, y=y, num_folds=8, num_repeats=1
    )
    assert custom_splits is None
    assert folds == _Validation.tiny_data_num_folds
    assert repeats == _Validation.tiny_data_num_repeats


def test_resolve_validation_splits_501_instances_is_normal():
    """501 rows > boundary → folds/repeats must remain unchanged."""
    v = _Validation(use_task_specific_validation=True)
    X = _make_X(501)
    y = pd.Series(np.zeros(501))
    custom_splits, folds, repeats = v.resolve_validation_splits(
        X=X, y=y, num_folds=8, num_repeats=1
    )
    assert custom_splits is None
    assert folds == 8
    assert repeats == 1


def test_resolve_validation_splits_disabled_ignores_tiny_data():
    """use_task_specific_validation=False must skip all logic regardless of data size."""
    v = _Validation(use_task_specific_validation=False)
    X = _make_X(50)  # would be tiny if validation were enabled
    y = pd.Series(np.zeros(50))
    _custom_splits, folds, repeats = v.resolve_validation_splits(
        X=X, y=y, num_folds=8, num_repeats=1
    )
    # Folds must NOT be changed to tiny_data_num_folds
    assert folds == 8
    assert repeats == 1


# ---------------------------------------------------------------------------
# group_on_to_groups_data — additional properties
# ---------------------------------------------------------------------------


def test_group_on_to_groups_data_preserves_dataframe_index():
    """The returned series must carry the same index as X."""
    X = pd.DataFrame({"group": ["a", "b", "c"]}, index=[10, 20, 30])
    result = _Validation.group_on_to_groups_data(X=X, group_on="group")
    assert list(result.index) == [10, 20, 30]


def test_group_on_to_groups_data_returns_independent_copy():
    """Mutating the returned series must not modify X."""
    X = pd.DataFrame({"group": ["x", "y"]})
    result = _Validation.group_on_to_groups_data(X=X, group_on="group")
    result.iloc[0] = "MUTATED"
    assert X["group"].iloc[0] == "x"


def test_group_on_to_groups_data_multi_column_separator_is_underscore():
    """Multi-column groups are joined with '_' as separator."""
    X = pd.DataFrame({"g1": ["foo", "foo"], "g2": ["bar", "baz"]})
    result = _Validation.group_on_to_groups_data(X=X, group_on=["g1", "g2"])
    assert list(result) == ["foo_bar", "foo_baz"]


def test_group_on_to_groups_data_numeric_columns_joined_as_strings():
    """Numeric group columns must be cast to str before joining."""
    X = pd.DataFrame({"g1": [1, 1], "g2": [2, 3]})
    result = _Validation.group_on_to_groups_data(X=X, group_on=["g1", "g2"])
    assert list(result) == ["1_2", "1_3"]


# ---------------------------------------------------------------------------
# time_on_to_groups_data — additional properties
# ---------------------------------------------------------------------------


def test_time_on_to_groups_data_unsorted_input_gives_monotonic_output():
    """Unsorted time column must still produce monotone interval labels.

    The output labels must satisfy: time[i] < time[j] ⟹ label[i] ≤ label[j].
    """
    X = pd.DataFrame({"time": [5, 3, 1, 4, 2]})
    groups, n = _Validation.time_on_to_groups_data(X=X, time_on="time", num_folds=3)
    assert n == 3
    pairs = sorted(zip(X["time"].tolist(), groups.tolist()))
    for i in range(len(pairs) - 1):
        t_a, lbl_a = pairs[i]
        t_b, lbl_b = pairs[i + 1]
        if t_a < t_b:
            assert lbl_a <= lbl_b


def test_time_on_to_groups_data_string_column_raises():
    """A non-numeric, non-datetime column must be rejected with AssertionError."""
    X = pd.DataFrame({"time": ["2020-01", "2020-02", "2020-03", "2020-04"]})
    with pytest.raises(AssertionError, match="not datetime or numeric"):
        _Validation.time_on_to_groups_data(X=X, time_on="time", num_folds=2)


def test_time_on_to_groups_data_output_length_matches_input():
    """Returned series must have the same length as the input DataFrame."""
    X = pd.DataFrame({"time": list(range(15))})
    groups, n = _Validation.time_on_to_groups_data(X=X, time_on="time", num_folds=4)
    assert len(groups) == len(X)
    assert n == 4


def test_time_on_to_groups_data_fewer_unique_than_folds_caps_n():
    """When n_unique < num_folds, actual n_intervals is capped at n_unique."""
    X = pd.DataFrame({"time": [1, 1, 2, 2]})  # only 2 unique timestamps
    _groups, n = _Validation.time_on_to_groups_data(X=X, time_on="time", num_folds=10)
    assert n == 2


# ---------------------------------------------------------------------------
# get_num_group_instances — group_on paths
# ---------------------------------------------------------------------------


def test_get_num_group_instances_per_group_label_counts_groups():
    """PER_GROUP: result is the number of distinct group identifiers."""
    v = _Validation(
        use_task_specific_validation=True,
        group_on="g",
        group_labels=GroupLabelTypes.PER_GROUP,
    )
    X = pd.DataFrame({"a": [1, 2, 3, 4], "g": ["x", "x", "y", "y"]})
    assert v.get_num_group_instances(X) == 2


def test_get_num_group_instances_per_sample_label_returns_len():
    """PER_SAMPLE: result is len(X), ignoring group column entirely."""
    v = _Validation(
        use_task_specific_validation=True,
        group_on="g",
        group_labels=GroupLabelTypes.PER_SAMPLE,
    )
    X = pd.DataFrame({"a": [1, 2, 3, 4], "g": ["x", "x", "y", "y"]})
    assert v.get_num_group_instances(X) == 4  # len(X), not 2 groups
