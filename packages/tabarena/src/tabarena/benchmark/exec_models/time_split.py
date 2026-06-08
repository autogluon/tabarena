from __future__ import annotations

import numpy as np
import pandas as pd


def split_time_index_into_intervals(
    *,
    time_data: pd.Series,
    goal_n_intervals: int,
    balance_on: str = "rows",
) -> tuple[pd.Series, int]:
    """Split a monotonically ordered time index into contiguous dynamic intervals.

    Rules:
    - Larger time values are always later in time
    - Equal time values are always assigned to the same interval
    - Intervals are created from the observed data, not equal-width spacing
    - Tries to create `goal_n_intervals`, but falls back to a smaller number if needed
    - Never returns fewer than 2 intervals

    Parameters
    ----------
    time_data : pd.Series
        Time input
    goal_n_intervals : int
        Desired number of intervals.
    balance_on : {"rows", "unique"}, default "rows"
        - "rows": balance intervals by number of rows
        - "unique": balance intervals by number of unique time values

    Returns:
    -------
    time_intervals: pd.Series
        Interval label for each row in the input time_data.
    actual_n_intervals : int
        Number of intervals actually used.
    """
    if goal_n_intervals < 2:
        raise ValueError("n_intervals must be at least 2.")
    if balance_on not in {"rows", "unique"}:
        raise ValueError("balance_on must be either 'rows' or 'unique'.")

    assert not time_data.isna().any(), "Time column contains nan values!"

    s = time_data.copy()

    # Aggregate identical time values so duplicates stay together
    counts = s.value_counts(dropna=False).sort_index().rename("row_count").to_frame()
    counts["unique_weight"] = 1

    n_unique = len(counts)
    if n_unique < 2:
        raise ValueError("Need at least 2 unique time values to create at least 2 intervals.")
    actual_n_intervals = min(goal_n_intervals, n_unique)
    if actual_n_intervals < 2:
        raise ValueError("Could not create at least 2 intervals.")

    weight_col = "row_count" if balance_on == "rows" else "unique_weight"
    weights = counts[weight_col].to_numpy()

    # Greedy partition of sorted unique values into contiguous groups
    # aiming for equal cumulative weight per interval.
    total_weight = weights.sum()
    cut_positions = []
    start = 0
    cumulative = np.cumsum(weights)

    for group_num in range(1, actual_n_intervals):
        target = group_num * total_weight / actual_n_intervals

        # Candidate cut indices are between unique values:
        # cut after index j means next group starts at j+1
        min_j = start
        max_j = n_unique - (actual_n_intervals - group_num) - 1

        # Choose cut whose cumulative weight is closest to target
        candidates = np.arange(min_j, max_j + 1)
        j = candidates[np.argmin(np.abs(cumulative[candidates] - target))]
        cut_positions.append(j)
        start = j + 1

    # Assign interval labels to each unique time value
    interval_labels_for_unique = np.empty(n_unique, dtype=int)
    prev = 0
    for interval_id, cut in enumerate(cut_positions):
        interval_labels_for_unique[prev : cut + 1] = interval_id
        prev = cut + 1
    interval_labels_for_unique[prev:] = len(cut_positions)

    mapping = pd.Series(interval_labels_for_unique, index=counts.index)

    time_intervals = time_data.map(mapping)

    return time_intervals, actual_n_intervals
