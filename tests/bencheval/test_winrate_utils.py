from __future__ import annotations

import numpy as np
import pandas as pd

from bencheval.winrate_utils import compute_winrate_matrix


def _results(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    methods = ["A (default)", "A (tuned)", "A (T+E)", "B (default)", "B (T+E)", "System"]
    tasks = [f"task_{i}" for i in range(7)]
    return pd.DataFrame(
        [{"task": task, "method": method, "error": float(rng.random())} for task in tasks for method in methods]
    )


def test_a_subset_of_methods_gives_the_submatrix():
    """Win rates are pairwise and per task, so who else is in the matrix cannot change them.

    The reporter leans on this: it computes one matrix over every tuning variant for the
    interactive page and takes the best-variant submatrix for the static figure and the
    CSV, instead of computing the smaller matrix a second time.
    """
    results = _results()
    subset = ["A (T+E)", "B (T+E)", "System"]

    full = compute_winrate_matrix(results)
    direct = compute_winrate_matrix(results[results["method"].isin(subset)])

    pd.testing.assert_frame_equal(
        full.loc[direct.index, direct.columns],
        direct,
        check_names=False,
    )


def test_rows_are_ordered_by_mean_win_rate():
    """Which also means the order is relative to the field, so a submatrix has to re-sort."""
    matrix = compute_winrate_matrix(_results())
    means = matrix.mean(axis=1).to_numpy()
    assert np.all(np.diff(means) <= 1e-9)
