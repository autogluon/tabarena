from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabarena.evaluation.split_allocation import (
    bootstrap_disagreement,
    cost_weighted_path,
    full_grid_methods,
    lite_allocation,
    per_split_ranks,
    runtime_s,
    split_noise,
    stability_rule_counts,
)


def _results(errors: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-split results from ``{dataset: array (n_splits, n_methods)}``."""
    rows = []
    for d, e in errors.items():
        for fold in range(e.shape[0]):
            for m in range(e.shape[1]):
                rows.append({"dataset": d, "fold": fold, "method": f"m{m}", "metric_error": float(e[fold, m])})
    return pd.DataFrame(rows)


def test_per_split_ranks_average_ties():
    ranks = per_split_ranks(_results({"a": np.array([[0.1, 0.1, 0.3]])}))
    assert ranks["rank"].tolist() == [1.5, 1.5, 3.0]


def test_cost_weighted_path_reaches_targets_within_bounds_and_prefers_cheap_noisy_datasets():
    cost = {"cheap": [1.0] * 10, "expensive": [100.0] * 10, "single": [5.0]}
    sigma = {"cheap": 1.0, "expensive": 1.0, "single": 1.0}
    start = lite_allocation({d: list(range(len(c))) for d, c in cost.items()})
    path = cost_weighted_path(start, cost, sigma, targets=[3, 6, 21, 50])
    assert path[3] == start
    assert sum(path[6].values()) == 6
    assert path[6]["cheap"] == 4 and path[6]["expensive"] == 1
    assert path[21] == {"cheap": 10, "expensive": 10, "single": 1}
    assert path[50] == path[21]


def test_cost_weighted_path_gives_noisier_datasets_more_splits_at_equal_cost():
    cost = {"noisy": [1.0] * 20, "quiet": [1.0] * 20}
    path = cost_weighted_path({"noisy": 1, "quiet": 1}, cost, {"noisy": 2.0, "quiet": 1.0}, targets=[12])
    assert path[12]["noisy"] > path[12]["quiet"]


def test_runtime_counts_the_first_splits():
    assert runtime_s({"a": 2, "b": 1}, {"a": [1.0, 2.0, 4.0], "b": [10.0, 20.0]}) == 13.0


def test_split_noise_fills_single_split_datasets_with_the_median():
    rng = np.random.default_rng(0)
    ranks = per_split_ranks(_results({"a": rng.random((5, 4)), "b": rng.random((5, 4)), "one": rng.random((1, 4))}))
    splits = {"a": list(range(5)), "b": list(range(5)), "one": [0]}
    sigma = split_noise(ranks, splits)
    assert sigma["one"] == pytest.approx(np.median([sigma["a"], sigma["b"]]))


def test_bootstrap_disagreement_is_zero_for_identical_splits_and_nan_for_one_split():
    same = np.tile(np.array([0.1, 0.2, 0.3, 0.4]), (6, 1))
    ranks = per_split_ranks(_results({"same": same, "one": same[:1]}))
    dis = bootstrap_disagreement(ranks, {"same": list(range(6)), "one": [0]}, {"same": 2, "one": 1}, pairs=20)
    assert dis["same"] == 0.0
    assert np.isnan(dis["one"])


def test_stability_rule_counts_one_split_when_splits_agree_and_caps_at_available():
    rng = np.random.default_rng(0)
    agree = np.tile(np.arange(6, dtype=float), (4, 1))
    noise = rng.random((4, 6))
    ranks = per_split_ranks(_results({"agree": agree, "noise": noise}))
    counts = stability_rule_counts(ranks, {"agree": list(range(4)), "noise": list(range(4))})
    assert counts["agree"] == 1
    assert 1 <= counts["noise"] <= 4


def test_cost_weighted_path_gives_a_dataset_without_splits_its_first_split_first():
    cost = {"none": [1000.0] * 3, "cheap": [1.0] * 10}
    path = cost_weighted_path({"none": 0, "cheap": 1}, cost, {"none": 1.0, "cheap": 5.0}, targets=[2])
    assert path[2] == {"none": 1, "cheap": 1}


def test_full_grid_methods_keeps_the_methods_that_cover_the_grid():
    splits = {"a": [0, 1, 2, 3], "b": [0, 1, 2, 3, 4, 5]}
    everywhere = [(d, s) for d, ss in splits.items() for s in ss]
    ran_on = {"full": everywhere, "almost": everywhere[1:], "core_only": [("a", 0), ("b", 0)]}

    class _Repo:
        def __init__(self, tasks):
            self._tasks = tasks

        def metrics(self):
            index = pd.MultiIndex.from_tuples(self._tasks, names=["dataset", "fold"])
            return pd.DataFrame({"metric_error": 0.0}, index=index)

    class _Context:
        methods = list(ran_on)

        def load_repo(self, methods, config_fallback=None):
            return _Repo(ran_on[methods[0]])

    assert full_grid_methods(_Context(), splits, min_coverage=0.9) == ["full", "almost"]
