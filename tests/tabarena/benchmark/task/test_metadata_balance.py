from __future__ import annotations

import pandas as pd
import pytest

from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.benchmark.task.metadata.balance import (
    EXTREME_IMBALANCE_RATIO_THRESHOLD,
    IMBALANCE_RATIO_THRESHOLD,
    TARGET_SKEWNESS_THRESHOLD,
    is_target_extremely_imbalanced,
    is_target_imbalanced,
    target_distribution_stats,
)


def test_classification_stats_are_the_class_ratio():
    minority, ratio, skew = target_distribution_stats(
        pd.Series(["a"] * 30 + ["b"] * 10 + [None]), is_classification=True
    )
    assert minority == pytest.approx(0.25)
    assert ratio == pytest.approx(3.0)
    assert skew is None


def test_regression_stats_are_the_skewness():
    from scipy import stats

    values = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    minority, ratio, skew = target_distribution_stats(values, is_classification=False)
    assert minority is None and ratio is None
    assert skew == pytest.approx(stats.skew(values.to_numpy()))
    # A constant or single-class target defines neither statistic.
    assert target_distribution_stats(pd.Series([2.0] * 5), is_classification=False) == (None, None, None)
    assert target_distribution_stats(pd.Series(["a"] * 5), is_classification=True) == (None, None, None)


@pytest.mark.parametrize(
    ("problem_type", "ratio", "skew", "expected"),
    [
        ("binary", IMBALANCE_RATIO_THRESHOLD, None, True),
        ("binary", IMBALANCE_RATIO_THRESHOLD - 0.01, None, False),
        ("multiclass", 10.0, None, True),
        ("regression", None, -TARGET_SKEWNESS_THRESHOLD, True),
        ("regression", None, 0.3, False),
        ("regression", None, None, None),
        ("binary", None, None, None),
        ("binary", float("nan"), None, None),
    ],
)
def test_is_target_imbalanced(problem_type, ratio, skew, expected):
    assert is_target_imbalanced(problem_type=problem_type, imbalance_ratio=ratio, skewness=skew) is expected


def test_extreme_is_the_far_end_of_imbalanced():
    assert EXTREME_IMBALANCE_RATIO_THRESHOLD > IMBALANCE_RATIO_THRESHOLD
    assert (
        is_target_extremely_imbalanced(problem_type="binary", imbalance_ratio=EXTREME_IMBALANCE_RATIO_THRESHOLD) is True
    )
    assert is_target_extremely_imbalanced(problem_type="binary", imbalance_ratio=IMBALANCE_RATIO_THRESHOLD) is False
    assert is_target_extremely_imbalanced(problem_type="regression", imbalance_ratio=None) is None
    assert is_target_extremely_imbalanced(problem_type="binary", imbalance_ratio=None) is None


def test_tabarena_v0_1_carries_the_verdict_on_the_grid_and_per_dataset():
    """The committed v0.1 metadata carries the target statistics, so every dataset gets a
    verdict and the two subsets partition the suite.
    """
    collection = TaskMetadataCollection.from_preset("TabArena-v0.1")
    grid = collection.task_grid()
    assert grid["target_imbalanced"].notna().all()
    per_dataset = collection.per_dataset_frame()
    verdict = per_dataset.set_index("dataset")["target_imbalanced"]
    assert verdict.notna().all()
    assert set(verdict.unique()) == {True, False}
    # Known anchors: a 2% minority share and a 1:1 split, plus a heavily skewed price target.
    assert verdict["APSFailure"] is True
    assert verdict["hazelnut-spread-contaminant-detection"] is False
    assert verdict["miami_housing"] is True
    assert verdict["airfoil_self_noise"] is False
    balanced = collection.subset_tasks(subset=["balanced"]).dataset_names()
    imbalanced = collection.subset_tasks(subset=["imbalanced"]).dataset_names()
    assert set(balanced).isdisjoint(imbalanced)
    assert sorted([*balanced, *imbalanced]) == sorted(collection.dataset_names())
    # Every extreme dataset is an imbalanced one; a 54:1 binary task is in, a 10:1 one is not.
    extreme = collection.subset_tasks(subset=["extreme"]).dataset_names()
    assert set(extreme) <= set(imbalanced)
    assert "APSFailure" in extreme
    assert "Diabetes130US" in imbalanced and "Diabetes130US" not in extreme
