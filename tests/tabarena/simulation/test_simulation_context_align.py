from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabarena.simulation.sim_utils import has_duplicate_keys
from tabarena.simulation.simulation_context import ZeroshotSimulatorContext
from tabarena.utils.rank_utils import RankScorer


def _frame(rows):
    df = pd.DataFrame(rows)
    df["metric_error_val"] = df["metric_error"]
    df["time_train_s"] = 1.0
    df["time_infer_s"] = 0.1
    return df


def _configs():
    rows = []
    for dataset, tid, problem_type, metric in (("dsB", 11, "binary", "log_loss"), ("dsA", 7, "regression", "rmse")):
        for fold in (1, 0):
            for i, cfg in enumerate(("cfg_1", "cfg_2", "cfg_3")):
                rows.append(
                    {
                        "dataset": dataset,
                        "tid": tid,
                        "fold": fold,
                        "framework": cfg,
                        "problem_type": problem_type,
                        "metric": metric,
                        "metric_error": 0.1 * (i + 1) + fold,
                    }
                )
    return _frame(rows)


def _baselines():
    rows = []
    for dataset, tid, problem_type, metric in (("dsB", 11, "binary", "log_loss"), ("dsA", 7, "regression", "rmse")):
        for fold in (0, 1):
            for cfg, err in (("AutoML_x", 0.05 + fold), ("cfg_1", 0.5 + fold)):  # cfg_1 also among the configs
                rows.append(
                    {
                        "dataset": dataset,
                        "tid": tid,
                        "fold": fold,
                        "framework": cfg,
                        "problem_type": problem_type,
                        "metric": metric,
                        "metric_error": err,
                    }
                )
    return _frame(rows)


def test_has_duplicate_keys():
    assert not has_duplicate_keys(np.array([0, 1, 2]), n_range=3)
    assert has_duplicate_keys(np.array([0, 1, 1]), n_range=3)
    assert not has_duplicate_keys(np.array([], dtype=np.int64), n_range=10)
    # sparse key range takes the hash-based path
    assert has_duplicate_keys(np.array([5, 10**6, 5]), n_range=10**7)
    assert not has_duplicate_keys(np.array([5, 10**6]), n_range=10**7)


def test_align_derives_task_structures_and_ranks():
    zsc = ZeroshotSimulatorContext(df_configs=_configs(), df_baselines=_baselines(), folds=None)
    assert zsc.unique_tasks == ["11_0", "11_1", "7_0", "7_1"]
    assert zsc.unique_datasets == ["dsA", "dsB"]
    assert zsc.task_to_dataset_dict == {"11_0": "dsB", "11_1": "dsB", "7_0": "dsA", "7_1": "dsA"}
    assert zsc.dataset_to_tid_dict == {"dsA": 7, "dsB": 11}
    assert zsc.task_to_fold_dict == {"11_0": 0, "11_1": 1, "7_0": 0, "7_1": 1}
    assert zsc.dataset_to_folds_dict == {"dsA": [0, 1], "dsB": [0, 1]}
    assert zsc.dataset_to_problem_type_dict == {"dsB": "binary", "dsA": "regression"}
    assert list(zsc.df_metrics.index) == ["dsB", "dsA"]  # first-occurrence order
    expected_tasks = [f"{tid}_{fold}" for tid, fold in zip(zsc.df_configs["tid"], zsc.df_configs["fold"], strict=True)]
    assert list(zsc.df_configs["task"]) == expected_tasks
    assert zsc.df_configs.index.equals(pd.RangeIndex(len(zsc.df_configs)))
    # with baselines present the default scores against the baselines only
    assert zsc.rank_scorer.error_dict["11_0"] == pytest.approx([0.05, 0.5])
    expected = [
        zsc.rank_scorer.rank(t, e) for t, e in zip(zsc.df_configs["task"], zsc.df_configs["metric_error"], strict=True)
    ]
    np.testing.assert_array_equal(zsc.df_configs_ranked["rank"].to_numpy(), expected)

    # scoring against configs and baselines: cfg_1 appears in both for the same task and the
    # scorer averages them (pivot_table semantics)
    zsc_all = ZeroshotSimulatorContext(
        df_configs=_configs(), df_baselines=_baselines(), folds=None, score_against_only_baselines=False
    )
    assert zsc_all.rank_scorer.error_dict["11_0"] == pytest.approx(sorted([0.3, 0.2, 0.3, 0.05]))


def test_align_fold_restriction_and_duplicate_detection():
    zsc = ZeroshotSimulatorContext(df_configs=_configs(), df_baselines=_baselines(), folds=[1])
    assert zsc.unique_tasks == ["11_1", "7_1"]
    assert set(zsc.df_configs["fold"]) == {1} and set(zsc.df_baselines["fold"]) == {1}
    assert zsc.dataset_to_folds_dict == {"dsA": [1], "dsB": [1]}

    duplicated = pd.concat([_configs(), _configs().iloc[:1]], ignore_index=True)
    with pytest.raises(AssertionError, match="Multiple rows in `df_configs` exist for a config task pair"):
        ZeroshotSimulatorContext(df_configs=duplicated, df_baselines=_baselines(), folds=None)


def test_rank_scorer_unique_and_duplicate_results_agree_with_pivot():
    df = _configs()
    df["task"] = df["tid"].astype(str) + "_" + df["fold"].astype(str)
    df.loc[0, "metric_error"] = np.nan
    tasks = sorted(df["task"].unique())
    fast = RankScorer(df_results=df, tasks=tasks)
    pivot = df.pivot_table(values="metric_error", index="task", columns="framework")
    for task in tasks:
        row = pivot.loc[task].to_numpy(dtype=float)
        assert list(fast.error_dict[task]) == sorted(row[~np.isnan(row)].tolist())
    doubled = pd.concat([df, df.assign(metric_error=df["metric_error"] + 1)], ignore_index=True)
    averaged = RankScorer(df_results=doubled, tasks=tasks)
    pivot2 = doubled.pivot_table(values="metric_error", index="task", columns="framework")
    for task in tasks:
        row = pivot2.loc[task].to_numpy(dtype=float)
        assert list(averaged.error_dict[task]) == sorted(row[~np.isnan(row)].tolist())
