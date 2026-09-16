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


def test_rank_column_is_lazy_and_matches_eager():
    zsc = ZeroshotSimulatorContext(df_configs=_configs(), df_baselines=_baselines(), folds=None)
    assert zsc._df_configs_ranked is None
    ranked = zsc.df_configs_ranked
    assert zsc._df_configs_ranked is ranked and zsc.df_configs_ranked is ranked
    expected = zsc.df_configs.copy()
    expected["rank"] = [
        zsc.rank_scorer.rank(t, e) for t, e in zip(expected["task"], expected["metric_error"], strict=True)
    ]
    pd.testing.assert_frame_equal(ranked, expected)

    # subset_datasets keeps the scorer: the remaining rows keep their ranks
    zsc.subset_datasets(["dsA"])
    assert zsc._df_configs_ranked is None
    after = zsc.df_configs_ranked
    pd.testing.assert_frame_equal(after, expected[expected["dataset"] == "dsA"])

    # subset_configs rebuilds the scorer over the remaining configs, as _update_all always did
    zsc2 = ZeroshotSimulatorContext(
        df_configs=_configs(), df_baselines=_baselines(), folds=None, score_against_only_baselines=False
    )
    zsc2.subset_configs(["cfg_1", "cfg_2"])
    rebuilt = zsc2.df_configs_ranked
    fresh = ZeroshotSimulatorContext(
        df_configs=_configs()[_configs()["framework"].isin(["cfg_1", "cfg_2"])],
        df_baselines=_baselines(),
        folds=None,
        score_against_only_baselines=False,
    ).df_configs_ranked
    np.testing.assert_array_equal(rebuilt["rank"].to_numpy(), fresh["rank"].to_numpy())


def test_context_pickles_string_columns_as_categoricals_and_restores_them():
    import pickle

    df_configs = _configs()
    df_configs["note"] = [None if i % 4 else "x" for i in range(len(df_configs))]  # string column with missing values
    df_configs["blob"] = [{"k": i} for i in range(len(df_configs))]  # non-string objects stay untouched
    zsc = ZeroshotSimulatorContext(df_configs=df_configs, df_baselines=_baselines(), folds=None)
    _ = zsc.df_configs_ranked

    state = zsc.__getstate__()
    assert set(state["_pickled_categorical_columns"]) == {"df_configs", "df_baselines", "_df_configs_ranked"}
    pickled_configs = state["df_configs"]
    assert isinstance(pickled_configs["dataset"].dtype, pd.CategoricalDtype)
    assert isinstance(pickled_configs["note"].dtype, pd.CategoricalDtype)
    assert pickled_configs["blob"].dtype == object
    assert zsc.df_configs["dataset"].dtype == object  # the live object is untouched

    restored = pickle.loads(pickle.dumps(zsc, protocol=5))
    for attr in ("df_configs", "df_baselines", "_df_configs_ranked", "df_metrics"):
        a, b = getattr(zsc, attr), getattr(restored, attr)
        pd.testing.assert_frame_equal(a, b)
        assert list(a.dtypes) == list(b.dtypes)
    assert restored.unique_tasks == zsc.unique_tasks
    assert np.array_equal(restored.rank_scorer.error_dict["11_0"], zsc.rank_scorer.error_dict["11_0"])


def test_context_pickles_int_columns_narrow_and_restores_them():
    """Integer result columns travel in the narrowest dtype that holds them and come back as
    they were; a column that needs int64 stays int64 on the wire.
    """
    import pickle

    import numpy as np

    from tabarena.simulation.context_artificial import load_repo_artificial

    context = load_repo_artificial()._zeroshot_context
    df = context.df_configs
    df["wide_int"] = np.int64(2**40) + np.arange(len(df), dtype=np.int64)
    df["already_narrow"] = np.arange(len(df), dtype=np.int8)
    int_columns = [c for c in df.columns if df[c].dtype.kind == "i"]
    assert "fold" in int_columns
    before = {c: str(df[c].dtype) for c in df.columns}

    state = context.__getstate__()
    narrowed = state["_pickled_narrowed_columns"]["df_configs"]
    assert narrowed["fold"] == "int64"
    assert "wide_int" not in narrowed
    assert "already_narrow" not in narrowed
    assert str(state["df_configs"]["fold"].dtype) == "int8"
    assert str(state["df_configs"]["wide_int"].dtype) == "int64"
    # the live frame is untouched
    assert {c: str(df[c].dtype) for c in df.columns} == before

    restored = pickle.loads(pickle.dumps(context, protocol=5))
    pd.testing.assert_frame_equal(restored.df_configs, df)
    assert {c: str(restored.df_configs[c].dtype) for c in df.columns} == before
    assert "_pickled_narrowed_columns" not in restored.__dict__
    assert len(pickle.dumps(context, protocol=5)) < len(
        pickle.dumps(state["df_configs"].astype({"fold": "int64"}), protocol=5)
    ) + len(pickle.dumps({k: v for k, v in state.items() if k != "df_configs"}, protocol=5))
