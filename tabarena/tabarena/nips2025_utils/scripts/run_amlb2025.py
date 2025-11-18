from __future__ import annotations

import pandas as pd

from autogluon.tabular import TabularDataset
from bencheval.tabarena import TabArena


if __name__ == '__main__':
    amlb2025_prefix = "s3://neerick-autogluon/tabarena/benchmarks/amlb2025/"

    results_files = [
        "amlb_all.csv"
    ]

    results_files = [f"{amlb2025_prefix}{result_file}" for result_file in results_files]

    results_dfs = [
        TabularDataset(result_file) for result_file in results_files
    ]

    df_results = pd.concat(results_dfs, ignore_index=True)

    df_results = df_results[~df_results["result"].isnull()]
    df_results["fold"] = df_results["fold"].astype(int)
    df_results["metric_error"] = -df_results["result"]
    optimum_result = df_results["metric"].map({
        "auc": 1.0,
        "neg_rmse": 0.0,
        "neg_logloss": 0.0,
    })
    df_results["metric_error"] = optimum_result - df_results["result"]
    df_results["metric"] = df_results["metric"].map({
        "auc": "roc_auc",
        "neg_rmse": "rmse",
        "neg_logloss": "logloss",
    })
    df_results["training_duration"] = df_results["training_duration"].fillna(df_results["duration"])
    df_results = df_results[~df_results["predict_duration"].isnull()]

    fillna_method = f"constantpredictor_60min"
    baseline_method = "RandomForest_60min"

    default_methods = [
        "constantpredictor_60min",
        "RandomForest_60min",
        "TunedRandomForest_60min",
    ]

    method_suffix = "_60min"

    banned_methods = [
        f"FEDOT{method_suffix}",
        f"NaiveAutoML{method_suffix}",
        f"autosklearn2{method_suffix}",
    ]

    df_results = df_results[(df_results["framework"].str.contains(method_suffix)) | (df_results["framework"].isin(default_methods))]
    df_results = df_results[~df_results["framework"].isin(banned_methods)]
    df_results = df_results[df_results["framework"] != fillna_method]

    df_results = df_results.rename(columns={
        "framework": "method",
        "training_duration": "time_train_s",
        "predict_duration": "time_infer_s",
    })

    arena = TabArena(
        method_col="method",
        task_col="task",
        error_col="metric_error",
        columns_to_agg_extra=[
            "time_train_s",
            "time_infer_s",
        ],
        seed_column="fold",
    )

    # df_fillna = df_results[df_results[arena.method_col] == fillna_method]
    # df_fillna = df_fillna.drop(columns=[arena.method_col])
    df_fillna = None
    df_results_fillna = arena.fillna_data(data=df_results, df_fillna=df_fillna, fillna_method="worst")

    leaderboard = arena.leaderboard(
        data=df_results_fillna,
        # include_error=True,
        include_elo=True,
        # include_failure_counts=True,
        include_mrr=True,
        include_rank_counts=True,
        include_winrate=True,
        elo_kwargs={
            "calibration_framework": baseline_method,
            "calibration_elo": 1000,
            "BOOTSTRAP_ROUNDS": 100,
        },
        baseline_method=baseline_method,
        relative_error_kwargs={"agg": "gmean"},
    )
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard)
