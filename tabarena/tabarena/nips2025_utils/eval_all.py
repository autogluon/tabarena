from __future__ import annotations

import time
from itertools import product
from pathlib import Path

import pandas as pd

from tabarena.nips2025_utils.per_dataset_tables import get_per_dataset_tables
from tabarena.paper.tabarena_evaluator import TabArenaEvaluator


def evaluate_all(
    df_results: pd.DataFrame,
    eval_save_path: str | Path,
    df_results_configs: pd.DataFrame = None,
    elo_bootstrap_rounds: int = 200,
    use_latex: bool = False,
):
    banned_pareto_methods = ["KNN", "LR"]

    evaluator_kwargs = {
        "use_latex": use_latex,
        "banned_pareto_methods": banned_pareto_methods,
    }

    eval_save_path = Path(eval_save_path)

    # TODO: Avoid hardcoding baselines
    _baselines = [
        "AutoGluon 1.4 (best, 4h)",
        "AutoGluon 1.4 (extreme, 4h)",
    ]
    _baseline_colors = [
        "black",
        "tab:purple"
    ]

    df_results = df_results.copy(deep=True)
    if "imputed" not in df_results.columns:
        df_results["imputed"] = False
    df_results["imputed"] = df_results["imputed"].fillna(0)

    if df_results_configs is not None:
        config_types_valid = df_results["config_type"].dropna().unique()
        df_results_configs_only_valid = df_results_configs[df_results_configs["config_type"].isin(config_types_valid)]
        plotter_runtime = TabArenaEvaluator(
            output_dir=eval_save_path / "ablation" / "all-runtimes",
            **evaluator_kwargs,
        )
        plotter_runtime.generate_runtime_plot(df_results=df_results_configs_only_valid)

    get_per_dataset_tables(
        df_results=df_results,
        save_path=eval_save_path / "per_dataset",
    )

    use_imputation_lst = [False, True]
    problem_type_lst = ["all", "classification", "regression", "binary", "multiclass"]
    dataset_subset_lst = [None, "small", "medium", "tabpfn"]
    with_baselines_lst = [True]
    lite_lst = [False, True]
    average_seeds_lst = [False]

    all_combinations = list(product(
        use_imputation_lst,
        problem_type_lst,
        with_baselines_lst,
        dataset_subset_lst,
        lite_lst,
        average_seeds_lst,
    ))
    n_combinations = len(all_combinations)

    # TODO: Use ray to speed up?
    ts = time.time()
    # plots for sub-benchmarks, with and without imputation
    for i, (use_imputation, problem_type, with_baselines, dataset_subset, lite, average_seeds) in enumerate(all_combinations):
        print(f"Running figure generation {i+1}/{n_combinations}... {(time.time() - ts):.1f}s elapsed...")

        evaluate_single(
            df_results=df_results,
            use_imputation=use_imputation,
            problem_type=problem_type,
            with_baselines=with_baselines,
            dataset_subset=dataset_subset,
            lite=lite,
            average_seeds=average_seeds,
            baselines=_baselines,
            baseline_colors=_baseline_colors,
            eval_save_path=eval_save_path,
            evaluator_kwargs=evaluator_kwargs,
            elo_bootstrap_rounds=elo_bootstrap_rounds,
        )


def evaluate_single(
    df_results,
    use_imputation,
    problem_type,
    with_baselines,
    dataset_subset,
    lite,
    average_seeds,
    baselines,
    baseline_colors,
    eval_save_path,
    evaluator_kwargs,
    elo_bootstrap_rounds: int = 200,
):
    from tabarena.nips2025_utils.compare import subset_tasks
    df_results = df_results.copy()

    subset = []
    folder_name = "all"
    if problem_type is not None:
        folder_name = f"{problem_type}"
        if problem_type == "all":
            pass
        else:
            subset.append(problem_type)
    if dataset_subset:
        folder_name_prefix = dataset_subset
        subset.append(dataset_subset)
    else:
        folder_name_prefix = "all"
    if lite:
        subset.append("lite")

    if subset:
        df_results = subset_tasks(df_results=df_results, subset=subset)

    if len(df_results) == 0:
        print(f"\tNo results after filtering, skipping...")
        return

    folder_name = str(Path(folder_name_prefix) / folder_name)
    if use_imputation:
        folder_name = folder_name + "-imputed"
    if not with_baselines:
        baselines = []
        baseline_colors = []
        folder_name = folder_name + "-nobaselines"

    datasets = list(df_results["dataset"].unique())

    imputed_freq = df_results.groupby(by=["ta_name", "ta_suite"])["imputed"].transform("mean")
    if not use_imputation:
        df_results = df_results.loc[imputed_freq <= 0]
    else:
        df_results = df_results.loc[imputed_freq < 1]  # always filter out methods that are imputed 100% of the time

    if len(datasets) == 0:
        return

    if lite:
        folder_name = str(Path("lite") / folder_name)
    if not average_seeds:
        folder_name = str(Path("no_average_seeds") / folder_name)

    plotter = TabArenaEvaluator(
        output_dir=eval_save_path / folder_name,
        datasets=datasets,
        elo_bootstrap_rounds=elo_bootstrap_rounds,
        **evaluator_kwargs,
    )

    plotter.eval(
        df_results=df_results,
        baselines=baselines,
        baseline_colors=baseline_colors,
        plot_extra_barplots=False,
        include_norm_score=True,
        plot_times=True,
        plot_other=False,
        average_seeds=average_seeds,
    )
