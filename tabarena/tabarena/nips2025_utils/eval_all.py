from __future__ import annotations

import time
from itertools import product
from pathlib import Path

import pandas as pd

from tabarena.paper.tabarena_evaluator import TabArenaEvaluator


def get_all_subset_combinations() -> list[tuple[bool, str, bool, str | None, bool, bool]]:
    use_imputation_lst = [False, True]
    problem_type_lst = ["all", "classification", "regression", "binary", "multiclass"]
    dataset_subset_lst = [None, "small", "medium", "tabpfn"]
    with_baselines_lst = [True]
    lite_lst = [False, True]
    average_seeds_lst = [False]

    return list(product(
        use_imputation_lst,
        problem_type_lst,
        with_baselines_lst,
        dataset_subset_lst,
        lite_lst,
        average_seeds_lst,
    ))

def get_website_folder_name(
    *,
    use_imputation: bool,
    problem_type: str,
    dataset_subset: str | None,
    lite: bool,
) -> Path:
    folder_name = Path("website_data")
    folder_name = folder_name / ("imputation_yes" if use_imputation else "imputation_no")
    folder_name = folder_name / ("splits_lite" if lite else "splits_all")
    folder_name = folder_name / f"tasks_{problem_type}"
    dataset_subset_name = dataset_subset if dataset_subset is not None else "all"
    folder_name = folder_name / f"datasets_{dataset_subset_name}"
    return folder_name


def evaluate_all(
    tabarena_context,
    df_results: pd.DataFrame,
    eval_save_path: str | Path,
    elo_bootstrap_rounds: int = 200,
    use_latex: bool = False,
    use_website_folder_names: bool = False,
):
    banned_pareto_methods = ["KNN", "LR"]

    evaluator_kwargs = {
        "use_latex": use_latex,
        "banned_pareto_methods": banned_pareto_methods,
    }

    eval_save_path = Path(eval_save_path)

    # TODO: Avoid hardcoding baselines
    baselines = [
        "AutoGluon 1.4 (best, 4h)",
        "AutoGluon 1.4 (extreme, 4h)",
    ]
    baseline_colors = [
        "black",
        "tab:purple",
    ]

    df_results = df_results.copy(deep=True)
    if "imputed" not in df_results.columns:
        df_results["imputed"] = False
    df_results["imputed"] = df_results["imputed"].fillna(0)

    all_combinations = get_all_subset_combinations()
    n_combinations = len(all_combinations)

    # TODO: Use ray to speed up?
    ts = time.time()
    # plots for sub-benchmarks, with and without imputation
    for i, (use_imputation, problem_type, with_baselines, dataset_subset, lite, average_seeds) in enumerate(all_combinations):
        print(f"Running figure generation {i+1}/{n_combinations}... {(time.time() - ts):.1f}s elapsed...")
        custom_folder_name = None

        if use_website_folder_names:
            custom_folder_name = str(get_website_folder_name(
                use_imputation=use_imputation,
                problem_type=problem_type,
                dataset_subset=dataset_subset,
                lite=lite
            ))

        evaluate_single(
            tabarena_context=tabarena_context,
            df_results=df_results,
            use_imputation=use_imputation,
            problem_type=problem_type,
            with_baselines=with_baselines,
            dataset_subset=dataset_subset,
            lite=lite,
            average_seeds=average_seeds,
            eval_save_path=eval_save_path,
            evaluator_kwargs=evaluator_kwargs,
            baselines=baselines,
            baseline_colors=baseline_colors,
            elo_bootstrap_rounds=elo_bootstrap_rounds,
            custom_folder_name=custom_folder_name,
        )




def evaluate_single(
    tabarena_context,
    df_results,
    use_imputation,
    problem_type,
    with_baselines,
    dataset_subset,
    lite,
    average_seeds,
    eval_save_path,
    evaluator_kwargs,
    baselines: list[str] | None = None,
    baseline_colors: list[str] | None = None,
    elo_bootstrap_rounds: int = 200,
    custom_folder_name: str | None = None,
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

    imputed_freq = df_results.groupby(by=["ta_name", "ta_suite"])["imputed"].transform("mean")
    if not use_imputation:
        df_results = df_results.loc[imputed_freq <= 0]
    else:
        df_results = df_results.loc[imputed_freq < 1]  # always filter out methods that are imputed 100% of the time

    if len(df_results) == 0:
        print(f"\tNo results after filtering, skipping...")
        return

    if lite:
        folder_name = str(Path("lite") / folder_name)
    if not average_seeds:
        folder_name = str(Path("no_average_seeds") / folder_name)

    if custom_folder_name is not None:
        folder_name = custom_folder_name

    plotter = TabArenaEvaluator(
        output_dir=eval_save_path / folder_name,
        elo_bootstrap_rounds=elo_bootstrap_rounds,
        tabarena_context=tabarena_context,
        **evaluator_kwargs,
    )

    eval_kwargs = {}
    if baselines is not None:
        eval_kwargs["baselines"] = baselines
    if baseline_colors is not None:
        eval_kwargs["baseline_colors"] = baseline_colors

    plotter.eval(
        df_results=df_results,
        plot_extra_barplots=False,
        include_norm_score=True,
        plot_times=True,
        plot_other=False,
        average_seeds=average_seeds,
        **eval_kwargs,
    )
