from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from tabarena.nips2025_utils.artifacts import tabarena_method_metadata_collection
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena import EvaluationRepository
from tabarena.simulation.ensemble_scorer_calibrated import EnsembleScorerCalibrated, EnsembleScorerCalibratedCV
from tabarena.simulation.ensemble_selection_config_scorer import EnsembleScorer
from autogluon.common import TabularDataset
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata


def run_cal(
    method_metadata: MethodMetadata,
    name_suffix: str,
    ensemble_cls,
    ensemble_kwargs: dict = None,
    **shared_kwargs,
):
    if ensemble_kwargs is None:
        ensemble_kwargs = {}
    ts = time.time()
    df_results = method_metadata.generate_hpo_result(
        ensemble_cls=ensemble_cls,
        ensemble_kwargs=ensemble_kwargs,
        **shared_kwargs,
    )
    te = time.time()
    time_total = te - ts
    df_results["method"] = df_results["ta_name"] + name_suffix
    df_results["method_type"] = "baseline"
    return df_results, time_total


def simulate_calibration(method_metadata, run_toy: bool = False, problem_types: list[str] = None):
    if problem_types is None:
        problem_types = ["multiclass"]
    if not method_metadata.path_processed_exists:
        # download the processed data if needed. Will take some time (~15 GB for methods with 201 configs)
        print(f"Downloading processed data for {method_metadata.method}...")
        method_metadata.method_downloader(verbose=True).download_processed()

    repo_cache_path = f"repo_tmp_cal_all_folds_{method_metadata.method}_{'_'.join(sorted(problem_types))}.pkl"

    if run_toy:
        if not (Path(repo_cache_path).exists() and Path(repo_cache_path).is_file()):
            repo = method_metadata.load_processed()
            repo = repo.subset(
                configs=repo.configs()[:5],
                problem_types=problem_types,
            )
            repo.save(path=repo_cache_path)
        # much faster for debugging
        repo = EvaluationRepository.load(path=repo_cache_path)
        shared_kwargs = dict(
            repo=repo,
            fit_order="original",
            backend="native",
        )
    else:
        repo = method_metadata.load_processed()  # the full data
        repo = repo.subset(problem_types=problem_types)
        shared_kwargs = dict(
            repo=repo,
            fit_order="original",
        )

    conf_original = dict(
        name_suffix="-ORIGINAL",
        ensemble_cls=EnsembleScorer,
    )

    conf_cal = dict(
        name_suffix="-CAL-LOGISTIC",
        ensemble_cls=EnsembleScorerCalibrated,
        ensemble_kwargs={
            "use_fast_metrics": False,
            "calibrate_per_model": False,
            "calibrate_after_ens": True,
            "calibrator_type": "logistic",
        },
    )

    conf_cal_cv = dict(
        name_suffix="-CAL-LOGISTIC-CV",
        ensemble_cls=EnsembleScorerCalibratedCV,
        ensemble_kwargs={
            "use_fast_metrics": False,
            "calibrate_per_model": False,
            "calibrate_after_ens": True,
            "calibrator_type": "logistic",
        },
    )

    conf_cal_per = dict(
        name_suffix="-CAL-LOGISTIC-PER",
        ensemble_cls=EnsembleScorerCalibrated,
        ensemble_kwargs={
            "use_fast_metrics": False,
            "calibrator_type": "logistic",
            "calibrate_per_model": True,
            "calibrate_after_ens": False,
        },
    )

    conf_cal_per_cv = dict(
        name_suffix="-CAL-LOGISTIC-PER-CV",
        ensemble_cls=EnsembleScorerCalibratedCV,
        ensemble_kwargs={
            "use_fast_metrics": False,
            "calibrator_type": "logistic",
            "calibrate_per_model": True,
            "calibrate_after_ens": False,
        },
    )

    conf_temp = dict(
        name_suffix="-CAL-TEMPERATURE",
        ensemble_cls=EnsembleScorerCalibrated,
        ensemble_kwargs={
            "use_fast_metrics": False,
            "calibrator_type": "ts-mix",
            "calibrate_per_model": False,
            "calibrate_after_ens": True,
        },
    )

    conf_cal_both = dict(
        name_suffix="-CAL-LOGISTIC-BOTH",
        ensemble_cls=EnsembleScorerCalibrated,
        ensemble_kwargs={
            "use_fast_metrics": False,
            "calibrator_type": "logistic",
            "calibrate_per_model": True,
            "calibrate_after_ens": True,
        },
    )

    conf_cal_both_cv = dict(
        name_suffix="-CAL-LOGISTIC-BOTH-CV",
        ensemble_cls=EnsembleScorerCalibratedCV,
        ensemble_kwargs={
            "use_fast_metrics": False,
            "calibrator_type": "logistic",
            "calibrate_per_model": True,
            "calibrate_after_ens": True,
        },
    )

    conf_list = [
        conf_original,
        conf_cal,
        conf_cal_cv,
        conf_cal_per,
        conf_cal_per_cv,
        # conf_temp,
        # conf_cal_both,
        # conf_cal_both_cv,
    ]

    df_results_lst = []
    time_dict = {}
    for conf in conf_list:
        df_results, runtime = run_cal(
            method_metadata=method_metadata,
            **conf,
            **shared_kwargs,
        )
        df_results_lst.append(df_results)
        time_dict[conf["name_suffix"]] = runtime

    new_results = pd.concat([
        *df_results_lst,
    ], ignore_index=True)
    new_results_methods = list(new_results["method"].unique())

    ta_context = TabArenaContext()
    leaderboard = ta_context.compare(
        output_dir=Path("output_test_calibration") / method_metadata.artifact_name / method_metadata.method,
        new_results=new_results,
        only_valid_tasks=True,
        average_seeds=False,
    )

    leaderboard_new_results = leaderboard[leaderboard["method"].isin(new_results_methods)]

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard)
        print(leaderboard_new_results)

    print(
        f"Simulation Runtimes:"
    )
    for name_suffix, runtime in time_dict.items():
        print(f"\t{name_suffix}:\t {runtime:.1f}s")
    print()
    return new_results


if __name__ == "__main__":
    metadata_lst = tabarena_method_metadata_collection.method_metadata_lst
    metadata_lst = [m for m in metadata_lst if m.method_type == "config"]
    run_toy = True  # If True, only calculates using up to 5 configs per method and runs sequentially (debugger friendly).
    run_only_small_methods = True  # If True, avoids downloading large method results (only runs models that have a single config). If False, will end up downloading 300+ GB of model predictions if not already present.
    # metadata_lst = [m for m in metadata_lst if m.method == "LightGBM"]
    # metadata_lst = [m for m in metadata_lst if m.method == "RealTabPFN-v2.5"]
    cache_overwrite = False
    problem_types = ["multiclass"]
    if run_only_small_methods:
        metadata_lst = [m for m in metadata_lst if not m.can_hpo]
    if run_toy:
        out_dir = "calibration_results_toy"
    else:
        out_dir = "calibration_results"
    if run_only_small_methods:
        out_dir += "_only_small"
    out_dir = Path(out_dir)
    out_dir = out_dir / f"{'_'.join(sorted(problem_types))}"

    num_methods = len(metadata_lst)
    new_results_lst = []
    for i, metadata in enumerate(metadata_lst):
        print(f"({i+1}/{num_methods}) Running calibration for {metadata.method}")
        cache_dir = Path(out_dir) / f"{metadata.artifact_name}" / f"{metadata.method}.pkl"
        if not cache_overwrite and cache_dir.exists():
            cur_new_results = TabularDataset.load(path=cache_dir)
        else:
            cur_new_results = simulate_calibration(method_metadata=metadata, run_toy=run_toy, problem_types=problem_types)
            TabularDataset.save(path=cache_dir, df=cur_new_results)
        new_results_lst.append(cur_new_results)
    all_new_results = pd.concat(new_results_lst, ignore_index=True)

    subset = []

    # all_new_results = all_new_results[all_new_results["method"].str.contains("-LOGISTIC")]

    ta_context = TabArenaContext()
    leaderboard = ta_context.compare(
        output_dir="output_test_calibration_all",
        new_results=all_new_results,
        only_valid_tasks=True,
        average_seeds=False,
        # score_on_val=True,  # Uncomment to look at validation scores instead of test scores
        subset=subset,
        remove_imputed=True,
    )

    leaderboard_val = ta_context.compare(
        output_dir="output_test_calibration_all_val",
        new_results=all_new_results,
        only_valid_tasks=True,
        average_seeds=False,
        score_on_val=True,  # Uncomment to look at validation scores instead of test scores
        subset=subset,
        remove_imputed=True,
    )
    leaderboard["elo_val"] = leaderboard["method"].map(leaderboard_val.set_index("method")["elo"])
    leaderboard["elo_delta"] = leaderboard["elo"] - leaderboard["elo_val"]

    all_new_results_methods = list(all_new_results["method"].unique())
    leaderboard_new_results = leaderboard[leaderboard["method"].isin(all_new_results_methods)]

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard)
        print(leaderboard_new_results)

        leaderboard_new_results_v2 = leaderboard_new_results[["method", "elo", "elo_val", "elo_delta", "improvability"]]
        print(leaderboard_new_results_v2)
