from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from tabarena.nips2025_utils.artifacts import tabarena_method_metadata_collection
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena import EvaluationRepository
from tabarena.simulation.ensemble_scorer_calibrated import EnsembleScorerCalibrated
from tabarena.simulation.ensemble_selection_config_scorer import EnsembleScorer
from autogluon.common import TabularDataset


def simulate_calibration(method_metadata, run_toy: bool = False):
    if not method_metadata.path_processed_exists:
        # download the processed data if needed. Will take some time (~15 GB for methods with 201 configs)
        print(f"Downloading processed data for {method_metadata.method}...")
        method_metadata.method_downloader(verbose=True).download_processed()

    repo_cache_path = f"repo_tmp_cal_all_folds_{method_metadata.method}.pkl"

    if run_toy:
        if not (Path(repo_cache_path).exists() and Path(repo_cache_path).is_file()):
            repo = method_metadata.load_processed()
            repo = repo.subset(
                configs=repo.configs()[:5],
                problem_types=["multiclass"],
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
        repo = repo.subset(problem_types=["multiclass"])
        shared_kwargs = dict(
            repo=repo,
            fit_order="original",
        )

    ts = time.time()
    df_results_hpo_og = method_metadata.generate_hpo_result(
        ensemble_cls=EnsembleScorer,  # Normal logic
        **shared_kwargs,
    )
    te = time.time()
    time_og = te - ts
    df_results_hpo_og["method"] = df_results_hpo_og["ta_name"] + "-ORIGINAL"
    df_results_hpo_og["method_type"] = "baseline"  # Do this so it appears on the leaderboard

    ts = time.time()
    df_results_hpo_cal = method_metadata.generate_hpo_result(
        ensemble_cls=EnsembleScorerCalibrated,  # Your logic
        ensemble_kwargs={
            "use_fast_metrics": False,
            "calibrate_per_model": False,
            "calibrate_after_ens": True,
            "calibrator_type": "logistic",
        },  # Feel free to add any init kwargs for the ensemble cls here
        **shared_kwargs,
    )
    te = time.time()
    time_cal = te - ts
    df_results_hpo_cal["method"] = df_results_hpo_cal["ta_name"] + "-CAL-LOGISTIC"
    df_results_hpo_cal["method_type"] = "baseline"  # Do this so it appears on the leaderboard

    ts = time.time()
    df_results_hpo_cal_per = method_metadata.generate_hpo_result(
        ensemble_cls=EnsembleScorerCalibrated,  # Your logic
        ensemble_kwargs={
            "use_fast_metrics": False,
            "calibrator_type": "logistic",
            "calibrate_per_model": True,
            "calibrate_after_ens": False,
        },  # Feel free to add any init kwargs for the ensemble cls here
        **shared_kwargs,
    )
    te = time.time()
    time_cal_per = te - ts
    df_results_hpo_cal_per["method"] = df_results_hpo_cal_per["ta_name"] + "-CAL-LOGISTIC-PER"
    df_results_hpo_cal_per["method_type"] = "baseline"  # Do this so it appears on the leaderboard


    ts = time.time()
    df_results_hpo_cal_temp = method_metadata.generate_hpo_result(
        ensemble_cls=EnsembleScorerCalibrated,  # Your logic
        ensemble_kwargs={
            "use_fast_metrics": False,
            "calibrator_type": "ts-mix",
            "calibrate_per_model": False,
            "calibrate_after_ens": True,
        },  # Feel free to add any init kwargs for the ensemble cls here
        **shared_kwargs,
    )
    te = time.time()
    time_cal_temp = te - ts
    df_results_hpo_cal_temp["method"] = df_results_hpo_cal_temp["ta_name"] + "-CAL-TEMPERATURE"
    df_results_hpo_cal_temp["method_type"] = "baseline"  # Do this so it appears on the leaderboard

    new_results = pd.concat([df_results_hpo_og, df_results_hpo_cal, df_results_hpo_cal_temp, df_results_hpo_cal_per], ignore_index=True)
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
        f"Simulation Runtimes:\n"
        f"\t  Original: {time_og:.1f}s\n"
        f"\tCal-Logist: {time_cal:.1f}s\n"
        f"\tCal-LogPer: {time_cal_per:.1f}s\n"
        f"\tCal-Temper: {time_cal_temp:.1f}s\n"
    )
    return new_results


if __name__ == "__main__":
    metadata_lst = tabarena_method_metadata_collection.method_metadata_lst
    metadata_lst = [m for m in metadata_lst if m.method_type == "config"]
    run_toy = True
    if run_toy:
        out_dir = "calibration_results_toy"
    else:
        out_dir = "calibration_results"

    num_methods = len(metadata_lst)
    new_results_lst = []
    for i, metadata in enumerate(metadata_lst):
        print(f"({i+1}/{num_methods}) Running calibration for {metadata.method}")
        cache_dir = Path(out_dir) / f"{metadata.artifact_name}" / f"{metadata.method}.pkl"
        if cache_dir.exists():
            cur_new_results = TabularDataset.load(path=cache_dir)
        else:
            cur_new_results = simulate_calibration(method_metadata=metadata, run_toy=run_toy)
            TabularDataset.save(path=cache_dir, df=cur_new_results)
        new_results_lst.append(cur_new_results)
    all_new_results = pd.concat(new_results_lst, ignore_index=True)

    # all_new_results = all_new_results[all_new_results["method"].str.contains("-LOGISTIC")]

    ta_context = TabArenaContext()
    leaderboard = ta_context.compare(
        output_dir="output_test_calibration_all",
        new_results=all_new_results,
        only_valid_tasks=True,
        average_seeds=False,
        score_on_val=True,
    )

    all_new_results_methods = list(all_new_results["method"].unique())
    leaderboard_new_results = leaderboard[leaderboard["method"].isin(all_new_results_methods)]

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard)
        print(leaderboard_new_results)
