from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tabarena.benchmark.experiment import ExperimentBatchRunner, AGModelBagExperiment
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.website.website_format import format_leaderboard


if __name__ == '__main__':
    expname = str(Path(__file__).parent / "experiments" / "quickstart_tabiclv2_amlb")  # folder location to save all experiment artifacts
    eval_dir = Path(__file__).parent / "eval" / "quickstart_tabiclv2_amlb"
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    # tabarena_context = TabArenaContext()
    # task_metadata = tabarena_context.task_metadata

    from autogluon.common import TabularDataset
    task_metadata = TabularDataset.load("../../tabarena/tabarena/nips2025_utils/scripts/task_metadata_amlb.csv")


    # Sample for a quick demo
    # datasets = ["anneal", "credit-g", "diabetes"]  # datasets = list(task_metadata["name"])
    # datasets = ["SDSS17"]
    task_metadata = task_metadata[task_metadata["NumberOfInstances"] <= 1000000]
    # task_metadata = task_metadata[task_metadata["NumberOfFeatures"] <= 501]
    task_metadata["dataset"] = task_metadata["name"]
    datasets = list(task_metadata["name"])
    # datasets = datasets[:4]
    # datasets = ["MIC"]  # TODO: Takes 60 seconds for 1 iter?
    # datasets = ["credit-g"]
    # datasets = ["Bioresponse"]
    # datasets = ["anneal"]
    # datasets = ["blood-transfusion-service-center"]
    folds = [0]
    # datasets = datasets[:1]

    # import your model classes
    from tabarena.benchmark.models.ag.tabicl.tabicl_model import TabICLv2Model

    # This list of methods will be fit sequentially on each task (dataset x fold)
    methods = [
        AGModelBagExperiment(
            name="TabICLv2_c1_BAG_L1_REPRODUCE_1_ESTIMATOR",
            model_cls=TabICLv2Model,
            model_hyperparameters={
                'n_estimators': 1,
                'ag_args_ensemble': {'model_random_seed': 0, 'vary_seed_across_folds': True}
            },
            num_bag_folds=2,
            time_limit=3600,
        ),
    ]

    exp_batch_runner = ExperimentBatchRunner(expname=expname, task_metadata=task_metadata)

    # Get the run artifacts.
    # Fits each method on each task (datasets * folds)
    results_lst: list[dict[str, Any]] = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        methods=methods,
        ignore_cache=ignore_cache,
    )

    def to_gib(x):
        return x / 1024 ** 3

    for result in results_lst:
        peak_mem_cpu = to_gib(result["memory_usage"]["peak_mem_cpu"])
        min_mem_cpu = to_gib(result["memory_usage"]["min_mem_cpu"])
        est_peak_mem_cpu = result["method_metadata"]["memory_usage_estimate"]
        if est_peak_mem_cpu is not None:
            est_peak_mem_cpu = to_gib(est_peak_mem_cpu)
        delta_mem_cpu = peak_mem_cpu - min_mem_cpu
        if est_peak_mem_cpu is not None:
            ratio_mem_cpu = delta_mem_cpu / est_peak_mem_cpu
        else:
            ratio_mem_cpu = -1
            est_peak_mem_cpu = -1

        disk_usage = to_gib(result["method_metadata"]["disk_usage"])

        print(
            f"{peak_mem_cpu:.4f}, "
            f"{est_peak_mem_cpu:.4f}, "
            f"{delta_mem_cpu:.4f}, "
            f"{ratio_mem_cpu:.4f}, " 
            f"{disk_usage:.4f}"
        )

        if result["memory_usage"]["gpu_tracking_enabled"]:
            peak_mem_gpu = to_gib(result["memory_usage"]["peak_mem_gpu"])
            peak_mem_gpu_reserved = to_gib(result["memory_usage"]["peak_mem_gpu_reserved"])
            print(f"\tGPU: {peak_mem_gpu:.2f} GB used, {peak_mem_gpu_reserved:.2f} GB reserved")

    # raise AssertionError

    # compute results
    end_to_end = EndToEnd.from_raw(results_lst=results_lst, task_metadata=task_metadata, cache=False, cache_raw=False)
    end_to_end_results = end_to_end.to_results()

    print(f"New Configs Hyperparameters: {end_to_end.configs_hyperparameters()}")

    # leaderboard: pd.DataFrame = end_to_end_results.compare_on_tabarena(
    #     output_dir=eval_dir,
    #     only_valid_tasks=True,  # True: only compare on tasks ran in `results_lst`
    #     use_model_results=True,  # If False: Will instead use the ensemble/HPO results
    #     new_result_prefix="Demo_",
    # )
    # leaderboard_website = format_leaderboard(df_leaderboard=leaderboard)
    # print(leaderboard_website.to_markdown(index=False))

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Results:\n{end_to_end_results.model_results.head(100)}")
