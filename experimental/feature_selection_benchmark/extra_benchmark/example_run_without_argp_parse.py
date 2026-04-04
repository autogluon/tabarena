from __future__ import annotations

from itertools import product

from tabarena.benchmark.feature_selection_methods.feature_selection_benchmark_utils import (
    get_fs_benchmark_preprocessing_pipelines,
)
from tabflow_slurm.benchmarking_setup.data_foundry_integration.data_foundry_task_creator import (
    get_metadata_for_benchmark_suite,
)
from tabflow_slurm.benchmarking_setup.download_feature_selection_datasets import DEFAULT_DATA_FOUNDRY_CACHE

from experimental.feature_selection_benchmark.extra_benchmark.feature_selection_benchmark_runner import run_benchmark


def simple_loop_runner():
    """Example runner without any parallelization.

    TODO: parallelize this loop (using SLURM), add code to save the result and check if we need to rerun things,
        and add loading results and computing metrics from it.
    """
    # Get metadata for the benchmark suite
    benchmark_task_metadata = get_metadata_for_benchmark_suite(
        "feature_selection_benchmark", data_foundry_cache=DEFAULT_DATA_FOUNDRY_CACHE
    )
    # We only need the task_id_str and can ignore the rest of the metadata.
    all_task_id_str = list(benchmark_task_metadata[["task_id_str"]].drop_duplicates()["task_id_str"])

    # Get some feature selection methods to run
    preprocessing_pipelines = get_fs_benchmark_preprocessing_pipelines(
        fs_methods=[
            "RandomFeatureSelector",
            "PearsonCorrelationFeatureSelector",
            "AccuracyFeatureSelector",
        ],
        proxy_model_config=["lgbm"],
        time_limit=[3600],
        total_budget=5,
        include_default=False,
    )

    # Which repeats to run
    repeats = [0, 1, 2]

    # Define the modes we want to test
    modes = ["validity", "stability"]
    modes_settings = {
        "validity": {"noise": [0.25, 0.5, 0.75, 1.0], "noise_type": ["gaussian"]},
        "stability": {},
    }

    for task_id_str, preprocessing_pipeline, repeat, mode in product(
        all_task_id_str, preprocessing_pipelines, repeats, modes
    ):
        mode_configs = modes_settings[mode]
        if len(mode_configs) == 0:
            mode_configs = {}
        else:
            mode_configs = product(*[[(k, v) for v in values] for k, values in mode_configs.items()])

        print(
            f"=== Feature Selection Benchmark:"
            f"\n\tTask: {task_id_str}"
            f"\n\tPreprocessing Pipeline: {preprocessing_pipeline}"
            f"\n\tRepeat: {repeat}"
            f"\n\tMode: {mode}"
            f"\n\tMode Configs: {mode_configs}"
        )

        result = run_benchmark(
            task_id_str=task_id_str,
            preprocessing_pipeline=preprocessing_pipeline,
            repeat=repeat,
            mode=mode,
            **mode_configs,
        )

        print(result)


if __name__ == "__main__":
    simple_loop_runner()