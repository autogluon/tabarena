from __future__ import annotations

from experimental.feature_selection_benchmark.data_integration.fs_data_constants import (
    BENCHMARK_DATA_FOUNDRY_TASKS,
    BENCHMARK_TASK_COLLECTION_NAME,
    DATA_FOUNDRY_CACHE,
    OPENML_CACHE,
)
from tabflow_slurm.benchmarking_setup.data_foundry_integration.data_foundry_task_creator import (
    download_data_foundry_datasets,
)

if __name__ == "__main__":
    download_data_foundry_datasets(
        benchmark_suite_name=BENCHMARK_TASK_COLLECTION_NAME,
        data_foundry_artifacts=BENCHMARK_DATA_FOUNDRY_TASKS,
        data_foundry_cache=DATA_FOUNDRY_CACHE,
        openml_cache=OPENML_CACHE,
    )
