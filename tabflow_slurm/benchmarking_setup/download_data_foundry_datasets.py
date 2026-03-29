from __future__ import annotations

from pathlib import Path

from tabflow_slurm.benchmarking_setup.data_foundry_integration.data_foundry_task_creator import (
    download_data_foundry_datasets,
)

DEFAULT_DATA_FOUNDRY_CACHE = Path(__file__).parent / ".data_foundry_cache"

EXAMPLE_DATA_FOUNDRY_TASKS = [
    # Grouped tiny data
    "musk/019cb408-670c-7088-bf5e-eb09cb01e9b2",
    # Temporal data
    "mercedes_benz_greener_manufacturing/019c0e8e-8749-7ff7-9c06-632c3ca2aa05",
    # IID Tabular Text Data
    "wine_world_cost/019c32f6-9391-7812-b543-66fbb299dc51",
]


if __name__ == "__main__":
    download_data_foundry_datasets(
        benchmark_suite_name="example_benchmark_suite",
        data_foundry_artifacts=EXAMPLE_DATA_FOUNDRY_TASKS,
        data_foundry_cache=DEFAULT_DATA_FOUNDRY_CACHE,
    )
