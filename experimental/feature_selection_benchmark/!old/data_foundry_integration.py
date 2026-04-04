from __future__ import annotations

from pathlib import Path

from tabflow_slurm.benchmarking_setup.data_foundry_integration.data_foundry_task_creator import (
    download_data_foundry_datasets,
)

DEFAULT_DATA_FOUNDRY_CACHE = Path(__file__).parent / ".data_foundry_cache"

EXAMPLE_DATA_FOUNDRY_TASKS = [
    "heart_failure_followup_survival/019c7bff-a1d3-788d-8b8d-86bad1c6fb5f",
    "lymphography/019c75c5-725c-708c-9794-eccc46c0bf81",
    "heart_disease_cleveland/019c7513-909c-707d-a9da-9852f346a015",
]

if __name__ == "__main__":
    download_data_foundry_datasets(
        benchmark_suite_name="feature_selection_benchmark_examples",
        data_foundry_artifacts=EXAMPLE_DATA_FOUNDRY_TASKS,
        data_foundry_cache=DEFAULT_DATA_FOUNDRY_CACHE,
    )