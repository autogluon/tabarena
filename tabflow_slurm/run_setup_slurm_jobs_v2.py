"""SLURM Scheduling code."""

from __future__ import annotations

from tabflow_slurm.benchmarking_setup.download_data_foundry_datasets import (
    get_metadata_for_benchmark_suite,
)
from tabflow_slurm.setup_slurm_base_v2 import BenchmarkSetup2026

# -- Minimal Working Example for new benchmark setup
BenchmarkSetup2026(
    benchmark_name="23032026_toy_example",
    task_metadata=get_metadata_for_benchmark_suite("example_benchmark_suite"),
    models=[
        ("LightGBM", 0),
    ],
    split_indices_to_run="lite",
    ignore_cache=True,
).setup_jobs()
