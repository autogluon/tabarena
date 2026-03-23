"""Code to use pre-defined benchmark environment and setup slurm jobs for (parallel) scheduling."""

from __future__ import annotations

from tabflow_slurm.setup_slurm_base import BenchmarkSetup

# -- Benchmark TabPFN-v2.6 23/03/2026
BenchmarkSetup(
    benchmark_name="230326_tabpfnv26",
    models=[
        ("TabPFN-v2.6", 0),
    ],
    num_gpus=1,
    configs_per_job=1,
    slurm_gpu_partition="alldlc2_gpu-h200",
    fake_memory_for_estimates=140,
    time_limit=60*60*2,
).setup_jobs()
