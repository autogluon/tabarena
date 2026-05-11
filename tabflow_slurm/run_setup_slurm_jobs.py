"""Code to use pre-defined benchmark environment and setup slurm jobs for (parallel) scheduling."""

from __future__ import annotations

from tabflow_slurm.setup_slurm_base import BenchmarkSetup

# -- Benchmark TabPFN-3 11/05/2026
BenchmarkSetup(
    benchmark_name="benchmark_tabpfn_3_11052026",
    models=[
        ("TabPFN-3", 0),
    ],
    num_gpus=1,
    configs_per_job=1,
    slurm_gpu_partition="alldlc2_gpu-h200",
    fake_memory_for_estimates=140,
).setup_jobs()

# # -- Benchmark XXX XX/XX/2026
# BenchmarkSetup(
#     benchmark_name="experiment_name_date",
#     models=[
#         ("ag_name", "all"),
#     ],
# ).setup_jobs()
