"""Code to use pre-defined benchmark environment and setup slurm jobs for (parallel) scheduling.

Example code:

# -- Benchmark XXX XX/XX/2026
BenchmarkSetup(
    benchmark_name="experiment_name_date",
    models=[
        ("ag_name", "all"),
    ],
).setup_jobs()

"""

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
    time_limit=60*60*2,
).setup_jobs()

# -- Benchmark LimiX 11/05/2026
BenchmarkSetup(
    benchmark_name="limix_11052026",
    models=[
        ("LimiX", 0),
    ],
    num_gpus=1,
    configs_per_job=1,
    slurm_gpu_partition="alldlc2_gpu-h200",
    fake_memory_for_estimates=140,
    # LimiX predict for large data is very slow, so we give it more time
    time_limit=60*60*4,
).setup_jobs()

# # -- Benchmark XXX XX/XX/2026
# BenchmarkSetup(
#     benchmark_name="experiment_name_date",
#     models=[
#         ("ag_name", "all"),
#     ],
# ).setup_jobs()
