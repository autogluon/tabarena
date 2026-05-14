"""Code to use pre-defined benchmark environment and setup slurm jobs for (parallel) scheduling."""

from __future__ import annotations

from tabflow_slurm.setup_slurm_base import BenchmarkSetup

# -- Benchmark iltm 14/05/2026
BenchmarkSetup(
    benchmark_name="benchmark_iltm_14052026",
    models=[
        ("iLTM", 0),
    ],
    num_gpus=1,
    configs_per_job=1,
    time_limit=60*60*1,
    tabarena_lite=True,
).setup_jobs()

# # -- Benchmark XXX XX/XX/2026
# BenchmarkSetup(
#     benchmark_name="experiment_name_date",
#     models=[
#         ("ag_name", "all"),
#     ],
# ).setup_jobs()
