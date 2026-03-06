"""Code to use pre-defined benchmark environment and setup slurm jobs for (parallel) scheduling."""

from __future__ import annotations

from tabflow_slurm.setup_slurm_base import BenchmarkSetup

# -- Benchmark XXX XX/XX/2026
BenchmarkSetup(
    benchmark_name="experiment_name_date",
    models=[
        ("ag_name", "all"),
    ],
).setup_jobs()
