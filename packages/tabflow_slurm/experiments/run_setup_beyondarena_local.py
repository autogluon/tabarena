"""Local, sequential variant of ``run_setup_beyondarena.py`` (no SLURM)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabflow_slurm import (
    BeyondArenaResourcesSetup,
    LocalSequentialSetup,
    ModelJob,
    PathSetup,
    TabArenaBenchmarkPlan,
)

BENCHMARK_NAME = "example_beyondarena_local"
WORKSPACE = str(Path.home() / "tabarena_local_workspace")
RUN_NOW = True  # set False to only generate the job JSON + print the run command

benchmark_plan = TabArenaBenchmarkPlan(
    benchmark_name=BENCHMARK_NAME,
    model_jobs=[
        ModelJob(models=("RandomForest", 0), name="cpu"),  # default config only
        ModelJob(models=("Linear", 1), name="cpu"),  # default + 1 random config
    ],
    # Scope to the smallest dataset (155 rows) so the demo is fast — the full
    # BeyondArena preset is 142 datasets. Add more names to widen it; only the
    # selected datasets are downloaded.
    tasks=TaskMetadataCollection.from_preset("BeyondArena").subset_tasks(
        dataset_names=["hepatitis_survival_prediction"],
        split_indices="lite",
    ),
    experiment_bundle=BeyondArenaExperimentBundle(),
    path_setup=PathSetup(
        workspace=WORKSPACE,
        # Use the interpreter running this script for the subprocess fits.
        python_path=sys.executable,
    ),
    # Modest resources for a laptop (defaults are all-CPUs / all-memory / 4h).
    resources_setup=BeyondArenaResourcesSetup(num_cpus=4, memory_limit=8, time_limit=600),
    scheduler_setup=LocalSequentialSetup(continue_on_error=True),
    # No foundation models selected, so there is nothing to prefetch.
    prefetch_model_weights=False,
)

commands = benchmark_plan.setup_jobs()

if RUN_NOW:
    # Run each group's command sequentially. `shell=True` mirrors how the SLURM
    # `sbatch` commands are emitted as strings; keep `WORKSPACE` free of spaces.
    for cmd in commands:
        print(f"\n>>> {cmd}")
        subprocess.run(cmd, shell=True, check=False)  # noqa: S602
