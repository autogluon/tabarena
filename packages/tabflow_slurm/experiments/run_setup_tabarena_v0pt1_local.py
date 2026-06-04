"""Local, sequential variant of ``run_setup_tabarena_v0pt1.py`` (no SLURM).

Same authoring model as the SLURM script — compose a ``TabArenaBenchmarkPlan`` and
call ``setup_jobs()`` — but swaps ``GCPSlurmSetup`` for ``LocalSequentialSetup``. The
plan still enumerates ``(task, fold, repeat, config)`` items, checks the cache, and
writes a job JSON; instead of ``sbatch`` commands it emits a single
``python -m tabflow_slurm.run_local <json>`` command that runs every item one at a
time, each in its own subprocess. With ``RUN_NOW=True`` this script also executes
that command, so a single ``python run_setup_tabarena_v0pt1_local.py`` does
generate-and-run end to end.

Defaults are CPU-only and scoped to one tiny dataset so it runs on any laptop/VM
with no GPU. The first run downloads the dataset(s) from OpenML.

Evaluate the results with ``run_eval_tabarena_v0pt1_local.py`` (reuse the same
``BENCHMARK_NAME`` and ``workspace``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TabArenaV0pt1LiteMetadataBundle
from tabflow_slurm import (
    LocalSequentialSetup,
    ModelJob,
    PathSetup,
    TabArenaBenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

BENCHMARK_NAME = "example_tabarena_v0pt1_local"
WORKSPACE = str(Path.home() / "tabarena_local_workspace")
RUN_NOW = True  # set False to only generate the job JSON + print the run command

benchmark_plan = TabArenaBenchmarkPlan(
    benchmark_name=BENCHMARK_NAME,
    model_jobs=[
        # CPU-only baselines, same `name` so they share one run (one command).
        ModelJob(models=("RandomForest", 0), name="cpu"),  # default config only
        ModelJob(models=("Linear", 1), name="cpu"),  # default + 1 random config
        # To add a GPU model (needs a local GPU + the `tabpfn` extra), give it a
        # GPU resource override so it becomes its own sequential run:
        #   ModelJob(models=("TabPFN-3", 0), name="gpu", resources={"num_gpus": 1}),
    ],
    # Scope to one tiny dataset so the demo is fast — the full Lite bundle is 51
    # datasets. Add more names (e.g. "diabetes") to widen it.
    tasks_to_run_setup=TabArenaV0pt1LiteMetadataBundle(
        dataset_names_to_run=["blood-transfusion-service-center"],
    ),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(),
    path_setup=PathSetup(
        workspace=WORKSPACE,
        # Use the interpreter running this script for the subprocess fits.
        python_path=sys.executable,
    ),
    # Modest resources for a laptop (defaults are 8 CPU / 32 GB / 1h).
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=4, memory_limit=8, time_limit=600),
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
