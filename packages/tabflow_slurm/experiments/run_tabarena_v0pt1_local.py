"""Local, sequential variant of ``run_tabarena_v0pt1.py`` (no SLURM): `setup` + `eval` in one file.

Same authoring model as the SLURM script — compose a ``TabArenaBenchmarkPlan`` and call
``setup_jobs()`` — but swaps ``GCPSlurmSetup`` for ``LocalSequentialSetup``. Instead of ``sbatch``
commands, `setup` emits a single ``python -m tabflow_slurm.run_local <json>`` command that runs
every ``(task, fold, repeat, config)`` item one at a time, each in its own subprocess. With
``RUN_NOW = True`` the `setup` subcommand also executes that command, so it is generate-and-run
end to end::

    python experiments/run_tabarena_v0pt1_local.py setup   # generate (+ run, if RUN_NOW)
    python experiments/run_tabarena_v0pt1_local.py eval      # build the leaderboard

`setup` and `eval` share ``BENCHMARK_NAME`` + ``WORKSPACE`` (defined once below) so they resolve the
same results dir. Defaults are CPU-only and scoped to one tiny dataset so it runs on any laptop/VM
with no GPU. The first run downloads the dataset(s) from OpenML.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabarena.contexts import TabArenaContext
from tabarena.evaluation import EvalMethod, TabArenaEvalConfig, run_eval
from tabflow_slurm import (
    LocalSequentialSetup,
    ModelJob,
    PathSetup,
    TabArenaBenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

# ── Shared identity — the ONE place these live; setup + eval both read them ──
BENCHMARK_NAME = "example_tabarena_v0pt1_local"
WORKSPACE = str(Path.home() / "tabarena_local_workspace")
PYTHON_PATH = sys.executable  # use the interpreter running this script for the subprocess fits
RUN_NOW = True  # set False to only generate the job JSON + print the run command


def _path_setup() -> PathSetup:
    return PathSetup(workspace=WORKSPACE, python_path=PYTHON_PATH)


def setup() -> None:
    """Generate the local job JSON and (if ``RUN_NOW``) run every item sequentially."""
    plan = TabArenaBenchmarkPlan(
        benchmark_name=BENCHMARK_NAME,
        model_jobs=[
            # CPU-only baselines, same `name` so they share one run (one command).
            ModelJob(models=("RandomForest", 0), name="cpu"),  # default config only
            ModelJob(models=("Linear", 1), name="cpu"),  # default + 1 random config
            # To add a GPU model (needs a local GPU + the `tabpfn` extra), give it a
            # GPU resource override so it becomes its own sequential run:
            #   ModelJob(models=("TabPFN-3", 0), name="gpu", resources={"num_gpus": 1}),
        ],
        # Scope to one tiny dataset so the demo is fast — the full Lite suite is 51 datasets.
        # Add more names (e.g. "diabetes") to widen it.
        context=TabArenaContext(),
        task_subset=TaskSubset(dataset_names=["blood-transfusion-service-center"], subset="lite"),
        experiment_bundle=TabArenaV0pt1ExperimentBundle(),
        path_setup=_path_setup(),
        # Modest resources for a laptop (defaults are 8 CPU / 32 GB / 1h).
        resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=4, memory_limit=8, time_limit=600),
        scheduler_setup=LocalSequentialSetup(continue_on_error=True),
        # No foundation models selected, so there is nothing to prefetch.
        prefetch_model_weights=False,
    )

    commands = plan.setup_jobs()

    if RUN_NOW:
        # Run each group's command sequentially. `shell=True` mirrors how the SLURM `sbatch`
        # commands are emitted as strings; keep `WORKSPACE` free of spaces.
        for cmd in commands:
            print(f"\n>>> {cmd}")
            subprocess.run(cmd, shell=True, check=False)  # noqa: S602


def evaluate() -> None:
    """Build the TabArena-v0.1 leaderboard from the local run's results (reads results from disk)."""
    config = TabArenaEvalConfig(
        benchmark_name=BENCHMARK_NAME,
        output_dir=_path_setup().get_output_path(BENCHMARK_NAME),
        methods=[
            EvalMethod("RandomForest", result_suffix=" [Rerun]"),
            EvalMethod("Linear", result_suffix=" [Rerun]"),
        ],
        figure_output_dir=Path(__file__).parent / "eval_output" / BENCHMARK_NAME,
        subsets=[["lite"]],
    )
    run_eval(config)


MODES = {"setup": setup, "eval": evaluate}
DEFAULT_MODE = "setup"  # bare invocation (no mode arg) runs this

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the setup or eval half of this benchmark.")
    parser.add_argument("mode", nargs="?", default=DEFAULT_MODE, choices=list(MODES))
    MODES[parser.parse_args().mode]()
