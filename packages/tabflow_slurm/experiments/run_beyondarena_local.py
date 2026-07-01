"""Local, sequential variant of ``run_beyondarena.py`` (no SLURM): `setup` + `eval` in one file.

    python experiments/run_beyondarena_local.py setup   # generate (+ run, if RUN_NOW)
    python experiments/run_beyondarena_local.py eval      # leaderboard of the local run

`setup` runs a couple of CPU baselines sequentially on the smallest BeyondArena dataset so the demo
is fast on any laptop/VM. `eval` builds a **self-contained** leaderboard over just this run — there
is no full-suite run available locally, so the fillna/calibration references are disabled
(``imputed_model_name`` / ``reference_model_name`` set to ``None``); see ``run_beyondarena.py`` for
the two-run contender-vs-suite comparison. `setup` and `eval` share ``BENCHMARK_NAME`` + ``WORKSPACE``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabarena.contexts import BeyondArenaContext
from tabarena.evaluation import BenchmarkRun, BeyondArenaEvalConfig, run_beyond_arena_eval
from tabflow_slurm import (
    BeyondArenaResourcesSetup,
    LocalSequentialSetup,
    ModelJob,
    PathSetup,
    TabArenaBenchmarkPlan,
)

# ── Shared identity — the ONE place these live; setup + eval both read them ──
BENCHMARK_NAME = "example_beyondarena_local"
WORKSPACE = str(Path.home() / "tabarena_local_workspace")
PYTHON_PATH = sys.executable  # use the interpreter running this script for the subprocess fits
RUN_NOW = True  # set False to only generate the job JSON + print the run command
LOCAL_MODELS = ["RandomForest", "Linear"]  # the baselines this demo runs + evaluates


def _path_setup() -> PathSetup:
    return PathSetup(workspace=WORKSPACE, python_path=PYTHON_PATH)


def setup() -> None:
    """Generate the local job JSON and (if ``RUN_NOW``) run every item sequentially."""
    plan = TabArenaBenchmarkPlan(
        benchmark_name=BENCHMARK_NAME,
        model_jobs=[
            ModelJob(models=("RandomForest", 0), name="cpu"),  # default config only
            ModelJob(models=("Linear", 1), name="cpu"),  # default + 1 random config
        ],
        # Scope to the smallest dataset (155 rows) so the demo is fast — the full BeyondArena suite
        # is 142 datasets. Add more names to widen it; only the selected datasets are downloaded.
        context=BeyondArenaContext(),
        task_subset=TaskSubset(dataset_names=["hepatitis_survival_prediction"], subset="lite"),
        experiment_bundle=BeyondArenaExperimentBundle(),
        path_setup=_path_setup(),
        # Modest resources for a laptop (defaults are all-CPUs / all-memory / 4h).
        resources_setup=BeyondArenaResourcesSetup(num_cpus=4, memory_limit=8, time_limit=600),
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
    """Build a self-contained BeyondArena leaderboard over just the local run."""
    config = BeyondArenaEvalConfig(
        runs=[
            BenchmarkRun(
                benchmark_name=BENCHMARK_NAME,
                output_dir=_path_setup().get_output_path(BENCHMARK_NAME),
                models=LOCAL_MODELS,
            ),
        ],
        figure_output_dir=Path(__file__).parent / "eval_output" / BENCHMARK_NAME,
        subsets_to_evaluate=[[]],  # full only; add e.g. ["tiny"] if the scoped datasets match it
        # No full-suite run locally, so disable the fillna/calibration references that would
        # otherwise require baselines (`RF (default)` / `XGB (default)`) absent from this run.
        imputed_model_name=None,
        reference_model_name=None,
    )
    run_beyond_arena_eval(config)


MODES = {"setup": setup, "eval": evaluate}
DEFAULT_MODE = "setup"  # bare invocation (no mode arg) runs this

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the setup or eval half of this benchmark.")
    parser.add_argument("mode", nargs="?", default=DEFAULT_MODE, choices=list(MODES))
    MODES[parser.parse_args().mode]()
