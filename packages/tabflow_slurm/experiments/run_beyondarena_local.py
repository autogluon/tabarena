"""Local, sequential variant of ``run_beyondarena.py`` (no SLURM): `setup` + `eval` in one file.

    python experiments/run_beyondarena_local.py setup   # generate (+ run, if RUN_NOW)
    python experiments/run_beyondarena_local.py eval      # leaderboard: local run vs. the uploaded suite

`setup` runs a couple of CPU baselines sequentially on the smallest BeyondArena dataset so the demo
is fast on any laptop/VM. `eval` builds the leaderboard the same way ``run_beyondarena.py`` does — the
**baselines come from the context** (the uploaded ``beyond_method_metadata_collection``), so this
compares the local run against the real suite rather than being self-contained. Because the local
models (``RandomForest`` / ``Linear``) are themselves uploaded baselines, this run is marked via
``RESULT_SUFFIX`` (rendered ``RandomForest [local]`` etc.) to stay distinct, and ``only_valid_tasks=True``
scopes the leaderboard to the one tiny task it ran. Needs network access to fetch the (tiny, <1 MB
each) cached baseline results. `setup` and `eval` share ``BENCHMARK_NAME`` + ``WORKSPACE``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabarena.contexts import BeyondArenaContext
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
RESULT_SUFFIX = " [local]"  # keeps this run distinct from the same-named uploaded baselines


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
    """Leaderboard: the local run vs. the uploaded BeyondArena baselines (from the context).

    We post-process the local run's raw results and register them via ``extra_methods=``; the
    baselines are the methods ``BeyondArenaContext`` already knows. ``only_valid_tasks=True`` scopes
    the leaderboard to the one tiny task this demo ran, so it compares the local models against the
    uploaded suite on exactly that task (fillna/calibration use the context's defaults, now that the
    reference baselines are present).
    """
    from tabarena.evaluation._eval_common import (
        MethodArtifact,
        post_process_to_results,
        resolve_ag_name,
        subset_label,
    )
    from tabarena.evaluation.beyond_metadata import load_beyond_task_metadata_collection

    task_metadata = load_beyond_task_metadata_collection("BeyondArena")

    # Post-process this run's raw ``results.pkl`` into the cache under its own suite name, marked with
    # RESULT_SUFFIX so the local RandomForest / Linear stay distinct from the uploaded baselines.
    local = post_process_to_results(
        [
            MethodArtifact(
                ag_name=resolve_ag_name(model),
                path_raw=_path_setup().get_output_path(BENCHMARK_NAME) / "data",
                suite=BENCHMARK_NAME,
                result_suffix=RESULT_SUFFIX,
            )
            for model in LOCAL_MODELS
        ],
        task_metadata=task_metadata,
    )

    context = BeyondArenaContext(
        extra_methods=local.to_method_metadata_lst(),
        only_valid_tasks=True,
    )

    figure_output_dir = Path(__file__).parent / "eval_output" / BENCHMARK_NAME
    for subset in [[]]:  # full only (== the one task ran, via only_valid_tasks); add e.g. ["tiny"]
        label = subset_label(sorted(subset))
        leaderboard = context.compare(output_dir=figure_output_dir / "subsets" / label, subset=subset or None)
        print(f"\n##### Leaderboard [{label}]")
        print(leaderboard.to_markdown(index=False))


MODES = {"setup": setup, "eval": evaluate}
DEFAULT_MODE = "setup"  # bare invocation (no mode arg) runs this

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the setup or eval half of this benchmark.")
    parser.add_argument("mode", nargs="?", default=DEFAULT_MODE, choices=list(MODES))
    MODES[parser.parse_args().mode]()
