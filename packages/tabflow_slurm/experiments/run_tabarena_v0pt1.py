"""Run the TabArena-v0.1 benchmark on a full GCP cluster node: `setup` + `eval` in one file.

ONE file, TWO subcommands. `setup` and `eval` share the same ``BENCHMARK_NAME``, ``PathSetup``
(``WORKSPACE`` + ``PYTHON_PATH``), and ``CACHE_CONFIG`` — defined once below — so the launch and
the evaluation can never drift apart::

    python experiments/run_tabarena_v0pt1.py setup   # generate + print the sbatch command(s)
    python experiments/run_tabarena_v0pt1.py eval     # build the leaderboard from the results

`setup` composes a ``TabArenaBenchmarkPlan`` and calls ``setup_jobs()`` to launch several models
with different per-model hardware on one shared default setup: TabPFN-3 on a GPU node and Linear
on a CPU node. The differing ``num_gpus`` puts them in two groups, so ``setup_jobs()`` emits two
``sbatch`` commands (one GPU run, one CPU run). Run it on the head node, then run the printed
commands. When the jobs finish, run the `eval` subcommand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabarena.contexts import TabArenaContext
from tabarena.evaluation import EvalMethod, TabArenaEvalConfig, run_eval
from tabflow_slurm import (
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    TabArenaBenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

# ── Shared identity — the ONE place these live; setup + eval both read them ──
BENCHMARK_NAME = "example_tabarena_v0pt1_29052026"
WORKSPACE = "/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace"
PYTHON_PATH = "/home/lennart_priorlabs_ai/.venvs/beyondarena_27052026/bin/python"

# Caches default to the library locations (OpenML ``~/.cache/openml``, HuggingFace
# ``~/.cache/huggingface/hub``, TabArena ``~/.cache/tabarena``). To relocate them — e.g. onto shared
# cluster storage so the head node and every worker resolve the same files — set this ONCE here; it
# is passed to both the setup context and the eval config, so they can't disagree:
#     from tabarena.caching import CacheConfig
#     CACHE_CONFIG = CacheConfig.from_root("/shared/tabarena-caches")
CACHE_CONFIG = None


def _path_setup() -> PathSetup:
    return PathSetup(workspace=WORKSPACE, python_path=PYTHON_PATH)


def setup() -> None:
    """Generate the job JSON and emit the ``sbatch`` command(s) for the run."""
    plan = TabArenaBenchmarkPlan(
        benchmark_name=BENCHMARK_NAME,
        model_jobs=[
            # GPU model: override the base (CPU-only) resources to request a GPU.
            ModelJob(models=("TabPFN-3", 0), name="gpu", resources={"num_gpus": 1}),
            # CPU model: no resource override, so it runs on the base CPU resources.
            ModelJob(models=("Linear", 1), name="cpu"),
        ],
        # The TabArena-v0.1 context owns the task metadata + subset predicates; `task_subset`
        # scopes `context.build_jobs` (here `subset="lite"` keeps each dataset's first split).
        context=TabArenaContext(cache_config=CACHE_CONFIG),
        task_subset=TaskSubset(subset="lite"),
        experiment_bundle=TabArenaV0pt1ExperimentBundle(),
        path_setup=_path_setup(),
        resources_setup=TabArenaV0pt1ResourcesSetup(),
        scheduler_setup=GCPSlurmSetup(),
    )
    plan.setup_jobs()


def evaluate() -> None:
    """Build the TabArena-v0.1 leaderboard from the run's cached results."""
    config = TabArenaEvalConfig(
        benchmark_name=BENCHMARK_NAME,
        output_dir=_path_setup().get_output_path(BENCHMARK_NAME),
        methods=[
            EvalMethod("TabPFN-3", result_suffix=" [Rerun]"),
            EvalMethod("Linear", result_suffix=" [Rerun]"),
        ],
        figure_output_dir=Path(__file__).parent / "eval_output" / BENCHMARK_NAME,
        subsets=[["lite"]],
        cache_config=CACHE_CONFIG,
    )
    run_eval(config)


MODES = {"setup": setup, "eval": evaluate}
DEFAULT_MODE = "setup"  # bare invocation (no mode arg) runs this

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the setup or eval half of this benchmark.")
    parser.add_argument("mode", nargs="?", default=DEFAULT_MODE, choices=list(MODES))
    MODES[parser.parse_args().mode]()
