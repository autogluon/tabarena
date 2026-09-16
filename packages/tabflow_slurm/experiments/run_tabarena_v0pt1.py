"""Run the TabArena-v0.1 benchmark on a full GCP cluster node: `setup` + `eval` in one file.

ONE file, TWO subcommands. `setup` and `eval` share the same ``BENCHMARK_NAME``, ``PathSetup``
(``WORKSPACE`` + ``PYTHON_PATH``), and ``CACHE_CONFIG`` — defined once below — so the launch and
the evaluation can never drift apart::

    python experiments/run_tabarena_v0pt1.py setup   # generate + print the sbatch command(s)
    python experiments/run_tabarena_v0pt1.py eval     # build the leaderboard from the results

``--scheduler skypilot`` (or ``skypilot-pool``) runs the same plan as SkyPilot managed jobs instead
of SLURM (see ``run_tabpfn3.py`` for the prerequisites); ``eval`` then syncs the bucket's results
into the workspace first.

`setup` composes a ``TabArenaV0pt1BenchmarkPlan`` (a ``TabArenaBenchmarkPlan`` pre-wired with the
TabArena-v0.1 building blocks) and calls ``setup_jobs()`` to launch several models
with different per-model hardware on one shared default setup: TabPFN-3 on a GPU node and Linear
on a CPU node. The differing ``num_gpus`` puts them in two groups, so ``setup_jobs()`` emits two
``sbatch`` commands (one GPU run, one CPU run). Run it on the head node, then run the printed
commands. When the jobs finish, run the `eval` subcommand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tabarena.benchmark.task.metadata import TaskSubset
from tabarena.contexts import TabArenaContext
from tabarena.evaluation import EvalMethod, TabArenaEvalConfig, run_eval
from tabflow_slurm import (
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    SchedulerSetup,
    SkyPilotSetup,
    TabArenaV0pt1BenchmarkPlan,
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

SCHEDULERS = ("slurm", "skypilot", "skypilot-pool")


def _path_setup() -> PathSetup:
    return PathSetup(workspace=WORKSPACE, python_path=PYTHON_PATH)


def _scheduler_setup(kind: str) -> tuple[SchedulerSetup, int]:
    """The scheduler for ``kind`` and the VRAM in GB of its GPU (for ``fake_memory_for_estimates``)."""
    if kind == "slurm":
        return GCPSlurmSetup(), 96  # RTX PRO 6000
    return SkyPilotSetup(secrets=("HF_TOKEN",), use_pool=kind == "skypilot-pool"), 96  # RTX PRO 6000 as well


def setup(scheduler: str = "slurm") -> None:
    """Generate the job files and emit the launch command(s) for the run."""
    scheduler_setup, vram_gb = _scheduler_setup(scheduler)
    # TabArenaV0pt1BenchmarkPlan pre-wires the v0.1 building blocks (TabArenaContext,
    # TabArenaV0pt1ExperimentBundle, GCPSlurmSetup, TabArenaV0pt1ResourcesSetup); any of them
    # can still be passed explicitly to override — as done for `context` here, which carries
    # the shared CACHE_CONFIG.
    plan = TabArenaV0pt1BenchmarkPlan(
        benchmark_name=BENCHMARK_NAME,
        model_jobs=[
            # GPU model: override the base (CPU-only) resources to request a GPU.
            ModelJob(
                models=("TabPFN-3", 0), name="gpu", resources={"num_gpus": 1, "fake_memory_for_estimates": vram_gb}
            ),
            # CPU model: no resource override, so it runs on the base CPU resources.
            ModelJob(models=("Linear", 1), name="cpu"),
        ],
        # The TabArena-v0.1 context owns the task metadata + subset predicates; `task_subset`
        # scopes `context.build_jobs` (here `subset="lite"` keeps each dataset's first split).
        context=TabArenaContext(cache_config=CACHE_CONFIG),
        task_subset=TaskSubset(subset="lite"),
        path_setup=_path_setup(),
        scheduler_setup=scheduler_setup,
    )
    plan.setup_jobs()


def evaluate(scheduler: str = "slurm") -> None:
    """Build the TabArena-v0.1 leaderboard from the run's cached results (synced first for SkyPilot)."""
    _scheduler_setup(scheduler)[0].sync_results_to_local(
        path_setup=_path_setup(), benchmark_name=BENCHMARK_NAME, force=True
    )
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
    parser.add_argument("--scheduler", choices=SCHEDULERS, default="slurm", help="Where the jobs run.")
    args = parser.parse_args()
    MODES[args.mode](args.scheduler)
