"""Benchmark TabPFN-3 on TabArena-v0.1 on SLURM or SkyPilot: `setup` + `eval` in one file.

ONE file, TWO subcommands, ONE switch. `setup` and `eval` share the same ``BENCHMARK_NAME`` and
``PathSetup`` (``WORKSPACE`` + ``PYTHON_PATH``), so the launch and the evaluation cannot drift, and
``--scheduler`` picks where the jobs run without touching the plan::

    python experiments/run_tabpfn3.py setup                            # GCP SLURM cluster (default)
    python experiments/run_tabpfn3.py setup --scheduler skypilot       # SkyPilot managed jobs, one spot VM each
    python experiments/run_tabpfn3.py setup --scheduler skypilot-pool  # SkyPilot job pool (venv built once per worker)
    python experiments/run_tabpfn3.py eval  --scheduler skypilot       # sync the bucket's results, then the leaderboard

TabPFN-3 is a GPU foundation model without a search space (``NUM_CONFIGS = 0``: the default config
only). Its weights live in the gated Hugging Face repo ``Prior-Labs/tabpfn_3``, so a SkyPilot worker
needs ``HF_TOKEN`` in the launching shell (forwarded as a SkyPilot secret); on SLURM the shared
Hugging Face cache already holds them after the head-node prefetch.

SkyPilot prerequisites: the cluster's ``sky`` CLI on ``PATH`` (it talks to the shared API server, which
resolves ``RTXPRO6000`` to a ``g4-standard-48`` and picks the regions; login nodes preset the endpoint,
elsewhere export it), ``HF_TOKEN`` for the gated weights, and ``gcloud`` for the bucket::

    export SKYPILOT_API_SERVER_ENDPOINT=http://skypilot-api:46580   # only where the login shell does not
    export HF_TOKEN=...
    sky check gcp

Both schedulers land on an RTX PRO 6000 (96 GB), so ``fake_memory_for_estimates`` is 96 either way;
``_scheduler_setup`` still returns the VRAM next to the scheduler so a different card stays a one-line change.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabarena.evaluation import EvalMethod, TabArenaEvalConfig, run_eval
from tabflow_slurm import (
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    SchedulerSetup,
    SkyPilotSetup,
    TabArenaV0pt1BenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

# ── Shared identity — the ONE place these live; setup + eval both read them ──
BENCHMARK_NAME = "tabpfn3_16092026"  # cache key; reused unchanged across relaunches and eval
WORKSPACE = "/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace"
PYTHON_PATH = "/home/lennart_priorlabs_ai/.venvs/tabarena_10082026/bin/python"

MODEL = "TabPFN-3"
NUM_CONFIGS = 0  # foundation model: default config only, no search space

SCHEDULERS = ("slurm", "skypilot", "skypilot-pool")


def _path_setup() -> PathSetup:
    return PathSetup(workspace=WORKSPACE, python_path=PYTHON_PATH)


def _scheduler_setup(kind: str) -> tuple[SchedulerSetup, int]:
    """The scheduler for ``kind`` and the VRAM in GB of the GPU its jobs land on.

    The VRAM feeds ``fake_memory_for_estimates`` so AutoGluon budgets the parallel bagging folds
    against the card instead of the node's RAM. Both SkyPilot modes use the same worker hardware
    (the default ``gpu_accelerator``, an RTX PRO 6000 like the SLURM partition); ``skypilot-pool``
    builds the venv once per pool worker instead of once per job.

    ``bundle_size=1``: one fit per claimed bundle. The worker budgets and records every item on its
    own, so a bigger bundle only shrinks the queue (fewer task objects to list); with 816 splits that
    is not worth the coarser load balancing. ``workers=32`` is the concurrency, SLURM's ``%N``: about
    what the RTX PRO 6000 partition sustains, and 32 spot g4-standard-48 at $1.80/h each.
    """
    if kind == "slurm":
        return GCPSlurmSetup(gpu_partition="gpurtxpro6000flex", bundle_size=1), 96
    sky = SkyPilotSetup(
        bundle_size=1,
        workers=32,
        secrets=("HF_TOKEN",),  # gated TabPFN weights; the value comes from the launching shell
        use_pool=kind == "skypilot-pool",
    )
    return sky, 96


def setup(scheduler: str) -> None:
    """Generate the job files and print the launch command(s) for ``scheduler``."""
    scheduler_setup, vram_gb = _scheduler_setup(scheduler)
    plan = TabArenaV0pt1BenchmarkPlan(
        benchmark_name=BENCHMARK_NAME,
        model_jobs=[
            ModelJob(
                models=(MODEL, NUM_CONFIGS),
                name="gpu",
                resources={"num_gpus": 1, "fake_memory_for_estimates": vram_gb},
            ),
        ],
        task_subset=TaskSubset(),  # the full task set, all splits; TaskSubset(subset="lite") for a first-split trial
        path_setup=_path_setup(),
        experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
        resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),  # auto-detect on the node
        scheduler_setup=scheduler_setup,
    )
    plan.setup_jobs()


def evaluate(scheduler: str) -> None:
    """Build the TabArena-v0.1 leaderboards and figures; a SkyPilot run's results are synced first."""
    scheduler_setup, _ = _scheduler_setup(scheduler)
    # Results written by SkyPilot workers live in the bucket; bring them into the workspace first.
    # A no-op for SLURM, whose nodes write into the workspace directly.
    scheduler_setup.sync_results_to_local(path_setup=_path_setup(), benchmark_name=BENCHMARK_NAME, force=True)
    config = TabArenaEvalConfig(
        benchmark_name=BENCHMARK_NAME,
        methods=[EvalMethod(MODEL, result_suffix=" [Rerun]")],  # TabPFN-3 is a hosted method: keep the two apart
        output_dir=_path_setup().get_output_path(BENCHMARK_NAME),
        figure_output_dir=Path(__file__).parent / "eval_output" / BENCHMARK_NAME,
        subsets=[[], ["binary"], ["multiclass"], ["regression"]],
        figure_file_type=("pdf", "png"),
    )
    run_eval(config)


MODES = {"setup": setup, "eval": evaluate}
DEFAULT_MODE = "setup"  # bare invocation (no mode arg) runs this

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the setup or eval half of this benchmark.")
    parser.add_argument("mode", nargs="?", default=DEFAULT_MODE, choices=list(MODES))
    parser.add_argument(
        "--scheduler",
        choices=SCHEDULERS,
        default="slurm",
        help="Where the jobs run: the GCP SLURM cluster (default), SkyPilot managed jobs on their own spot VMs, "
        "or a SkyPilot job pool.",
    )
    args = parser.parse_args()
    MODES[args.mode](args.scheduler)
