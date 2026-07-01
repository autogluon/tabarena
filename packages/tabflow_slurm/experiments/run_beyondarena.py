"""Run the BeyondArena (data-foundry) benchmark on a GCP cluster node: `setup` + `eval` in one file.

ONE file, TWO subcommands, sharing the new run's ``BENCHMARK_NAME`` / ``PathSetup`` /
``CONTENDER_MODEL`` (defined once below)::

    python experiments/run_beyondarena.py setup   # launch the new contender run
    python experiments/run_beyondarena.py eval      # leaderboard: contender vs. the full suite

`setup` launches a fresh run of a single contender (``CONTENDER_MODEL``). The tasks come from the
Data Foundry ``BeyondArena`` collection, which ``BeyondArenaContext`` owns: it loads reference
metadata (no downloads); the benchmark setup later materializes (downloads + converts) only the
surviving datasets on this head node. Scope a subset with ``task_subset=TaskSubset(dataset_names=[...])``.

`eval` combines two runs into one leaderboard: an existing full-suite run (the baselines, which also
supply the fillna/calibration references) and the new contender run from `setup`. Because each run
caches under its own ``benchmark_name``, the contender does not collide with the old run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
from tabarena.contexts import BeyondArenaContext
from tabarena.evaluation import BenchmarkRun, BeyondArenaEvalConfig, run_beyond_arena_eval
from tabflow_slurm import (
    BeyondArenaResourcesSetup,
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    TabArenaBenchmarkPlan,
)

# ── Shared identity — the ONE place these live; setup + eval both read them ──
BENCHMARK_NAME = "example_beyondarena_31052026"  # the NEW contender run's cache name
WORKSPACE = "/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace"
PYTHON_PATH = "/home/lennart_priorlabs_ai/.venvs/beyondarena_27052026/bin/python"
CONTENDER_MODEL = "TabPFN-3"  # the model this run adds on top of the existing full suite

# Existing full-suite run that supplies the baselines + fillna/calibration references. `eval` reads
# its raw results from ``OLD_OUTPUT_DIR/data`` under its own cache name. Adjust to your completed run.
OLD_BENCHMARK_NAME = "beyond_iid_benchmark_2026"
OLD_OUTPUT_DIR = "/home/lennart_priorlabs_ai/workspace/benchmarking/output/beyond_iid_benchmark_2026_migrated"
FULL_MODEL_SUITE = [
    "Linear",
    "RandomForest",
    "ExtraTrees",
    "CatBoost",
    "LightGBM",
    "XGBoost",
    "RealMLP",
    "TabM",
    "TabDPT",
    "TabPFN-2.6",
    "TabICLv2",
]
SUBSETS = [
    [],  # full
    ["random"],
    ["temporal"],
    ["grouped"],
    ["tiny"],
    ["small"],
    ["medium"],
    ["large"],
    ["low-dim"],
    ["high-dim"],
    ["text"],
    ["high-cardinality"],
]


def _path_setup() -> PathSetup:
    return PathSetup(workspace=WORKSPACE, python_path=PYTHON_PATH)


def setup() -> None:
    """Generate the job JSON and emit the ``sbatch`` command(s) for the contender run."""
    plan = TabArenaBenchmarkPlan(
        benchmark_name=BENCHMARK_NAME,
        model_jobs=[
            ModelJob(
                models=(CONTENDER_MODEL, 0),
                name="gpu",
                resources={
                    "num_gpus": 1,
                    "fake_memory_for_estimates": 80,  # we have an 80 GB VRAM GPU
                    "time_limit": 3600 * 12,  # higher time limit for a TFM job, as for TabICL
                },
            ),
        ],
        # No `task_subset` runs the full suite; scope it with e.g.
        # `task_subset=TaskSubset(dataset_names=[...])` so only those datasets are fetched.
        context=BeyondArenaContext(),
        experiment_bundle=BeyondArenaExperimentBundle(),
        path_setup=_path_setup(),
        resources_setup=BeyondArenaResourcesSetup(),
        scheduler_setup=GCPSlurmSetup(bundle_size=1),
    )
    plan.setup_jobs()


def evaluate() -> None:
    """Combine the existing full-suite run and the new contender run into one leaderboard."""
    config = BeyondArenaEvalConfig(
        runs=[
            # Existing full suite — its `RF (default)` / `XGB (default)` also feed fillna/calibration.
            BenchmarkRun(
                benchmark_name=OLD_BENCHMARK_NAME,
                output_dir=OLD_OUTPUT_DIR,
                models=FULL_MODEL_SUITE,
                only_load_cache=True,
            ),
            # New contender run from `setup` (own cache name, so no collision with the old run).
            BenchmarkRun(
                benchmark_name=BENCHMARK_NAME,
                output_dir=_path_setup().get_output_path(BENCHMARK_NAME),
                models=[CONTENDER_MODEL],
                only_load_cache=True,
            ),
        ],
        figure_output_dir=Path(__file__).parent / "eval_output" / BENCHMARK_NAME,
        subsets_to_evaluate=SUBSETS,
        # Highlight the contender in the result plots: its own line in the per-family plot,
        # star-marked / top-layered in the per-model plot.
        contender_models=[CONTENDER_MODEL],
    )
    run_beyond_arena_eval(config)


MODES = {"setup": setup, "eval": evaluate}
DEFAULT_MODE = "setup"  # bare invocation (no mode arg) runs this

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the setup or eval half of this benchmark.")
    parser.add_argument("mode", nargs="?", default=DEFAULT_MODE, choices=list(MODES))
    MODES[parser.parse_args().mode]()
