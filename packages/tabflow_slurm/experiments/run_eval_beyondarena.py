"""Evaluate BeyondArena results, comparing an existing full-suite run against a new TabPFN-3 run.

Counterpart to ``run_eval_tabarena_v0pt1.py`` for the BeyondArena (data-foundry) benchmark, and the
eval companion to ``run_setup_beyondarena.py``. Each :class:`BenchmarkRun` points at a completed
benchmark output dir + its cache ``benchmark_name`` + the models to pull from it; the runner
post-processes each run into the cache and combines them into one leaderboard over the BeyondArena
subsets.

Here we compare two runs:

* ``OLD_RUN`` — an existing full-model-suite BeyondArena run (the 11 baselines).
* ``NEW_RUN`` — a fresh run with only ``TabPFN-3`` (e.g. from ``run_setup_beyondarena.py``).

Because each run caches under its own ``benchmark_name``, TabPFN-3 from the new run does not collide
with anything in the old run. Task metadata is loaded once from the self-contained committed
BeyondArena reference CSV (no warehouse merge).

Note: ``fillna``/calibration baselines (``RF (default)`` / ``XGB (default)``) come from the OLD
full-suite run — keep it in ``runs`` so the leaderboard can impute/calibrate.
"""

from __future__ import annotations

from pathlib import Path

from tabarena.evaluation import BenchmarkRun, BeyondArenaEvalConfig, run_beyond_arena_eval
from tabflow_slurm import PathSetup

# Full BeyondArena baseline suite (matches the legacy run_eval.py model list).
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

# Same subsets the legacy BeyondArena eval reports.
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

path_setup = PathSetup(
    workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
    python_path="/home/lennart_priorlabs_ai/.venvs/beyondarena_27052026/bin/python",
)

# Existing full-suite run. ``benchmark_name`` is its cache artifact name; ``output_dir`` is where its
# raw results live (``<output_dir>/data``). Adjust to point at your completed run.
OLD_BENCHMARK_NAME = "beyond_iid_benchmark_2026"
OLD_OUTPUT_DIR = "/home/lennart_priorlabs_ai/workspace/benchmarking/output/beyond_iid_benchmark_2026_migrated"

# New TabPFN-3-only run (e.g. launched via run_setup_beyondarena.py).
NEW_BENCHMARK_NAME = "example_beyondarena_31052026"

config = BeyondArenaEvalConfig(
    runs=[
        BenchmarkRun(
            benchmark_name=OLD_BENCHMARK_NAME,
            output_dir=OLD_OUTPUT_DIR,
            models=FULL_MODEL_SUITE,
            only_load_cache=True,
        ),
        BenchmarkRun(
            benchmark_name=NEW_BENCHMARK_NAME,
            output_dir=path_setup.get_output_path(NEW_BENCHMARK_NAME),
            models=["TabPFN-3"],
            only_load_cache=True,
        ),
    ],
    figure_output_dir=Path(__file__).parent / "eval_output" / NEW_BENCHMARK_NAME,
    subsets_to_evaluate=SUBSETS,
    # Highlight TabPFN-3 in the result plots: its own line in the per-family plot,
    # star-marked / top-layered in the per-model plot.
    contender_models=["TabPFN-3"],
)

if __name__ == "__main__":
    run_beyond_arena_eval(config)
