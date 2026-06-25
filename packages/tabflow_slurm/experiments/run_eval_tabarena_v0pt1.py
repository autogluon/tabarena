"""Evaluate the results of a TabArenaBenchmarkPlan run.

Counterpart to ``run_setup_tabarena_v0pt1.py``: reuse the SAME ``benchmark_name``
and ``PathSetup`` (same ``workspace``) used to launch the plan, so the raw
results are found at ``PathSetup.get_output_path(benchmark_name)/data``. Lists the
methods that were run and produces a TabArena-v0.1 leaderboard for them.
"""

from __future__ import annotations

from pathlib import Path

from tabarena.evaluation import EvalMethod, TabArenaEvalConfig, run_eval
from tabflow_slurm import PathSetup

BENCHMARK_NAME = "example_tabarena_v0pt1_29052026"
WORKSPACE = "/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace"

path_setup = PathSetup(
    workspace=WORKSPACE,
    python_path="/home/lennart_priorlabs_ai/.venvs/beyondarena_27052026/bin/python",
)

config = TabArenaEvalConfig(
    benchmark_name=BENCHMARK_NAME,
    output_dir=path_setup.get_output_path(BENCHMARK_NAME),
    methods=[
        EvalMethod("TabPFN-3", result_suffix=" [Rerun]"),
        EvalMethod("Linear", result_suffix=" [Rerun]"),
    ],
    figure_output_dir=Path(__file__).parent / "eval_output" / BENCHMARK_NAME,
    subsets=[
        ["lite"],
    ],
    # Caches default to the library locations (~/.cache/...). If the run used a relocated cache
    # (see run_setup_tabarena_v0pt1.py), pass the SAME object so eval reads the same baselines:
    #     from tabarena.caching import CacheConfig
    #     cache_config=CacheConfig.from_root("/shared/tabarena-caches"),
)

run_eval(config)
