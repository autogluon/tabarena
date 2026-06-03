"""Evaluate a local ``run_setup_tabarena_v0pt1_local.py`` run.

Counterpart to ``run_setup_tabarena_v0pt1_local.py``: reuse the SAME
``BENCHMARK_NAME`` and ``workspace`` so the raw results are found at
``PathSetup.get_output_path(BENCHMARK_NAME)/data``. Evaluation is identical to the
SLURM variant — it only reads results from disk and is not cluster-specific.
"""

from __future__ import annotations

import sys
from pathlib import Path

from tabarena.evaluation import EvalMethod, TabArenaEvalConfig, run_eval
from tabflow_slurm import PathSetup

BENCHMARK_NAME = "example_tabarena_v0pt1_local"
WORKSPACE = str(Path.home() / "tabarena_local_workspace")

path_setup = PathSetup(
    workspace=WORKSPACE,
    python_path=sys.executable,
)

config = TabArenaEvalConfig(
    benchmark_name=BENCHMARK_NAME,
    output_dir=path_setup.get_output_path(BENCHMARK_NAME),
    methods=[
        EvalMethod("RandomForest", result_suffix=" [Rerun]"),
        EvalMethod("Linear", result_suffix=" [Rerun]"),
    ],
    figure_output_dir=Path(__file__).parent / "eval_output" / BENCHMARK_NAME,
    subsets=[
        ["lite"],
    ],
)

run_eval(config)
