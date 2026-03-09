"""Example code to run TabArena(-Lite) experiments with a custom model."""

from __future__ import annotations

from pathlib import Path

import openml

from tabarena.benchmark.experiment import run_experiments_new

from examples.benchmarking.custom_tabarena_model_fs.custom_fs_model import get_configs_for_custom_model_fs

TABARENA_DIR = str(Path(__file__).parent / "tabarena_out" / "Boruta_LightGBM")
"""Output directory for saving the results and result artifacts from TabArena."""


def run_tabarena_lite_for_custom_model_fs():
    """Put all the code together to run a TabArenaLite experiment for
    the custom random forest model.
    """
    # Get all tasks from TabArena-v0.1
    task_ids = openml.study.get_suite("tabarena-v0.1").tasks

    # Gets 1 default and 1 random config = 2 configs
    model_experiments = get_configs_for_custom_model_fs(num_random_configs=1)

    run_experiments_new(
        output_dir=TABARENA_DIR,
        model_experiments=model_experiments,
        tasks=task_ids,
        repetitions_mode="TabArena-Lite",
    )


if __name__ == "__main__":
    run_tabarena_lite_for_custom_model_fs()
