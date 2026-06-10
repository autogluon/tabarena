"""Example code to run TabArena(-Lite) experiments with a custom model."""

from __future__ import annotations

from pathlib import Path

from custom_random_forest_model import get_configs_for_custom_rf

from tabarena.benchmark.experiment import ExperimentBatchRunner
from tabarena.benchmark.task.metadata import TaskMetadataCollection

TABARENA_DIR = str(Path(__file__).parent / "tabarena_out" / "custom_model")
"""Output directory for saving the results and result artifacts from TabArena."""


def run_tabarena_lite_for_custom_rf():
    """Put all the code together to run a TabArenaLite experiment for
    the custom random forest model.
    """
    # TabArena-v0.1 Lite: the first split (r0f0) of each of the 51 datasets.
    task_metadata = TaskMetadataCollection.from_preset("TabArena-v0.1-lite")

    # Gets 1 default and 1 random config = 2 configs
    model_experiments = get_configs_for_custom_rf(num_random_configs=1)

    runner = ExperimentBatchRunner(expname=TABARENA_DIR, task_metadata=task_metadata)
    runner.run_all(methods=model_experiments)


if __name__ == "__main__":
    run_tabarena_lite_for_custom_rf()
