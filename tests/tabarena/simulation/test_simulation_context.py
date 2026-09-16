from __future__ import annotations

import pandas as pd

from tabarena.simulation.context_artificial import load_repo_artificial
from tabarena.utils import task_to_tid_fold


def test_get_tasks_as_dataset_fold():
    """``get_tasks(as_dataset_fold=True)`` maps every task through the tid -> dataset dict."""
    context = load_repo_artificial()._zeroshot_context
    tasks = context.get_tasks()
    assert len(tasks) == 6
    tid_to_dataset = {tid: dataset for dataset, tid in context.dataset_to_tid_dict.items()}
    expected = [(tid_to_dataset[tid], fold) for tid, fold in (task_to_tid_fold(task=task) for task in tasks)]
    assert context.get_tasks(as_dataset_fold=True) == expected
    assert context.get_tasks(datasets=["ada"], as_dataset_fold=True) == [("ada", 0), ("ada", 1), ("ada", 2)]
    assert context._task_to_dataset_fold(tasks[0]) == expected[0]


def test_metrics_filtered_ranks_match_full_frame_without_building_it():
    """A filtered `metrics()` call ranks only the selected rows and returns exactly the rows of the
    full ranked frame, while leaving the full frame unbuilt.
    """
    repo = load_repo_artificial()
    zsc = repo._zeroshot_context
    assert zsc.__dict__.get("_df_configs_ranked") is None
    configs = repo.configs()[:1]
    tasks = [("abalone", 0), ("ada", 2)]
    filtered = {
        "t": repo.metrics(tasks=tasks),
        "c": repo.metrics(configs=configs, set_index=False),
        "tc": repo.metrics(tasks=tasks, configs=configs),
        "empty": repo.metrics(tasks=tasks, configs=["not_a_config"]),
    }
    assert zsc.__dict__.get("_df_configs_ranked") is None
    full = zsc.df_configs_ranked  # builds the full frame
    assert zsc.__dict__.get("_df_configs_ranked") is not None
    reference = {
        "t": repo.metrics(tasks=tasks),
        "c": repo.metrics(configs=configs, set_index=False),
        "tc": repo.metrics(tasks=tasks, configs=configs),
        "empty": repo.metrics(tasks=tasks, configs=["not_a_config"]),
    }
    for key in filtered:
        pd.testing.assert_frame_equal(filtered[key], reference[key])
    assert list(filtered["c"].columns) == list(full.columns)
    assert len(filtered["empty"]) == 0
