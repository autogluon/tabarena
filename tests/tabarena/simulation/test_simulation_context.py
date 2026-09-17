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


def test_context_drops_resource_columns_by_default():
    """The compute-resource result columns are left out of the context's frames unless asked for;
    everything else is unchanged.
    """
    from tabarena.simulation.simulation_context import ZeroshotSimulatorContext

    base = load_repo_artificial()._zeroshot_context
    df_configs = base.df_configs.drop(columns=["task"]).copy()
    df_configs["num_cpus"] = 8
    df_configs["num_gpus"] = 0
    df_configs["disk_usage"] = 12345
    df_configs["kept_extra"] = 1.5
    df_metadata = base.df_metadata

    context = ZeroshotSimulatorContext(df_configs=df_configs, df_metadata=df_metadata)
    for c in ZeroshotSimulatorContext.DROPPED_RESULT_COLUMNS:
        assert c not in context.df_configs.columns
    assert "kept_extra" in context.df_configs.columns
    assert context.drop_columns == ZeroshotSimulatorContext.DROPPED_RESULT_COLUMNS
    # the caller's frame is not modified
    assert "num_cpus" in df_configs.columns

    kept = ZeroshotSimulatorContext(df_configs=df_configs, df_metadata=df_metadata, drop_columns=())
    assert {"num_cpus", "num_gpus", "disk_usage"} <= set(kept.df_configs.columns)
    pd.testing.assert_frame_equal(
        kept.df_configs.drop(columns=list(ZeroshotSimulatorContext.DROPPED_RESULT_COLUMNS)), context.df_configs
    )
    pd.testing.assert_frame_equal(
        kept.df_configs_ranked["rank"].to_frame(), context.df_configs_ranked["rank"].to_frame()
    )

    # frames without the columns are untouched
    plain = ZeroshotSimulatorContext(df_configs=base.df_configs.drop(columns=["task"]), df_metadata=df_metadata)
    pd.testing.assert_frame_equal(plain.df_configs, context.df_configs.drop(columns=["kept_extra"]))
