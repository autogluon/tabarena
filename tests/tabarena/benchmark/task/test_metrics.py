"""Metric names are canonical wherever tasks and results are produced (``rmse``, not AutoGluon's alias
``root_mean_squared_error``), so tasks and results from every source join on one name.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from tabarena.benchmark.experiment.experiment_runner import ExperimentRunner
from tabarena.benchmark.task import UserTask
from tabarena.benchmark.task.data_foundry.adapter import DEFAULT_EVAL_METRICS
from tabarena.benchmark.task.metadata import TabArenaTaskMetadata
from tabarena.benchmark.task.metrics import (
    DEFAULT_EVAL_METRIC_BY_PROBLEM_TYPE,
    EVAL_METRIC_ALIASES,
    normalize_eval_metric,
)

_METADATA_DIR = Path(__import__("tabarena").__file__).parent / "benchmark" / "task" / "metadata" / "sources" / "data"
_ALIAS = "root_mean_squared_error"


def test_normalize_eval_metric():
    assert normalize_eval_metric(_ALIAS) == "rmse"
    assert normalize_eval_metric("rmse") == "rmse"
    assert normalize_eval_metric("roc_auc") == "roc_auc"


@pytest.mark.parametrize("path", sorted(_METADATA_DIR.glob("*_tasks_metadata.csv")), ids=lambda p: p.name)
def test_shipped_task_metadata_uses_canonical_metric_names(path):
    metrics = set(pd.read_csv(path, usecols=["eval_metric"])["eval_metric"].dropna())
    assert metrics, path
    assert not metrics & set(EVAL_METRIC_ALIASES), (
        f"{path.name} stores metric aliases {metrics & set(EVAL_METRIC_ALIASES)}"
    )


def test_default_metric_tables_are_canonical():
    for metric in DEFAULT_EVAL_METRIC_BY_PROBLEM_TYPE.values():
        assert normalize_eval_metric(metric) == metric
    for metrics in DEFAULT_EVAL_METRICS.values():
        assert all(normalize_eval_metric(m) == m for m in metrics)


def test_task_metadata_canonicalizes_alias_on_construction():
    row = pd.read_csv(_METADATA_DIR / "BeyondArena_tasks_metadata.csv").query("problem_type == 'regression'").iloc[0]
    row["eval_metric"] = _ALIAS
    assert TabArenaTaskMetadata.from_row(row).eval_metric == "rmse"


@pytest.fixture
def regression_task_with_alias(tmp_path) -> UserTask:
    """A cached regression UserTask whose OpenML task records the alias as its evaluation measure."""
    X, y = make_regression(n_samples=90, n_features=4, random_state=0)
    dataset = pd.DataFrame(X, columns=[f"num_{i}" for i in range(4)]).assign(target=y)
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.33, random_state=0)
    task = UserTask(task_name="toy_regression_alias", task_cache_path=tmp_path)
    oml_task = task.create_local_openml_task(
        dataset=dataset,
        target_feature="target",
        problem_type="regression",
        splits={0: {0: (train_idx, test_idx)}},
        eval_metric=_ALIAS,
    )
    task.save_local_openml_task(oml_task)
    return task


def test_task_wrapper_and_its_metadata_report_canonical_name(regression_task_with_alias):
    wrapper = regression_task_with_alias.load()
    assert wrapper.eval_metric == "rmse"
    metadata = wrapper.compute_metadata(
        tabarena_task_name=regression_task_with_alias.tabarena_task_name,
        task_id_str=regression_task_with_alias.task_id_str,
    )
    assert metadata.eval_metric == "rmse"


def test_experiment_runner_records_canonical_name(regression_task_with_alias):
    wrapper = regression_task_with_alias.load()
    runner = ExperimentRunner(
        method_cls=object, task=wrapper, fold=0, task_name="toy", method="m", eval_metric_name=_ALIAS, warmup=False
    )
    assert runner.eval_metric_name == "rmse"
