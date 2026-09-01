"""`compare` forwards its figure options rather than fixing them."""

from __future__ import annotations

import pandas as pd
import pytest

from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.evaluation.leaderboard_reporter import LeaderboardReporter
from tabarena.nips2025_utils import compare as compare_module


@pytest.fixture
def task_metadata() -> TaskMetadataCollection:
    return TaskMetadataCollection.from_legacy_df(
        pd.DataFrame(
            {
                "dataset": ["ds"],
                "name": ["ds"],
                "tid": [0],
                "problem_type": ["binary"],
                "n_folds": [2],
                "n_repeats": [1],
                "n_features": [3],
                "n_classes": [2],
                "NumberOfInstances": [150],
                "n_samples_train_per_fold": [100],
                "n_samples_test_per_fold": [50],
                "target_feature": ["t"],
            }
        )
    )


@pytest.fixture
def df_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["a", "b", "a", "b"],
            "dataset": ["ds"] * 4,
            "fold": [0, 0, 1, 1],
            "metric_error": [0.1, 0.2, 0.15, 0.25],
            "metric": ["roc_auc"] * 4,
            "problem_type": ["binary"] * 4,
            "time_train_s": [1.0] * 4,
            "time_infer_s": [0.1] * 4,
        }
    )


@pytest.fixture
def captured_eval_kwargs(monkeypatch) -> dict:
    """Stop at the `eval` call: what matters here is which options reach it."""
    captured: dict = {}

    def fake_eval(self, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(LeaderboardReporter, "eval", fake_eval)
    return captured


def test_the_time_figure_is_off_unless_asked_for(df_results, task_metadata, captured_eval_kwargs):
    compare_module.compare(df_results=df_results, output_dir=None, task_metadata=task_metadata)

    assert captured_eval_kwargs["plot_times"] is False


def test_the_time_figure_can_be_asked_for(df_results, task_metadata, captured_eval_kwargs):
    compare_module.compare(df_results=df_results, output_dir=None, task_metadata=task_metadata, plot_times=True)

    assert captured_eval_kwargs["plot_times"] is True
