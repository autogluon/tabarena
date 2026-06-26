from __future__ import annotations

import numpy as np
import pandas as pd

from tabarena.evaluation.repo_metrics import RepoMetrics

# Standard metric columns assemble_metrics keeps.
_COLS = ["metric_error", "time_train_s", "time_infer_s", "metric", "problem_type"]


def _df_configs() -> pd.DataFrame:
    # c1 on both folds; c2 only on fold 0 (so it is missing on (d1, fold 1)).
    rows = [
        ("d1", 0, "c1", 0.10, 1.0, 0.1, "acc", "binary", 0.15),
        ("d1", 1, "c1", 0.30, 1.0, 0.1, "acc", "binary", 0.35),
        ("d1", 0, "c2", 0.20, 2.0, 0.2, "acc", "binary", 0.25),
    ]
    return pd.DataFrame(rows, columns=["dataset", "fold", "framework", *_COLS, "metric_error_val"])


def _df_baselines() -> pd.DataFrame:
    rows = [
        ("d1", 0, "RF", 0.50, 0.5, 0.05, "acc", "binary"),
        ("d1", 1, "RF", 0.60, 0.5, 0.05, "acc", "binary"),
    ]
    return pd.DataFrame(rows, columns=["dataset", "fold", "framework", *_COLS])


class _FakeZeroshotContext:
    def __init__(self, df_configs: pd.DataFrame, df_baselines: pd.DataFrame | None):
        self.df_configs = df_configs
        self.df_baselines = df_baselines


class _FakeRepo:
    """Minimal duck-typed repo exposing only what ``assemble_metrics`` reads."""

    def __init__(self, df_configs, df_baselines, config_fallback=None):
        self._zeroshot_context = _FakeZeroshotContext(df_configs, df_baselines)
        self._config_fallback = config_fallback
        self.task_metadata = None

    def datasets(self):
        return list(self._zeroshot_context.df_configs["dataset"].unique())


def _metrics(config_fallback=None) -> RepoMetrics:
    return RepoMetrics(repo=_FakeRepo(_df_configs(), _df_baselines(), config_fallback=config_fallback))


def test_assemble_metrics_concatenates_configs_and_baselines():
    out = _metrics().assemble_metrics()
    assert list(out.index.names) == ["dataset", "fold", "framework"]
    assert set(out.columns) == set(_COLS)  # no fillna -> no imputed column
    # 3 config rows (c1 x2, c2 x1) + 2 baseline rows (RF x2)
    assert len(out) == 5
    assert set(out.index.get_level_values("framework")) == {"c1", "c2", "RF"}
    assert out.loc[("d1", 0, "c1"), "metric_error"] == 0.10


def test_include_metric_error_val_adds_column_and_nans_baselines():
    out = _metrics().assemble_metrics(include_metric_error_val=True)
    assert "metric_error_val" in out.columns
    assert out.loc[("d1", 0, "c1"), "metric_error_val"] == 0.15
    # baselines carry no validation error -> filled with NaN
    assert np.isnan(out.loc[("d1", 0, "RF"), "metric_error_val"])


def test_configs_and_baselines_filters():
    # baselines=[] drops the baseline rows
    only_configs = _metrics().assemble_metrics(baselines=[])
    assert "RF" not in set(only_configs.index.get_level_values("framework"))
    # configs=["c1"] keeps only that config (baselines still included)
    out = _metrics().assemble_metrics(configs=["c1"])
    frameworks = set(out.index.get_level_values("framework"))
    assert "c1" in frameworks and "c2" not in frameworks and "RF" in frameworks


def test_fillna_auto_imputes_missing_config_from_fallback():
    # fillna defaults to "auto" -> enabled because a config_fallback is set.
    out = _metrics(config_fallback="c1").assemble_metrics()
    assert "imputed" in out.columns
    # c2 was missing on (d1, fold 1): imputed from c1's (d1, fold 1) row (metric_error 0.30).
    assert bool(out.loc[("d1", 1, "c2"), "imputed"])
    assert out.loc[("d1", 1, "c2"), "metric_error"] == 0.30
    assert out.loc[("d1", 1, "c2"), "impute_method"] == "c1"
    # present rows are not imputed
    assert not bool(out.loc[("d1", 0, "c1"), "imputed"])
