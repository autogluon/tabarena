from __future__ import annotations

import pandas as pd
import pytest

from tabarena.evaluation._fillna import fillna_metrics


def _frames(key_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``m_a`` is complete over both tasks; ``m_b`` is missing ``(d1, fold 1)``."""
    df_to_fill = pd.DataFrame(
        {
            "dataset": ["d1", "d1", "d1"],
            "fold": [0, 1, 0],
            key_col: ["m_a", "m_a", "m_b"],
            "metric_error": [0.1, 0.2, 0.3],
        }
    )
    df_fillna = pd.DataFrame(
        {
            "dataset": ["d1", "d1"],
            "fold": [0, 1],
            key_col: ["fb", "fb"],
            "metric_error": [0.9, 0.8],
        }
    )
    return df_to_fill, df_fillna


@pytest.mark.parametrize("key_col", ["method", "framework"])
def test_fillna_marks_and_fills_missing_rows(key_col: str):
    df_to_fill, df_fillna = _frames(key_col)
    out = fillna_metrics(df_to_fill, df_fillna, key_col=key_col)
    out = out.set_index(["dataset", "fold", key_col]).sort_index()

    # every key x (dataset, fold) task now exists (2 keys x 2 tasks)
    assert len(out) == 4
    # the missing (d1, fold 1, m_b) row is imputed from the fallback's value (0.8)
    assert out.loc[("d1", 1, "m_b"), "imputed"]
    assert out.loc[("d1", 1, "m_b"), "metric_error"] == 0.8
    # rows already present keep their own value and are not imputed
    assert not out.loc[("d1", 0, "m_b"), "imputed"]
    assert out.loc[("d1", 0, "m_b"), "metric_error"] == 0.3
    assert not out["imputed"].loc[("d1", slice(None), "m_a")].any()


def test_fillna_preserves_intrinsic_columns():
    df_to_fill, df_fillna = _frames("method")
    df_to_fill["method_type"] = df_to_fill["method"].map({"m_a": "config", "m_b": "baseline"})
    df_fillna["method_type"] = "config"  # the fallback's value must NOT leak into the imputed m_b row

    out = fillna_metrics(df_to_fill, df_fillna, key_col="method", preserve_columns=["method_type"])
    out = out.set_index(["dataset", "fold", "method"])
    # imputed row keeps m_b's own intrinsic value, not the fallback's
    assert out.loc[("d1", 1, "m_b"), "method_type"] == "baseline"


def test_fillna_rejects_non_constant_preserve_column():
    df_to_fill, df_fillna = _frames("method")
    df_to_fill["method_type"] = ["config", "baseline", "config"]  # m_a has two values -> invalid
    with pytest.raises(AssertionError):
        fillna_metrics(df_to_fill, df_fillna, key_col="method", preserve_columns=["method_type"])
