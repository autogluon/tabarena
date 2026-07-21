from __future__ import annotations

import pandas as pd
import pytest

from tabarena.plot.interactive.pareto_explorer import build_pareto_explorer_html


def _scatter_points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["A", "A", "B"],
            "variant": ["Default", "Tuned", "Default"],
            "family": ["Tree-based", "Tree-based", "Foundation Model"],
            "x": [0.1, 0.5, 1.0],
            "imp": [10.0, 8.0, 5.0],
            "elo": [1200.0, 1300.0, 1500.0],
            "imputed": [False, False, True],
            "imputed_pct": [0.0, 0.0, 25.0],
        }
    )


def test_build_scatter_explorer(tmp_path):
    out = build_pareto_explorer_html(
        points=_scatter_points(),
        save_path=tmp_path / "explorer.html",
        x_label="Inference time per 1K samples (s), median — log scale",
    )
    html = out.read_text(encoding="utf-8")
    # All placeholders substituted, data + interaction hooks present.
    assert "__CONFIG_JSON__" not in html
    assert "__POINTS_JSON__" not in html
    assert "__PAGE_TITLE__" not in html
    assert '"mode": "scatter"' in html
    assert '"method":"A"' in html
    assert "Pareto front" in html


def test_build_trajectory_explorer(tmp_path):
    points = pd.DataFrame(
        {
            "method": ["A", "A", "B", "B"],
            "family": ["Tree-based", "Tree-based", "Foundation Model", "Foundation Model"],
            "x": [1.0, 10.0, 2.0, 20.0],
            "imp": [10.0, 8.0, 9.0, 6.0],
            "elo": [1200.0, 1300.0, 1250.0, 1500.0],
            "n_configs": [1, 8, 1, 8],
        }
    )
    out = build_pareto_explorer_html(
        points=points,
        save_path=tmp_path / "trajectories.html",
        mode="trajectory",
        x_label="Train time per 1K samples (s), median — log scale",
    )
    html = out.read_text(encoding="utf-8")
    assert '"mode": "trajectory"' in html
    assert '"n_configs":' in html


def test_nonpositive_x_rows_are_dropped(tmp_path):
    points = _scatter_points()
    points.loc[0, "x"] = 0.0  # cannot be placed on a log axis
    out = build_pareto_explorer_html(points=points, save_path=tmp_path / "e.html", x_label="x")
    html = out.read_text(encoding="utf-8")
    assert '"x":0.0' not in html


def test_missing_metric_columns_raise(tmp_path):
    points = _scatter_points().drop(columns=["imp", "elo"])
    with pytest.raises(ValueError, match="metric column"):
        build_pareto_explorer_html(points=points, save_path=tmp_path / "e.html", x_label="x")


def test_unknown_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        build_pareto_explorer_html(points=_scatter_points(), save_path=tmp_path / "e.html", mode="bars", x_label="x")
