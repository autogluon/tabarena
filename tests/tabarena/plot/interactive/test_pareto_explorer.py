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
            "x_infer": [0.1, 0.5, 1.0],
            "x_train": [1.0, 20.0, 5.0],
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
    )
    html = out.read_text(encoding="utf-8")
    # All placeholders substituted, data + interaction hooks present.
    assert "__CONFIG_JSON__" not in html
    assert "__POINTS_JSON__" not in html
    assert "__PAGE_TITLE__" not in html
    assert "__FAMILY_CSS_VARS__" not in html
    assert '"mode": "scatter"' in html
    assert '"method":"A"' in html
    assert "Pareto front" in html
    # Family colors injected from the shared leaderboard scheme.
    assert "--fam-foundation: #b07cf0;" in html


def test_x_keys_selects_and_orders_axes(tmp_path):
    out = build_pareto_explorer_html(
        points=_scatter_points(),
        save_path=tmp_path / "explorer.html",
        x_keys=["x_train"],
    )
    html = out.read_text(encoding="utf-8")
    assert '"key": "x_train"' in html
    assert '"key": "x_infer"' not in html


def test_build_trajectory_explorer(tmp_path):
    points = pd.DataFrame(
        {
            "method": ["A", "A", "B", "B"],
            "family": ["Tree-based", "Tree-based", "Foundation Model", "Foundation Model"],
            "x_train": [1.0, 10.0, 2.0, 20.0],
            "x_infer": [0.1, 0.2, 0.3, 0.4],
            "imp": [10.0, 8.0, 9.0, 6.0],
            "elo": [1200.0, 1300.0, 1250.0, 1500.0],
            "n_configs": [1, 8, 1, 8],
        }
    )
    out = build_pareto_explorer_html(
        points=points,
        save_path=tmp_path / "trajectories.html",
        mode="trajectory",
        x_keys=["x_train", "x_infer"],
    )
    html = out.read_text(encoding="utf-8")
    assert '"mode": "trajectory"' in html
    assert '"n_configs":' in html


def test_nonpositive_x_rows_are_dropped(tmp_path):
    points = _scatter_points()
    points.loc[0, "x_infer"] = 0.0  # cannot be placed on a log axis
    out = build_pareto_explorer_html(points=points, save_path=tmp_path / "e.html")
    html = out.read_text(encoding="utf-8")
    assert '"x_infer":0.0' not in html


def test_missing_metric_columns_raise(tmp_path):
    points = _scatter_points().drop(columns=["imp", "elo"])
    with pytest.raises(ValueError, match="metric column"):
        build_pareto_explorer_html(points=points, save_path=tmp_path / "e.html")


def test_missing_x_columns_raise(tmp_path):
    points = _scatter_points().drop(columns=["x_infer", "x_train"])
    with pytest.raises(ValueError, match="x column"):
        build_pareto_explorer_html(points=points, save_path=tmp_path / "e.html")


def test_point_labels_are_measured_so_long_names_stay_inside_the_plot(tmp_path):
    """A label goes on whichever side of its point has room for the whole name.

    A system's name ("AutoGluon 1.6 (noncommercial, 4h)") is several times the width the
    old fixed 110px guess assumed, so a point in the right-hand third had its name run
    off the plot, and the de-overlap pass under-counted how far a name reached.
    """
    out = build_pareto_explorer_html(points=_scatter_points(), save_path=tmp_path / "e.html")
    html = out.read_text(encoding="utf-8")
    assert "const textWidth = makeTextMeasurer(svg, { size: 13, weight: 700 });" in html
    assert "const toRight = px + 10 + w <= W - M.r;" in html
    assert 'const spanOf = l => (l.anchor === "start" ? [l.x, l.x + l.w] : [l.x - l.w, l.x]);' in html


def test_the_highlighted_front_follows_the_metric(tmp_path):
    """Switching the y-axis has to re-highlight, because each metric has its own front.

    A method can lead on relative gain and sit mid-field on Elo, so an active set carried
    over from the previous metric leaves methods drawn on the front but greyed out. The
    chart only re-highlights while the reader has not picked methods themselves.
    """
    out = build_pareto_explorer_html(points=_scatter_points(), save_path=tmp_path / "e.html")
    html = out.read_text(encoding="utf-8")
    assert "if (state.followFront) showFront(); else render();" in html
    assert 'metricSelect.addEventListener("change", ev => setMetric(ev.target.value));' in html
    assert "setMetric(d.metric);" in html
    # ...and a hand-picked selection turns the following off.
    assert html.count("state.followFront = false;") >= 2


def test_unknown_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        build_pareto_explorer_html(points=_scatter_points(), save_path=tmp_path / "e.html", mode="bars")
