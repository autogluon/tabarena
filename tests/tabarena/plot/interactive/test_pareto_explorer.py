from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from tabarena.plot.interactive.pareto_explorer import build_pareto_explorer_html
from tabarena.plot.plot_pareto_focus import FAMILY_COLORS, SYSTEM_SHADE_COLORS, marker_edge_color


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
    # Marker edges: the family color darkened for the light surface, as in the static figures.
    edge = marker_edge_color(FAMILY_COLORS["Foundation Model"])
    assert f"--fam-foundation-edge: {edge};" in html
    assert "const edge = on ? famEdge(p) : null;" in html
    # The system shades get the same color and edge pair.
    assert f"--fam-system-api: {SYSTEM_SHADE_COLORS['closed-source-api']};" in html


def test_time_axes_are_offered_on_a_grouped_selector(tmp_path):
    from tabarena.plot.interactive.pareto_explorer import X_AXIS_KEYS

    points = _scatter_points()
    for key in X_AXIS_KEYS:
        points[key] = [1.0, 2.0, 3.0]
    html = build_pareto_explorer_html(points=points, save_path=tmp_path / "explorer.html").read_text(encoding="utf-8")
    # every axis rides along, in the selector's order, the first shown at load
    positions = [html.index(f'"key": "{key}"') for key in X_AXIS_KEYS]
    assert positions == sorted(positions) and len(X_AXIS_KEYS) == 12
    assert '"group": "Median, per 1K samples"' in html and '"group": "Mean, per task"' in html
    assert '"axisLabel": "Total (train + inference) time per task (s), mean over datasets' in html
    assert 'id="xaxis-select"' in html and "function setXAxis(key)" in html
    assert "tabarena-explorer-xaxis" in html


def test_subsets_put_a_subset_selector_on_the_page(tmp_path):
    points = pd.concat([_scatter_points().assign(subset="all"), _scatter_points().assign(subset="tiny")])
    html = build_pareto_explorer_html(
        points=points, subsets=[("all", "All tasks"), ("tiny", "Tiny")], save_path=tmp_path / "explorer.html"
    ).read_text(encoding="utf-8")
    assert '"subsets": [{"key": "all", "label": "All tasks"}, {"key": "tiny", "label": "Tiny"}]' in html
    assert html.count('"subset":"tiny"') == 3 and "function setSubset(key)" in html
    with pytest.raises(ValueError, match="subsets the `subsets` list lacks"):
        build_pareto_explorer_html(points=points, subsets=[("all", "All")], save_path=tmp_path / "bad.html")
    with pytest.raises(ValueError, match="missing required column 'subset'"):
        build_pareto_explorer_html(points=_scatter_points(), subsets=[("all", "All")], save_path=tmp_path / "bad.html")


def test_a_page_without_subsets_has_no_subset_selector_config(tmp_path):
    html = build_pareto_explorer_html(points=_scatter_points(), save_path=tmp_path / "explorer.html").read_text(
        encoding="utf-8"
    )
    assert '"subsets": null' in html


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


def test_end_to_end_marker_joins_the_legend_when_a_system_is_plotted(tmp_path):
    points = _scatter_points()
    points.loc[points.index[-1], "variant"] = "End-to-end"
    out = tmp_path / "explorer.html"
    build_pareto_explorer_html(points, save_path=out)
    html = out.read_text()
    assert 'if (SHOWN.some(p => p.variant === "End-to-end"))' in html  # the shown subset's points
    assert "End-to-end</span>" in html


def test_dim_off_front_is_off_by_default_and_reaches_the_page(tmp_path):
    points = _scatter_points()
    default = build_pareto_explorer_html(points, save_path=tmp_path / "default.html").read_text()
    dimmed = build_pareto_explorer_html(points, save_path=tmp_path / "dim.html", dim_off_front=True).read_text()
    assert '"dimOffFront": false' in default
    assert '"dimOffFront": true' in dimmed
    assert "const dim = CONFIG.dimOffFront && on && !front.points.has(p);" in dimmed


def test_host_metric_echo_does_not_undo_a_local_pick(tmp_path):
    """The host repeats its metric on every height report; only a change of the host's choice applies."""
    out = build_pareto_explorer_html(points=_scatter_points(), save_path=tmp_path / "explorer.html")
    html = out.read_text(encoding="utf-8")
    assert "let lastHostMetric = null;" in html
    assert "if (d.metric === lastHostMetric) return;" in html


def test_a_narrow_chart_shortens_its_axis_title(tmp_path):
    """A chart half the page wide cannot hold the full time-axis title; it drops the aggregation, then
    falls back to the selector's label.
    """
    html = build_pareto_explorer_html(_scatter_points(), save_path=tmp_path / "explorer.html").read_text()
    assert 'const titles = [full, full.replace(", median over datasets", ""), xAxis().label];' in html


def test_axis_limits_are_typed_per_view_and_reset(tmp_path):
    """Each (subset, metric, time axis) keeps its own typed limits; unusable ones fall back to the automatic limits,
    which the empty boxes show, and the data is drawn inside the plot's clip.
    """
    html = build_pareto_explorer_html(_scatter_points(), save_path=tmp_path / "explorer.html").read_text()
    for k in ("xmin", "xmax", "ymin", "ymax"):
        assert f'id="lim-{k}"' in html
    assert 'id="btn-auto-limits"' in html
    assert "function viewKey() { return JSON.stringify([subsetKey, metricKey, xKey]); }" in html
    assert "const xr = axisRange(typed.xmin, typed.xmax, auto.xmin, auto.xmax, !linearX);" in html
    assert "input.placeholder = fmtLimit(auto[k]);" in html
    assert 'const plot = el("g", { "clip-path": "url(#plotclip)" }, svg);' in html


def test_the_figure_title_is_drawn_in_the_chart_and_edited_per_subset(tmp_path):
    """The title opens on the page's title and the shown subset's label, sits in the chart (so paper view and the
    exports carry it), and a title typed for a subset stays with it.
    """
    points = pd.concat([_scatter_points().assign(subset=key) for key in ("all", "binary")])
    html = build_pareto_explorer_html(
        points, save_path=tmp_path / "explorer.html", subsets=[("all", "all"), ("binary", "binary")]
    ).read_text()
    assert 'id="fig-title"' in html
    assert 'id="title"' not in html
    assert 'return [CONFIG.title, subset].filter(Boolean).join(" · ");' in html
    assert "titleBySubset.set(subsetKey, ev.target.value);" in html
    assert 'const H = (usableSize("h") || CHART_H) + (title ? TITLE_H : 0);' in html
    titled = build_pareto_explorer_html(_scatter_points(), save_path=tmp_path / "titled.html", title="classification")
    assert '"title": "classification"' in titled.read_text()


def test_methods_are_renamed_for_display_only(tmp_path):
    """A rename (preset by the builder or typed on a chip) changes what the chart, chips, tooltips and table call a
    method; the method stays the key, and typed names are escaped where they become markup.
    """
    html = build_pareto_explorer_html(
        _scatter_points(), save_path=tmp_path / "explorer.html", method_names={"A": "Alpha <tuned>"}
    ).read_text()
    assert '"methodNames": {"A": "Alpha <tuned>"}' in html
    assert "const names = new Map(Object.entries(CONFIG.methodNames || {}));" in html
    assert "t.textContent = displayName(l.method);" in html
    assert "<td>${escHtml(displayName(p.method))}</td>" in html
    assert 'id="btn-reset-names"' in html
    plain = build_pareto_explorer_html(_scatter_points(), save_path=tmp_path / "plain.html").read_text()
    assert '"methodNames": {}' in plain


def test_the_chart_size_is_typed_for_the_page(tmp_path):
    """A typed width and height replace the panel's width and the default height on every view, and the legend wraps
    to a typed width, so the exports are as narrow as the chart.
    """
    html = build_pareto_explorer_html(_scatter_points(), save_path=tmp_path / "explorer.html").read_text()
    assert 'id="size-w"' in html
    assert 'id="size-h"' in html
    assert 'id="btn-auto-size"' in html
    assert 'const W = usableSize("w") || autoW;' in html
    assert 'const H = (usableSize("h") || CHART_H) + (title ? TITLE_H : 0);' in html
    assert 'legendStrip.style.maxWidth = usableSize("w") ? W + "px" : "";' in html


def test_unselected_methods_can_be_hidden(tmp_path):
    """Hidden rather than greyed out, unselected methods draw no points, connectors or hover targets, and the dashed
    front joins the shown points; the builder can open the page that way.
    """
    html = build_pareto_explorer_html(_scatter_points(), save_path=tmp_path / "explorer.html").read_text()
    assert '"hideUnselected": false' in html
    assert 'id="btn-hide-off"' in html
    assert "const visible = state.hideOff ? VIEW.filter(p => isOn(p.method)) : VIEW;" in html
    assert "const front = computeFront(metric, visible);" in html
    assert "for (const p of visible) {" in html
    hidden = build_pareto_explorer_html(_scatter_points(), save_path=tmp_path / "hidden.html", hide_unselected=True)
    assert '"hideUnselected": true' in hidden.read_text()


def test_a_metric_x_axis_is_linear_and_keeps_incomplete_points_on_request(tmp_path):
    """A metric x-axis is linear (negative values stay) and draws no front; ``drop_incomplete=False`` keeps a point
    that lacks a value, which each view then skips where it has none.
    """
    points = _scatter_points().assign(x_adv=[-2.0, 1.0, 3.0])
    points.loc[points.index[0], "elo"] = float("nan")
    html = build_pareto_explorer_html(
        points, save_path=tmp_path / "explorer.html", x_keys=["x_infer", "x_adv"], drop_incomplete=False
    ).read_text()
    config = json.loads(re.search(r"const CONFIG = (\{.*?\});\n", html).group(1))
    assert [a.get("scale", "log") for a in config["xAxes"]] == ["log", "linear"]
    assert len(json.loads(re.search(r"const POINTS = (\[.*?\]);\n", html, re.S).group(1))) == len(points)
    assert "const drawFront = !linearX && metric.front !== false;" in html
    assert "if (xAxis().diagonalWith === metricKey) {" in html
    assert "const VIEW = SHOWN.filter(p => onView(p, metric));" in html
    dropped = build_pareto_explorer_html(points, save_path=tmp_path / "dropped.html", x_keys=["x_infer", "x_adv"])
    assert (
        len(json.loads(re.search(r"const POINTS = (\[.*?\]);\n", dropped.read_text(), re.S).group(1)))
        == len(points) - 1
    )
