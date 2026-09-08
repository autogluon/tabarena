from __future__ import annotations

import matplotlib
import pandas as pd
import pytest
from matplotlib.colors import to_rgb

matplotlib.use("Agg", force=True)

from tabarena.plot.plot_pareto_focus import compute_front_methods, marker_edge_color, plot_pareto_focus


def _toy_points() -> pd.DataFrame:
    """Four points, three methods; method C is dominated by A's default."""
    return pd.DataFrame(
        {
            "Method": ["A", "A", "B", "C"],
            "Type": ["Default", "Tuned", "Default", "Default"],
            "Family": ["Tree-based", "Tree-based", "Foundation Model", "Neural Network"],
            "x": [0.1, 0.5, 1.0, 0.2],
            "y": [10.0, 8.0, 5.0, 30.0],
        }
    )


def test_compute_front_methods_min_min():
    front, methods = compute_front_methods(
        _toy_points(), x_col="x", y_col="y", method_col="Method", max_X=False, max_Y=False
    )
    assert methods == {"A", "B"}
    # Staircase runs from the best-x end toward the best-y end.
    assert front[0] == (0.1, 10.0)
    assert front[-1] == (1.0, 5.0)


def test_compute_front_methods_max_y():
    _, methods = compute_front_methods(
        _toy_points(), x_col="x", y_col="y", method_col="Method", max_X=False, max_Y=True
    )
    # Higher-is-better on y: A's default (leftmost) and C's higher value define the front.
    assert methods == {"A", "C"}


def test_plot_pareto_focus_writes_figure(tmp_path):
    save_path = tmp_path / "pareto_focus.png"
    plot_pareto_focus(
        data=_toy_points(),
        x_col="x",
        y_col="y",
        focus_methods=["C"],
        x_label="Time (s)",
        y_label="Improvability (%)",
        title="Toy",
        save_path=save_path,
    )
    assert save_path.is_file()
    assert save_path.stat().st_size > 0


def test_marker_edge_color_darkens_emphasized_markers_only():
    family = "#5cb85c"
    muted_edge = marker_edge_color(family, emphasized=False)
    emphasized_edge = marker_edge_color(family, emphasized=True)
    assert muted_edge == pytest.approx(to_rgb(family))
    assert all(e < m for e, m in zip(emphasized_edge, muted_edge, strict=True))


def test_label_halo_off_keeps_svg_labels_as_text(tmp_path):
    """A haloed label is written as glyph outlines; without the halo it stays a text node."""
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        for halo in (True, False):
            plot_pareto_focus(
                data=_toy_points(),
                x_col="x",
                y_col="y",
                focus_methods=["C"],
                label_halo=halo,
                save_path=tmp_path / f"halo_{halo}.svg",
            )
    assert ">C<" not in (tmp_path / "halo_True.svg").read_text()
    assert ">C<" in (tmp_path / "halo_False.svg").read_text()
