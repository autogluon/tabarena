from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.use("Agg", force=True)

from tabarena.plot.plot_pareto_focus import compute_front_methods, plot_pareto_focus


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
