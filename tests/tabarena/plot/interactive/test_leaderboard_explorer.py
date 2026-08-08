from __future__ import annotations

import pandas as pd

from tabarena.plot.interactive.leaderboard_explorer import build_leaderboard_explorer_html


def _website_leaderboard() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Model": [
                "[A-Very-Long-Method-Name (default)](https://example.org)",
                "B (tuned)",
                "C (default)",
            ],
            "TypeName": ["Foundation Model", "Tree-based", "Foundation Model"],
            "Elo [⬆️]": [1500.0, 1300.0, 1200.0],
            "Elo 95% CI": ["+80/-70", "+60/-55", "+50/-40"],
            "Rank [⬇️]": [5.0, 10.0, 15.0],
            "Improvability (%) [⬇️]": [8.0, 12.0, 16.0],
            "Median Train Time (s/1K) [⬇️]": [5.0, 40.0, 2.0],
            "Median Predict Time (s/1K) [⬇️]": [0.1, 0.5, 0.2],
            "Imputed (%) [⬇️]": [0.0, 0.0, 25.0],
        }
    )


def test_build_leaderboard_explorer(tmp_path):
    out = build_leaderboard_explorer_html(
        _website_leaderboard(),
        save_path=tmp_path / "leaderboard_overview_explorer.html",
    )
    html = out.read_text(encoding="utf-8")
    assert "__CONFIG_JSON__" not in html
    assert "__PAGE_TITLE__" not in html
    assert '"method":"A-Very-Long-Method-Name"' in html


def test_edge_labels_are_kept_inside_the_viewport(tmp_path):
    """The outermost method names must be clamped into the plot, not clipped by it.

    The first and last columns are centred half a slot from the edge while a name may
    span two slots, so without the clamp a long name (the top method, e.g. after a new
    model takes first place) is cut off at the SVG viewport boundary.
    """
    out = build_leaderboard_explorer_html(
        _website_leaderboard(),
        save_path=tmp_path / "leaderboard_overview_explorer.html",
    )
    html = out.read_text(encoding="utf-8")
    assert "Math.max(half + 1, Math.min(cx, plotW - half - 1))" in html


def test_names_paint_above_the_column_hairlines(tmp_path):
    """A nudged name must read over a neighbouring column's hairline, not under it.

    The hairlines therefore go in a group created before the one holding the names, and
    the names carry a halo in the surface color.
    """
    out = build_leaderboard_explorer_html(
        _website_leaderboard(),
        save_path=tmp_path / "leaderboard_overview_explorer.html",
    )
    html = out.read_text(encoding="utf-8")
    assert html.index("const xLines =") < html.index("const xLabels =")
    assert '"paint-order": "stroke", stroke: "var(--paper)"' in html
