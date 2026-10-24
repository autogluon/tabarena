from __future__ import annotations

import pandas as pd

from tabarena.plot.interactive.leaderboard_table import build_leaderboard_table_html, leaderboard_table_points


def test_points_flag_a_train_time_that_includes_the_prediction_call():
    df = pd.DataFrame(
        {
            "#": [0, 1],
            "Model": ["A (default)", "B"],
            "TypeName": ["Tree-based", "System"],
            "Elo [⬆️]": [1500.0, 1400.0],
            "FitsAtPredict": [False, True],
        }
    )
    assert leaderboard_table_points(df)["fits_at_predict"].tolist() == [False, True]
    # Tables written before the column existed have no such method.
    assert leaderboard_table_points(df.drop(columns="FitsAtPredict"))["fits_at_predict"].tolist() == [False, False]


def test_cells_take_the_light_ramp_on_a_light_surface(tmp_path):
    """The website's light theme (or paper view) shades the cells in its pastel ramp, under dark digits."""
    df = pd.DataFrame({"Model": ["A (default)", "B"], "TypeName": ["Tree-based", "Tree-based"], "Elo [⬆️]": [1.0, 2.0]})
    html = build_leaderboard_table_html(df, save_path=tmp_path / "table.html").read_text(encoding="utf-8")
    assert "const LIGHT_RAMP = [[0, [135, 201, 160]]" in html
    assert "const ramp = lightSurface() ? LIGHT_RAMP : RAMP;" in html
