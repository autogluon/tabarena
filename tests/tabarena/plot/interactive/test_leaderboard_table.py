from __future__ import annotations

import json
import re

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


def _config(html: str) -> dict:
    return json.loads(re.search(r"const CONFIG = (\{.*?\});\n", html, re.S).group(1))


def test_secondary_columns_start_switched_off(tmp_path):
    """The table opens on Elo, Improvability and the runtimes; the Columns menu brings the others in."""
    df = pd.DataFrame(
        {
            "#": [0, 1],
            "Model": ["A (default)", "B (tuned)"],
            "TypeName": ["Tree-based", "Tree-based"],
            "Elo [⬆️]": [1500.0, 1400.0],
            "Score [⬆️]": [0.8, 0.6],
            "Improvability (%) [⬇️]": [3.0, 5.0],
            "Hardware": ["CPU", "CPU"],
        }
    )
    html = build_leaderboard_table_html(df, save_path=tmp_path / "table.html").read_text(encoding="utf-8")
    off = {c["key"] for c in _config(html)["columns"] if c.get("on") is False}
    assert off == {"score", "hardware"}
    # Positions are stored as published (0 first) and shown from 1.
    assert '">\' + (p.position + 1) + "</td>"' in html


def test_the_host_can_badge_new_methods(tmp_path):
    df = pd.DataFrame({"Model": ["A (default)"], "TypeName": ["Tree-based"], "Elo [⬆️]": [1.0]})
    html = build_leaderboard_table_html(df, save_path=tmp_path / "table.html").read_text(encoding="utf-8")
    assert '"tabarena-explorer-new"' in html
    assert 'class="new-tag"' in html
