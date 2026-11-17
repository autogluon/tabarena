from __future__ import annotations

import json
import re

import pandas as pd

from tabarena.plot.interactive.composite_explorer import build_composite_explorer_html


def _composite() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [["CatBoost (T+E)", "TabPFN-3", "AutoGluon 1.5 (extreme, 4h)"], ["Elo", "Improv%"]], names=["method", "metric"]
    )
    return pd.DataFrame(
        {"all": [1500, 4.0, 1450, 5.0, 1400, 6.0], "binary": [1520, 3.5, 1480, None, 1390, 6.5]},
        index=index,
    )


def _config(html: str) -> dict:
    return json.loads(re.search(r"const CONFIG = (\{.*?\});\n", html, re.S).group(1))


def _points(html: str) -> list[dict]:
    return json.loads(re.search(r"const POINTS = (\[.*?\]);\n", html, re.S).group(1))


def test_the_page_carries_the_composite_as_published(tmp_path):
    page = build_composite_explorer_html(
        _composite(),
        save_path=tmp_path / "composite_leaderboard_explorer.html",
        higher_better={"Elo": True, "Improv%": False},
        title="Leaderboard",
        system_names=frozenset({"AutoGluon 1.5 (extreme, 4h)"}),
    )
    html = page.read_text(encoding="utf-8")
    config = _config(html)
    assert config["subsets"] == ["all", "binary"]
    assert config["metrics"] == ["Elo", "Improv%"]
    assert config["values"]["Elo"] == [[1500.0, 1520.0], [1450.0, 1480.0], [1400.0, 1390.0]]
    # A missing cell stays missing rather than turning into a number.
    assert config["values"]["Improv%"][1] == [5.0, None]
    assert config["higherBetter"] == {"Elo": True, "Improv%": False}
    points = {p["method"]: p for p in _points(html)}
    assert list(points) == ["CatBoost (T+E)", "TabPFN-3", "AutoGluon 1.5 (extreme, 4h)"]
    assert points["CatBoost (T+E)"]["model"] == "CatBoost"
    assert points["AutoGluon 1.5 (extreme, 4h)"]["family"] != points["CatBoost (T+E)"]["family"]


def test_an_empty_composite_writes_nothing(tmp_path):
    empty = _composite().iloc[0:0]
    assert build_composite_explorer_html(empty, save_path=tmp_path / "x.html", higher_better={}) is None
    assert not (tmp_path / "x.html").exists()
