"""The host messages every embedded chart answers: the site's theme, followed methods, linked hover."""

from __future__ import annotations

import pandas as pd
import pytest

from tabarena.plot.interactive._explorer_shared import EXPLORER_BASE_JS
from tabarena.plot.interactive.leaderboard_explorer import build_leaderboard_explorer_html
from tabarena.plot.interactive.leaderboard_table import build_leaderboard_table_html
from tabarena.plot.interactive.pareto_explorer import build_pareto_explorer_html
from tabarena.plot.interactive.winrate_explorer import build_winrate_explorer_html


def _board() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Model": ["A (default)", "B (tuned)"],
            "TypeName": ["Tree-based", "Foundation Model"],
            "Elo [⬆️]": [1500.0, 1300.0],
            "Improvability (%) [⬇️]": [8.0, 12.0],
        }
    )


def _points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["A", "A", "B"],
            "variant": ["Default", "Tuned", "Default"],
            "family": ["Tree-based", "Tree-based", "Foundation Model"],
            "x_infer": [0.1, 0.5, 1.0],
            "x_train": [1.0, 20.0, 5.0],
            "imp": [10.0, 8.0, 5.0],
            "elo": [1200.0, 1300.0, 1500.0],
            "n_configs": [1, 8, 1],
        }
    )


def _matrix() -> pd.DataFrame:
    labels = ["A (default)", "B (default)"]
    return pd.DataFrame([[None, 0.6], [0.4, None]], index=labels, columns=labels)


BUILDERS = {
    "pareto": lambda path: build_pareto_explorer_html(_points(), save_path=path),
    "trajectory": lambda path: build_pareto_explorer_html(_points(), save_path=path, mode="trajectory"),
    "overview": lambda path: build_leaderboard_explorer_html(_board(), save_path=path),
    "table": lambda path: build_leaderboard_table_html(_board(), save_path=path),
    "winrate": lambda path: build_winrate_explorer_html(_matrix(), save_path=path),
}


def test_the_theme_comes_from_the_url_hash_before_any_chart_is_drawn():
    """The leaderboard loads each page as `<file>.html#theme=dark`; the stamp has to be on the root
    before the first paint, so it is read at the top of the shared script.
    """
    head = EXPLORER_BASE_JS.split("const FAM_ORDER")[0]
    assert "theme=(dark|light)" in head
    assert 'document.documentElement.setAttribute("data-theme", HASH_THEME)' in head


@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_every_chart_follows_and_relays_hover(tmp_path, kind):
    html = BUILDERS[kind](tmp_path / f"{kind}.html").read_text(encoding="utf-8")
    assert '"tabarena-explorer-theme"' in html
    # The shared definition plus the page's own registration.
    assert html.count("onFollowMessage(") == 2
    assert html.count("onHoverMessage(") == 2
    # The definition, then where the pointer arrives and where it leaves.
    assert html.count("postHover(") >= 3
    # A click on a method's marks asks the host to follow it on every chart.
    assert '"tabarena-explorer-pick"' in html
    assert html.count("pickMethod(") >= 2
