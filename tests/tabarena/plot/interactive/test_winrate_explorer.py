from __future__ import annotations

import json
import re

import pandas as pd

from tabarena.plot.interactive.winrate_explorer import _split_label, build_winrate_explorer_html


def _matrix(labels: list[str]) -> pd.DataFrame:
    n = len(labels)
    values = [[None if i == j else round(0.4 + 0.02 * (i - j), 6) for j in range(n)] for i in range(n)]
    return pd.DataFrame(values, index=labels, columns=labels)


def _points(html: str) -> list[dict]:
    return json.loads(re.search(r"const POINTS = (\[.*?\]);\n", html, re.S).group(1))


def test_tuned_ensembled_labels_keep_their_variant():
    """The reporter writes the tag as "(T+E)", and that has to parse as a variant.

    Reading it as part of the model name instead left every tuned + ensembled row
    variant-less, so the page offered a lone "Default" toggle and its "One per model"
    control had nothing to collapse.
    """
    assert _split_label("TabM (T+E)") == ("TabM", "Tuned + Ens.")
    assert _split_label("TabM (tuned + ensemble)") == ("TabM", "Tuned + Ens.")
    assert _split_label("TabM (tuned + ensembled)") == ("TabM", "Tuned + Ens.")
    assert _split_label("TabM (tuned)") == ("TabM", "Tuned")
    assert _split_label("TabM (default)") == ("TabM", "Default")
    # A parenthetical that is not a tuning variant belongs to the name: systems carry
    # their configuration there.
    assert _split_label("AutoGluon 1.6 (noncommercial, 4h)") == ("AutoGluon 1.6 (noncommercial, 4h)", "")


def test_build_winrate_explorer(tmp_path):
    labels = ["TabM (T+E)", "TabM (tuned)", "TabM (default)", "AutoGluon 1.6 (extreme, 4h)"]
    out = build_winrate_explorer_html(
        _matrix(labels),
        save_path=tmp_path / "winrate_explorer.html",
        system_names=frozenset({"AutoGluon 1.6 (extreme, 4h)"}),
    )
    html = out.read_text(encoding="utf-8")
    assert "__CONFIG_JSON__" not in html
    assert "__POINTS_JSON__" not in html
    variants = {p["method"]: p["variant"] for p in _points(html)}
    assert variants == {
        "TabM (T+E)": "Tuned + Ens.",
        "TabM (tuned)": "Tuned",
        "TabM (default)": "Default",
        "AutoGluon 1.6 (extreme, 4h)": "",
    }
    assert {p["family"] for p in _points(html) if p["model"] == "AutoGluon 1.6 (extreme, 4h)"} == {"System"}


def test_win_rates_are_rounded_for_the_page(tmp_path):
    """A full-variant matrix is a few thousand cells; unrounded floats dominate the file.

    One decimal of a percent is all the page prints, so four decimals of the fraction is
    already more precision than it can show.
    """
    out = build_winrate_explorer_html(
        pd.DataFrame(
            [[None, 0.123456789], [0.876543211, None]],
            index=["A (default)", "B (default)"],
            columns=["A (default)", "B (default)"],
        ),
        save_path=tmp_path / "winrate_explorer.html",
    )
    html = out.read_text(encoding="utf-8")
    assert "0.1235" in html
    assert "0.123456789" not in html


def test_a_single_method_has_nothing_to_compare(tmp_path):
    assert build_winrate_explorer_html(_matrix(["A (default)"]), save_path=tmp_path / "w.html") is None
