from __future__ import annotations

import json
import re

from tabarena.plot.interactive.explorer_hub import build_explorer_hub_html


def _manifest(html: str) -> dict:
    return json.loads(re.search(r"const M = (\{.*?\});\n", html, re.S).group(1))


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<!doctype html>", encoding="utf-8")


def test_the_hub_lists_the_pages_the_run_wrote(tmp_path):
    _touch(tmp_path / "composite_leaderboard_explorer.html")
    _touch(tmp_path / "pareto_front_explorer.html")
    _touch(tmp_path / "all" / "per_dataset_explorer.html")
    _touch(tmp_path / "all" / "head_to_head_explorer.html")
    _touch(tmp_path / "core&!large" / "head_to_head_explorer.html")
    _touch(tmp_path / "core&!large" / "winrate_explorer.html")
    page = build_explorer_hub_html(tmp_path, subsets=["core&!large", "all", "missing"])
    assert page == tmp_path / "explorers.html"
    manifest = _manifest(page.read_text(encoding="utf-8"))
    assert [r["label"] for r in manifest["root"]] == ["Leaderboard", "Pareto front"]
    tabs = {t["key"]: t for t in manifest["subsetTabs"]}
    assert list(tabs) == ["per_dataset", "head_to_head", "winrate"]
    # Subsets follow the composite leaderboard's column order and one without any page is dropped; a space would be
    # encoded in a link, `&` and `!` need not.
    assert manifest["subsets"] == ["all", "core&!large"]
    assert tabs["head_to_head"]["pages"] == {
        "core&!large": "core&!large/head_to_head_explorer.html",
        "all": "all/head_to_head_explorer.html",
    }
    assert tabs["per_dataset"]["pages"] == {"all": "all/per_dataset_explorer.html"}


def test_the_hub_finds_the_subsets_itself_and_skips_an_empty_run(tmp_path):
    assert build_explorer_hub_html(tmp_path) is None
    _touch(tmp_path / "b" / "winrate_explorer.html")
    _touch(tmp_path / "a" / "winrate_explorer.html")
    (tmp_path / "figures").mkdir()
    manifest = _manifest(build_explorer_hub_html(tmp_path).read_text(encoding="utf-8"))
    assert manifest["subsets"] == ["a", "b"]
    assert manifest["root"] == []


def test_the_subset_picker_names_subsets_as_the_composite_leaderboard_does(tmp_path):
    for subset in ("all&lite", "binary&lite"):
        _touch(tmp_path / subset / "winrate_explorer.html")
    manifest = _manifest(
        build_explorer_hub_html(tmp_path, subsets=["all&lite", "binary&lite"]).read_text(encoding="utf-8")
    )
    assert manifest["labels"] == {"all&lite": "all", "binary&lite": "binary"}
    assert manifest["subsetTabs"][0]["pages"]["all&lite"] == "all&lite/winrate_explorer.html"
    kept = _manifest(
        build_explorer_hub_html(tmp_path, subsets=["all&lite", "binary&lite"], strip_lite=False).read_text(
            encoding="utf-8"
        )
    )
    assert kept["labels"] == {"all&lite": "all&lite", "binary&lite": "binary&lite"}


def test_the_hub_hosts_the_explorers_view_switch(tmp_path):
    _touch(tmp_path / "pareto_front_explorer.html")
    html = build_explorer_hub_html(tmp_path).read_text(encoding="utf-8")
    # Framed, a chart opens on its figure and hides its own controls: the hub switches it to its edit view.
    assert '"tabarena-explorer-paper"' in html and 'id="paper"' in html
