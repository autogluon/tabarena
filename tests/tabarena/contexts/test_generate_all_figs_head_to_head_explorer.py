"""``generate_all_figs``'s head-to-head browser."""

from __future__ import annotations

import json
import re

import pytest

from tabarena.contexts.abstract_arena_context import AbstractArenaContext

from .test_generate_all_figs_per_dataset_explorer import _arena, _subset_folder


def test_the_head_to_head_explorer_reads_the_subset_folder(tmp_path):
    """Both methods of the per-split results are offered, a `new_results` one under its own name."""
    _subset_folder(tmp_path / "all", with_trajectories=False)
    page = AbstractArenaContext._save_head_to_head_explorer(
        _arena(), tmp_path / "all", leaderboard=None, subset_label="all"
    )

    assert page == tmp_path / "all" / "head_to_head_explorer.html"
    html = page.read_text(encoding="utf-8")
    config = json.loads(re.search(r"const CONFIG = (\{.*?\});\n", html).group(1))
    names = [m["name"] for m in config["methods"]]
    assert sorted(names) == ["CatBoost (default)", "My portfolio"]
    assert {names[config["defaultA"]], names[config["defaultB"]]} == set(names)
    assert [d["name"] for d in config["datasets"]] == ["Alpha", "Beta"]
    # the page's Elo pins the method the arena's leaderboard pins
    assert names[config["elo"]["calibration"]] == "CatBoost (default)"
    assert "<title>Arena head-to-head — all</title>" in html


def test_no_head_to_head_explorer_without_the_compare_pass_files(tmp_path):
    (tmp_path / "all").mkdir()
    assert (
        AbstractArenaContext._save_head_to_head_explorer(
            _arena(), tmp_path / "all", leaderboard=None, subset_label="all"
        )
        is None
    )


def test_the_head_to_head_explorer_follows_the_compare_pass(monkeypatch):
    captured = []

    def fake_subset_figs(self, subset, **kwargs):
        captured.append(kwargs)
        return ("", None)

    monkeypatch.setattr(AbstractArenaContext, "_generate_subset_figs", fake_subset_figs)
    monkeypatch.setattr(AbstractArenaContext, "_default_subsets", property(lambda self: [[]]))
    monkeypatch.setattr(AbstractArenaContext, "_save_composite_pareto_explorer", lambda *args: None)
    context = AbstractArenaContext.__new__(AbstractArenaContext)

    AbstractArenaContext.generate_all_figs(context, output_dir="unused", max_workers=1)
    assert captured[-1]["save_head_to_head_explorer"] is True
    AbstractArenaContext.generate_all_figs(context, output_dir="unused", plot_compare=False, max_workers=1)
    assert captured[-1]["save_head_to_head_explorer"] is False
    with pytest.raises(ValueError, match="requires plot_compare=True"):
        AbstractArenaContext.generate_all_figs(
            context, output_dir="unused", plot_compare=False, save_head_to_head_explorer=True
        )


def test_the_head_to_head_explorer_opens_on_the_named_pair(tmp_path):
    _subset_folder(tmp_path / "all", with_trajectories=False)
    page = AbstractArenaContext._save_head_to_head_explorer(
        _arena(), tmp_path / "all", leaderboard=None, subset_label="all", methods=("My portfolio", "CatBoost (default)")
    )
    config = json.loads(re.search(r"const CONFIG = (\{.*?\});\n", page.read_text(encoding="utf-8")).group(1))
    names = [m["name"] for m in config["methods"]]
    assert (names[config["defaultA"]], names[config["defaultB"]]) == ("My portfolio", "CatBoost (default)")
