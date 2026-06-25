"""Tests for ``LeaderboardReporter`` plotting helpers.

Focused on :meth:`LeaderboardReporter._plot_only_to_hidden_methods`, the pure
translation that powers the ``plot_only`` plotting allowlist (issue #306): it
turns an allowlist of method *display names* into the ``hidden_methods``
denylist the plot helpers already honor, so scoring (Elo / ranks) is never
touched — only the figures are filtered.
"""

from __future__ import annotations

from tabarena.evaluation.leaderboard_reporter import LeaderboardReporter

# ``f_map_type_name`` maps a config method's *short* config_type to its long
# display name (e.g. "GBM" -> "LightGBM"); baselines are not keys, so they map
# to themselves. This mirrors what ``eval`` builds from the method metadata.
_F_MAP = {"GBM": "LightGBM", "CAT": "CatBoost", "TABM": "TabM"}


class TestPlotOnlyToHiddenMethods:
    def test_hides_the_complement_of_plot_only(self):
        # Keep one config (LightGBM) and one baseline (TabPFN-3); everything else
        # is hidden. Display-name surface: configs via f_map_type_name, baselines as-is.
        hidden = LeaderboardReporter._plot_only_to_hidden_methods(
            ["LightGBM", "TabPFN-3"],
            framework_types=["GBM", "CAT", "TABM"],
            baselines=["TabPFN-3", "AutoGluon 1.5 (extreme, 4h)"],
            f_map_type_name=_F_MAP,
        )
        assert hidden == sorted(["CatBoost", "TabM", "AutoGluon 1.5 (extreme, 4h)"])

    def test_keeping_everything_hides_nothing(self):
        hidden = LeaderboardReporter._plot_only_to_hidden_methods(
            ["LightGBM", "CatBoost"],
            framework_types=["GBM", "CAT"],
            baselines=[],
            f_map_type_name={"GBM": "LightGBM", "CAT": "CatBoost"},
        )
        assert hidden == []

    def test_unions_with_existing_hidden_methods(self):
        # A pre-existing hidden_methods denylist composes with the plot_only complement.
        hidden = LeaderboardReporter._plot_only_to_hidden_methods(
            ["LightGBM"],
            framework_types=["GBM", "CAT"],
            baselines=[],
            f_map_type_name={"GBM": "LightGBM", "CAT": "CatBoost"},
            existing_hidden=["KNN"],
        )
        assert hidden == sorted(["CatBoost", "KNN"])

    def test_unknown_name_is_ignored_and_warned(self, capsys):
        # "SAP-RPT-1" matches nothing (the display name is "SAP-RPT-OSS"): it is
        # warned about and ignored — and since it is not a valid keep, SAP-RPT-OSS
        # is (correctly) hidden, which is exactly the typo-guard signal we want.
        hidden = LeaderboardReporter._plot_only_to_hidden_methods(
            ["LightGBM", "SAP-RPT-1"],
            framework_types=["GBM", "CAT"],
            baselines=["SAP-RPT-OSS"],
            f_map_type_name={"GBM": "LightGBM", "CAT": "CatBoost"},
        )
        assert hidden == sorted(["CatBoost", "SAP-RPT-OSS"])
        assert "SAP-RPT-1" in capsys.readouterr().out
