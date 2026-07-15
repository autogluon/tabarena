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


class TestMetricVsDateFrame:
    """``_metric_vs_date_frame`` powers the metric-vs-introduction-date figures."""

    @staticmethod
    def _reporter(method_metadata_info) -> LeaderboardReporter:
        # Only the attributes `_metric_vs_date_frame` reads; the full constructor needs results.
        reporter = LeaderboardReporter.__new__(LeaderboardReporter)
        reporter.method_metadata_info = method_metadata_info
        return reporter

    def test_mixed_precision_dates_all_parse(self):
        """'YYYY', 'YYYY-MM', and 'YYYY-MM-DD' must all survive parsing regardless of which
        precision comes first (plain ``pd.to_datetime`` infers the format from the first
        element and coerces the other precisions to NaT).
        """
        import pandas as pd

        meta = pd.DataFrame(
            {
                "ta_name": ["A", "B", "C"],
                "ta_suite": ["s", "s", "s"],
                "date_introduced": ["2026-06-30", "2017-06", "2006"],  # day-first on purpose
                "display_name": ["A", "B", "C"],
            }
        )
        leaderboard = pd.DataFrame(
            {"ta_name": ["A", "B", "C"], "ta_suite": ["s", "s", "s"], "elo": [1500.0, 1400.0, 1300.0]}
        )
        df = self._reporter(meta)._metric_vs_date_frame(leaderboard, metric="elo", higher_is_better=True)
        assert len(df) == 3
        dates = dict(zip(df["_label"], df["_date"], strict=False))
        assert dates["A"] == pd.Timestamp("2026-06-30")
        assert dates["B"] == pd.Timestamp("2017-06-01")
        assert dates["C"] == pd.Timestamp("2006-01-01")

    def test_best_elo_per_family_and_missing_dates_dropped(self):
        import pandas as pd

        meta = pd.DataFrame(
            {
                "ta_name": ["A", "D"],
                "ta_suite": ["s", "s"],
                "date_introduced": ["2020-01", None],
                "display_name": ["MethodA", "MethodD"],
            }
        )
        leaderboard = pd.DataFrame(
            {"ta_name": ["A", "A", "D"], "ta_suite": ["s", "s", "s"], "elo": [1200.0, 1350.0, 1500.0]}
        )
        df = self._reporter(meta)._metric_vs_date_frame(leaderboard, metric="elo", higher_is_better=True)
        assert df["_label"].tolist() == ["MethodA"]  # D has no date; A deduped to best Elo
        assert df["elo"].tolist() == [1350.0]

    def test_no_metadata_is_none(self):
        import pandas as pd

        leaderboard = pd.DataFrame({"ta_name": ["A"], "ta_suite": ["s"], "elo": [1500.0]})
        assert self._reporter(None)._metric_vs_date_frame(leaderboard, metric="elo", higher_is_better=True) is None

    def test_lower_is_better_metric_and_missing_metric_column(self):
        """With ``higher_is_better=False`` the per-family dedup keeps the lowest value; a
        leaderboard without the metric column is a no-op.
        """
        import pandas as pd

        meta = pd.DataFrame(
            {"ta_name": ["A"], "ta_suite": ["s"], "date_introduced": ["2020-01"], "display_name": ["A"]}
        )
        leaderboard = pd.DataFrame({"ta_name": ["A", "A"], "ta_suite": ["s", "s"], "improvability": [0.4, 0.2]})
        reporter = self._reporter(meta)
        df = reporter._metric_vs_date_frame(leaderboard, metric="improvability", higher_is_better=False)
        assert df["improvability"].tolist() == [0.2]
        assert reporter._metric_vs_date_frame(leaderboard, metric="elo", higher_is_better=True) is None
