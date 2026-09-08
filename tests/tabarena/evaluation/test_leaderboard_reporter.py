"""Tests for ``LeaderboardReporter`` plotting helpers.

Focused on :meth:`LeaderboardReporter._plot_only_to_hidden_methods`, the pure
translation that powers the ``plot_only`` plotting allowlist (issue #306): it
turns an allowlist of method *display names* into the ``hidden_methods``
denylist the plot helpers already honor, so scoring (Elo / ranks) is never
touched — only the figures are filtered.
"""

from __future__ import annotations

import pandas as pd

from tabarena.evaluation import leaderboard_reporter as module
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


def test_date_introduced_plots_are_opt_in():
    """``eval`` must not render the date-introduced figures unless asked.

    They are a paper figure, and the four of them (two scatters plus two GIF timelapses)
    dominate the runtime of an otherwise quick evaluation, so the default stays off. Pinned
    here because the cost of a silent flip back is paid by every caller.
    """
    import inspect

    default = inspect.signature(LeaderboardReporter.eval).parameters["plot_date_introduced"].default
    assert default is False


class TestComputeOnlyReporter:
    """``output_dir=None`` builds a reporter that writes nothing and renders nothing."""

    def test_no_output_dir_and_no_matplotlib_style(self, monkeypatch):
        """The style setup is what imports matplotlib + tueplots, so a compute-only reporter must
        not run it — that import is the bulk of the reporter's construction cost.
        """
        import tabarena.evaluation.leaderboard_reporter as module

        called: list[bool] = []
        # `_init_global_rcparams` is `functools.cache`d, so patch it to observe the call itself.
        monkeypatch.setattr(module, "_init_global_rcparams", lambda: called.append(True))

        reporter = LeaderboardReporter(output_dir=None, task_metadata=[], use_latex=True)
        assert reporter.output_dir is None
        assert reporter.rc_context_params == {}  # latex rcparams need matplotlib; not applied
        assert called == []

        LeaderboardReporter(output_dir="some/dir", task_metadata=[])
        assert called == [True]

    def test_plot_defaults_to_true(self):
        """Back-compat: every existing caller keeps getting the full figure suite."""
        import inspect

        assert inspect.signature(LeaderboardReporter.eval).parameters["plot"].default is True


class TestParetoFocusKwargs:
    def test_kwargs_reach_every_pareto_figure(self, monkeypatch, tmp_path):
        """``pareto_focus_kwargs`` is forwarded unchanged to each of the four figure calls."""
        calls: list[dict] = []
        monkeypatch.setattr(module, "plot_pareto_focus", lambda **kwargs: calls.append(kwargs))
        monkeypatch.setattr(module, "_init_global_rcparams", lambda: None)
        monkeypatch.setattr(LeaderboardReporter, "build_pareto_explorer", lambda self, leaderboard: None)
        reporter = LeaderboardReporter(output_dir=tmp_path, task_metadata=[])
        leaderboard = pd.DataFrame(
            {
                "method": ["GBM (default)", "GBM (tuned + ensemble)", "TabPFN-3"],
                "config_type": ["GBM", "GBM", None],
                "elo": [1000.0, 1100.0, 1300.0],
                "improvability": [0.2, 0.1, 0.05],
                "median_time_train_s_per_1K": [1.0, 100.0, 5.0],
                "median_time_infer_s_per_1K": [0.1, 1.0, 0.5],
            }
        )

        reporter.plot_pareto(
            leaderboard, framework_types=["GBM"], pareto_focus_kwargs={"muted_size": 12, "muted_alpha": 0.2}
        )

        assert len(calls) == 4
        assert all(kwargs["muted_size"] == 12 and kwargs["muted_alpha"] == 0.2 for kwargs in calls)
        # The plain figure options still arrive alongside.
        assert all(kwargs["variant_markers"] is reporter.style_markers for kwargs in calls)
