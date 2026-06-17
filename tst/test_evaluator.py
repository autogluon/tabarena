from __future__ import annotations

import itertools

import pandas as pd
import pytest

from bencheval.evaluator import BenchmarkEvaluator

# ---------------------------------------------------------------------------
# Deterministic, hand-computable fixture: 3 methods x 3 tasks, no ties.
#
#   task    A     B     C
#   t1     0.1   0.2   0.4
#   t2     0.1   0.3   0.5
#   t3     0.2   0.3   0.6
#
# A strictly beats B strictly beats C on every task, so per-task ranks are
# always (A=1, B=2, C=3) and every aggregate metric is exactly computable.
# ---------------------------------------------------------------------------
_ERRORS = {
    ("A", "t1"): 0.1, ("B", "t1"): 0.2, ("C", "t1"): 0.4,
    ("A", "t2"): 0.1, ("B", "t2"): 0.3, ("C", "t2"): 0.5,
    ("A", "t3"): 0.2, ("B", "t3"): 0.3, ("C", "t3"): 0.6,
}  # fmt: skip
_METHODS = ["A", "B", "C"]
_TASKS = ["t1", "t2", "t3"]


def _make_data(seeds: list[int] | None = None) -> pd.DataFrame:
    rows = []
    seed_list: list[int | None] = [None] if seeds is None else list(seeds)
    for m, t, s in itertools.product(_METHODS, _TASKS, seed_list):
        row = {"method": m, "task": t, "metric_error": _ERRORS[(m, t)], "time_train_s": 2.0, "time_infer_s": 0.5}
        if s is not None:
            row["seed"] = s
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def ev() -> BenchmarkEvaluator:
    return BenchmarkEvaluator()


@pytest.fixture
def data() -> pd.DataFrame:
    return _make_data()


class TestPerTaskMetrics:
    def test_compute_results_per_task(self, ev, data):
        rpt = ev.compute_results_per_task(data).set_index(["method", "task"])
        # ranks
        assert rpt.loc[("A", "t1"), "rank"] == pytest.approx(1.0)
        assert rpt.loc[("B", "t1"), "rank"] == pytest.approx(2.0)
        assert rpt.loc[("C", "t1"), "rank"] == pytest.approx(3.0)
        # improvability = 1 - best/error  (0 for the best method)
        assert rpt.loc[("A", "t1"), "improvability"] == pytest.approx(0.0)
        assert rpt.loc[("C", "t1"), "improvability"] == pytest.approx(0.75)
        assert rpt.loc[("B", "t3"), "improvability"] == pytest.approx(1 - 0.2 / 0.3)
        # loss_rescaled = (error - best) / (worst - best)
        assert rpt.loc[("A", "t1"), "loss_rescaled"] == pytest.approx(0.0)
        assert rpt.loc[("C", "t1"), "loss_rescaled"] == pytest.approx(1.0)
        assert rpt.loc[("B", "t1"), "loss_rescaled"] == pytest.approx(0.1 / 0.3)

    def test_rank_ties_use_average(self, ev):
        df = pd.DataFrame(
            {
                "method": ["A", "B", "C"],
                "task": ["t1", "t1", "t1"],
                "metric_error": [0.1, 0.1, 0.3],
                "time_train_s": [1.0, 1.0, 1.0],
                "time_infer_s": [0.1, 0.1, 0.1],
            }
        )
        r = ev.compute_results_per_task(df).set_index("method")["rank"]
        # A and B tie for best -> average of positions 1 and 2 = 1.5
        assert r.loc["A"] == pytest.approx(1.5)
        assert r.loc["B"] == pytest.approx(1.5)
        assert r.loc["C"] == pytest.approx(3.0)


class TestAggregateMetrics:
    def test_rank_counts(self, ev, data):
        rpt = ev.compute_results_per_task(data)
        rc = ev.compute_rank_counts(results_per_task=rpt)
        # each method's bucket counts sum to the number of tasks (3)
        assert (rc.sum(axis=1).round(6) == 3.0).all()
        assert rc.loc["A", "rank=1_count"] == pytest.approx(3.0)
        assert rc.loc["B", "rank=2_count"] == pytest.approx(3.0)
        assert rc.loc["C", "rank=3_count"] == pytest.approx(3.0)
        assert rc.loc["A", "rank>3_count"] == pytest.approx(0.0)

    def test_winrate(self, ev, data):
        rpt = ev.compute_results_per_task(data)
        wr = ev.compute_winrate(rpt)
        assert wr.loc["A"] == pytest.approx(1.0)
        assert wr.loc["B"] == pytest.approx(0.5)
        assert wr.loc["C"] == pytest.approx(0.0)

    def test_winrate_matrix(self, ev, data):
        rpt = ev.compute_results_per_task(data)
        wm = ev.compute_winrate_matrix(rpt)
        assert wm.loc["A", "B"] == pytest.approx(1.0)
        assert wm.loc["A", "C"] == pytest.approx(1.0)
        assert wm.loc["C", "A"] == pytest.approx(0.0)
        assert wm.loc["B", "C"] == pytest.approx(1.0)

    def test_mrr(self, ev, data):
        rpt = ev.compute_results_per_task(data)
        mrr = ev.compute_mrr(rpt)
        assert mrr.loc["A"] == pytest.approx(1.0)
        assert mrr.loc["B"] == pytest.approx(0.5)
        assert mrr.loc["C"] == pytest.approx(1 / 3)

    def test_relative_error(self, ev, data):
        rpt = ev.compute_results_per_task(data)
        rel = ev.compute_relative_error(rpt, baseline_method="A")
        assert rel.loc["A"] == pytest.approx(1.0)
        assert rel.loc["B"] == pytest.approx((2 + 3 + 1.5) / 3)
        assert rel.loc["C"] == pytest.approx((4 + 5 + 3) / 3)

    def test_baseline_advantage(self, ev, data):
        # exercises the consolidated `_resolve_groupby_columns` helper
        rpt = ev.compute_results_per_task(data)
        ba = ev.compute_baseline_advantage(rpt, baseline_method="A")
        assert ba.loc["A"] == pytest.approx(0.0)
        assert ba.loc["B"] == pytest.approx(-0.5)
        assert ba.loc["C"] == pytest.approx((-0.75 - 0.8 - 0.4 / 0.6) / 3)

    def test_frontier_advantage(self, ev, data):
        # exercises the consolidated `_resolve_groupby_columns` helper
        rpt = ev.compute_results_per_task(data)
        fa = ev.compute_frontier_advantage(rpt)
        assert fa.loc["A"] == pytest.approx(0.5)
        assert fa.loc["B"] == pytest.approx(-0.5)
        assert fa.loc["C"] == pytest.approx((-0.75 - 0.8 - 0.4 / 0.6) / 3)

    def test_elo_orders_by_dominance(self, ev, data):
        rpt = ev.compute_results_per_task(data)
        elo = ev.compute_elo(rpt, include_quantiles=False)
        assert elo.loc["A", "elo"] > elo.loc["B", "elo"] > elo.loc["C", "elo"]


class TestLeaderboard:
    def test_leaderboard_full(self, ev, data):
        lb = ev.leaderboard(
            data,
            include_error=True,
            include_elo=True,
            include_winrate=True,
            include_improvability=True,
            include_mrr=True,
            include_rescaled_loss=True,
            include_rank_counts=True,
            include_relative_error=True,
            include_skill_score=True,
            include_baseline_advantage=True,
            include_frontier_advantage=True,
            baseline_method="A",
        )
        # sorted by rank ascending
        assert list(lb.index) == ["A", "B", "C"]
        # rank / winrate / mrr
        assert [lb.loc[m, "rank"] for m in _METHODS] == pytest.approx([1.0, 2.0, 3.0])
        assert [lb.loc[m, "winrate"] for m in _METHODS] == pytest.approx([1.0, 0.5, 0.0])
        assert [lb.loc[m, "mrr"] for m in _METHODS] == pytest.approx([1.0, 0.5, 1 / 3])
        # mean error (equal task weighting)
        assert [lb.loc[m, "metric_error"] for m in _METHODS] == pytest.approx([0.4 / 3, 0.8 / 3, 0.5])
        # improvability / loss_rescaled point estimates
        assert lb.loc["A", "improvability"] == pytest.approx(0.0)
        assert lb.loc["B", "improvability"] == pytest.approx(0.5)
        assert lb.loc["C", "loss_rescaled"] == pytest.approx(1.0)
        assert lb.loc["A", "loss_rescaled"] == pytest.approx(0.0)
        # relative error / skill score
        assert lb.loc["A", "relative_error"] == pytest.approx(1.0)
        assert lb.loc["A", "skill_score"] == pytest.approx(0.0, abs=1e-9)
        # advantages
        assert lb.loc["A", "frontier_advantage"] == pytest.approx(0.5)
        assert lb.loc["B", "baseline_advantage"] == pytest.approx(-0.5)
        # elo ordering
        assert lb.loc["A", "elo"] > lb.loc["B", "elo"] > lb.loc["C", "elo"]
        # rank-count buckets
        assert lb.loc["A", "rank=1_count"] == pytest.approx(3.0)
        assert lb.loc["B", "rank=2_count"] == pytest.approx(3.0)
        assert lb.loc["C", "rank=3_count"] == pytest.approx(3.0)

    def test_seeded_leaderboard_matches_unseeded(self):
        # identical results across two seeds -> same aggregates as the unseeded case.
        # Also exercises the seed-present branch of `_resolve_groupby_columns`.
        ev_s = BenchmarkEvaluator(seed_column="seed")
        df = _make_data(seeds=[0, 1])
        for average_seeds in (True, False):
            lb = ev_s.leaderboard(
                df,
                average_seeds=average_seeds,
                include_baseline_advantage=True,
                baseline_method="A",
            )
            assert list(lb.index) == ["A", "B", "C"]
            assert lb.loc["A", "rank"] == pytest.approx(1.0)
            assert lb.loc["C", "rank"] == pytest.approx(3.0)
            assert lb.loc["B", "baseline_advantage"] == pytest.approx(-0.5)


class TestMetricSpecRemoval:
    def test_score_if_remove_method(self, ev, data):
        # metric_spec_rank.compute/score route through `_resolve_groupby_columns`
        rpt = ev.compute_results_per_task(data)
        spec = ev.metric_spec_rank()
        # C's weighted-mean rank on the full set is 3.0
        assert ev.score_if_remove_method(spec, rpt, method_1="C", method_2="C") == pytest.approx(3.0)
        # removing B promotes C from rank 3 to rank 2 on every task
        assert ev.score_if_remove_method(spec, rpt, method_1="C", method_2="B") == pytest.approx(2.0)

    def test_greedy_score_matrix(self, ev, data):
        rpt = ev.compute_results_per_task(data)
        spec = ev.metric_spec_rank()
        gm = ev.greedy_score_matrix(spec, rpt, methods_1=["C"])
        # greedy removes the other two methods one at a time; C's rank improves
        # 3 -> 2 -> 1. Which removed method maps to which score depends on tie-break
        # order, so assert on the set of resulting scores instead.
        assert set(gm.index) == {"A", "B"}
        assert sorted(gm["C"].dropna().tolist()) == pytest.approx([1.0, 2.0])


class TestValidation:
    def test_missing_column_raises(self, ev, data):
        with pytest.raises(ValueError):
            ev.verify_data(data.drop(columns=["time_infer_s"]))

    def test_negative_error_raises(self, ev, data):
        bad = data.copy()
        bad.loc[0, "metric_error"] = -0.5
        with pytest.raises(ValueError):
            ev.verify_data(bad)

    def test_non_dense_raises(self, ev, data):
        # drop the (A, t1) row -> not every method has every task
        with pytest.raises(AssertionError):
            ev.verify_data(data.iloc[1:])

    def test_clean_data_clips_tiny_negative(self, ev, data):
        d = data.copy()
        d.loc[0, "metric_error"] = -1e-16  # within negative_error_threshold (-1e-15)
        cleaned = ev.clean_data(d)
        assert cleaned["metric_error"].min() >= 0

    def test_fillna_worst(self, ev, data):
        missing = data[~((data["method"] == "A") & (data["task"] == "t1"))].copy()
        filled = ev.fillna_data(missing, fillna_method="worst")
        assert len(filled) == 9  # all (method, task) combos restored
        a_t1 = filled[(filled["method"] == "A") & (filled["task"] == "t1")]
        assert len(a_t1) == 1
        # worst error on t1 among present rows is C = 0.4
        assert a_t1["metric_error"].iloc[0] == pytest.approx(0.4)


class TestMetricRegistry:
    def test_metrics_arg_equivalent_to_flags(self, ev, data):
        # Selecting metrics via `metrics=` yields an identical leaderboard to the
        # equivalent `include_*` flags (elo omitted: its bootstrap CI is not seeded).
        keys = [
            "winrate",
            "improvability",
            "mrr",
            "rank_counts",
            "baseline_advantage",
            "frontier_advantage",
            "relative_error",
            "skill_score",
        ]
        by_flags = ev.leaderboard(
            data,
            include_elo=False,
            include_winrate=True,
            include_improvability=True,
            include_mrr=True,
            include_rank_counts=True,
            include_baseline_advantage=True,
            include_frontier_advantage=True,
            include_relative_error=True,
            include_skill_score=True,
            baseline_method="A",
        )
        by_metrics = ev.leaderboard(data, metrics=keys, baseline_method="A")
        pd.testing.assert_frame_equal(by_flags, by_metrics)

    def test_metrics_overrides_flags(self, ev, data):
        # `metrics=` takes precedence; the include_* flags are ignored for selection.
        lb = ev.leaderboard(data, include_winrate=True, include_improvability=True, metrics=["elo"])
        assert "elo" in lb.columns
        assert "winrate" not in lb.columns
        assert "improvability" not in lb.columns

    def test_metrics_empty_keeps_only_rank(self, ev, data):
        lb = ev.leaderboard(data, metrics=[])
        assert "rank" in lb.columns  # rank is always emitted
        for absent in ("elo", "winrate", "improvability", "mrr"):
            assert absent not in lb.columns

    def test_unknown_metric_raises(self, ev, data):
        with pytest.raises(ValueError, match="Unknown leaderboard metric"):
            ev.leaderboard(data, metrics=["bogus"])

    def test_baseline_metric_skipped_without_baseline(self, ev, data):
        # `relative_error` needs a baseline; without one it is silently skipped (parity
        # with the legacy flag behaviour), while non-baseline metrics still appear.
        lb = ev.leaderboard(data, metrics=["relative_error", "winrate"])
        assert "winrate" in lb.columns
        assert "relative_error" not in lb.columns
