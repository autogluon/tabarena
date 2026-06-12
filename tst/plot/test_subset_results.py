"""Tests for the per-family / per-model subset-overview plots.

Renders against small synthetic leaderboards (Agg backend, tmp_path output) — the goal is to
exercise the full draw path for both plot kinds × both default metrics, including the contender
line, imputed hatching, and graceful skipping when no configured methods are present.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

from tabarena.plot.subset_results import MetricSpec, plot_subset_results

_SUBTYPE_SUFFIX = {"tuned_ensemble": "tuned + ensemble", "default": "default"}


def _leaderboard(seed: int) -> pd.DataFrame:
    """One synthetic subset leaderboard with the columns the plots consume."""
    rng = np.random.default_rng(seed)
    rows = []
    methods = [
        ("RealMLP", "tuned_ensemble"),
        ("RealMLP", "default"),
        ("CatBoost", "tuned_ensemble"),
        ("CatBoost", "default"),
        ("RandomForest", "default"),
        ("TabPFN-2.6", "default"),  # TFM family; flagged imputed below
        ("TabPFN-3", "default"),  # the contender
    ]
    for name, subtype in methods:
        elo = float(rng.uniform(1000, 1400))
        improvability = float(rng.uniform(0.02, 0.3))
        rows.append(
            {
                "method": f"{name} ({_SUBTYPE_SUFFIX[subtype]})",
                "method_subtype": subtype,
                "elo": elo,
                "elo+": 25.0,
                "elo-": 20.0,
                "improvability": improvability,
                "improvability+": 0.02,
                "improvability-": 0.015,
                "imputed": 0.3 if name == "TabPFN-2.6" else 0.0,
                "n_datasets_total": 42,
            },
        )
    return pd.DataFrame(rows)


@pytest.fixture
def leaderboards() -> dict[str, pd.DataFrame]:
    return {label: _leaderboard(seed) for seed, label in enumerate(["random", "tiny", "full"])}


def test_renders_all_kinds_and_metrics_with_contender(leaderboards, tmp_path):
    saved = plot_subset_results(leaderboards, tmp_path, contenders=["TabPFN-3"])
    assert sorted(saved) == [
        "per_family_elo",
        "per_family_improvability",
        "per_model_elo",
        "per_model_improvability",
    ]
    for path in saved.values():
        assert path.exists()
        assert path.with_suffix(".png").exists()


def test_unknown_metric_raises(leaderboards, tmp_path):
    with pytest.raises(ValueError, match="Unknown metric"):
        plot_subset_results(leaderboards, tmp_path, metrics=["winrate"])


def test_custom_metric_spec(leaderboards, tmp_path):
    spec = MetricSpec(
        name="winrate",
        column="elo",  # reuse an existing column; only the spec plumbing is under test
        higher_is_better=True,
        ylabel="Elo",
        ylabel_family="Elo (Best Model)",
        tick_step=100,
    )
    saved = plot_subset_results(leaderboards, tmp_path, plot_kinds=["per_model"], metrics=[spec])
    assert list(saved) == ["per_model_winrate"]


def test_per_family_skipped_when_no_known_methods(leaderboards, tmp_path):
    renamed = {label: lb.assign(method="Mystery-" + lb["method"]) for label, lb in leaderboards.items()}
    saved = plot_subset_results(renamed, tmp_path, metrics=["elo"])
    # per_family has nothing to draw (no family contains Mystery-* methods); per_model plots any method.
    assert list(saved) == ["per_model_elo"]
