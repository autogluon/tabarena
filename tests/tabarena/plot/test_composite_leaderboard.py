"""Tests for the composite cross-subset leaderboard.

Runs the full pipeline against small synthetic compact website-format
leaderboards (Agg backend, tmp_path output): tuning-variant collapsing,
method exclusion, top-N selection, ``&lite`` stripping, subset/column
ordering, and the CSV + PNG artifacts.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

from tabarena.plot.composite_leaderboard import (
    collapse_tuning_variants,
    generate_composite_leaderboard,
)

_METHODS = [
    "TabPFN-3 (default)",
    "TabPFN-3 (tuned)",
    "TabPFN-3 (tuned + ensembled)",
    "LightGBM (default)",
    "LightGBM (tuned)",
    "RealMLP (tuned)",
    "AutoGluon 1.4 (extreme, 4h)",
]


def _leaderboard(seed: int) -> pd.DataFrame:
    """One synthetic subset leaderboard in the compact website format."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Model": _METHODS,
            "Elo": rng.uniform(1000, 1800, len(_METHODS)).round(0),
            "Impro%": rng.uniform(0, 10, len(_METHODS)).round(3),
            "TrainTime (s/1K)": rng.uniform(1, 100, len(_METHODS)).round(2),
            "PredTime (s/1K)": rng.uniform(0.1, 10, len(_METHODS)).round(3),
        }
    )


def test_generate_composite_leaderboard(tmp_path):
    leaderboards = {
        "regression": _leaderboard(1),
        "all": _leaderboard(2),
        "binary": _leaderboard(3),
    }
    composite = generate_composite_leaderboard(
        leaderboards,
        output_dir=tmp_path,
        excluded_method_prefixes=("AutoGluon 1.4",),
    )

    # The sort-by subset ("all") is hoisted first, remaining follow SUBSET_ORDER.
    assert list(composite.columns) == ["all", "binary", "regression"]
    assert composite.index.names == ["method", "metric"]
    methods = composite.index.get_level_values("method")
    # T+E collapses its (default)/(tuned) siblings; lone suffixes pass through
    # with (default) stripped.
    assert set(methods) == {"TabPFN-3 (T+E)", "LightGBM", "LightGBM (tuned)", "RealMLP (tuned)"}
    # Methods ordered by descending Elo on the sort-by subset.
    elo_all = composite.xs("Elo", level="metric")["all"]
    assert list(elo_all) == sorted(elo_all, reverse=True)

    for name in (
        "composite_leaderboard.csv",
        "composite_leaderboard.png",
        "composite_leaderboard_elo.png",
        "composite_leaderboard_improv.png",
    ):
        assert (tmp_path / name).exists(), name


def test_generate_composite_leaderboard_top_n_and_lite(tmp_path):
    leaderboards = {"all&lite": _leaderboard(1), "binary&lite": _leaderboard(2)}
    composite = generate_composite_leaderboard(
        leaderboards,
        output_dir=tmp_path,
        top_n=2,
        save_png=False,
    )
    # `&lite` is auto-stripped (every subset carries it) and top-2 methods kept.
    assert list(composite.columns) == ["all", "binary"]
    assert composite.index.get_level_values("method").nunique() == 2
    assert not (tmp_path / "composite_leaderboard.png").exists()
    assert (tmp_path / "composite_leaderboard.csv").exists()


def test_generate_composite_leaderboard_sort_by_fallback(tmp_path):
    # Without an "all" subset, sort_by=None falls back to the first subset.
    composite = generate_composite_leaderboard(
        {"binary": _leaderboard(1), "regression": _leaderboard(2)},
        output_dir=tmp_path,
        save_png=False,
    )
    assert next(iter(composite.columns)) == "binary"


def test_generate_composite_leaderboard_missing_metric_raises(tmp_path):
    lb = _leaderboard(1).drop(columns=["Impro%"])
    with pytest.raises(KeyError, match="Impro%"):
        generate_composite_leaderboard({"all": lb}, output_dir=tmp_path)


def test_collapse_tuning_variants_keeps_lone_tuned():
    lb = _leaderboard(0).set_index("Model")
    collapsed = collapse_tuning_variants({"all": lb})["all"]
    # RealMLP has (tuned) but no T+E sibling → passes through unchanged.
    assert "RealMLP (tuned)" in collapsed.index
    # TabPFN-3's (default)/(tuned) are absorbed by its T+E row.
    assert "TabPFN-3 (T+E)" in collapsed.index
    assert "TabPFN-3 (default)" not in collapsed.index
    assert "TabPFN-3 (tuned)" not in collapsed.index
