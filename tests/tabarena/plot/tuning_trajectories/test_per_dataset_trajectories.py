from __future__ import annotations

import pandas as pd
import pytest

from tabarena.plot.tuning_trajectories.plot_pareto_over_tuning_time import (
    compute_per_dataset_trajectories,
)


class _StubContext:
    """Enough of ``TabArenaContext`` for the no-subset, no-fillna path."""

    task_metadata_collection = None
    subset_predicates: dict = {}


def _combined_data() -> pd.DataFrame:
    """Two datasets x two splits x two trajectory points, plus one always-worst method."""
    rows = []
    errors = {
        ("CatBoost-1", "CatBoost", 1): {"alpha": [0.30, 0.30], "beta": [1.00, 1.00]},
        ("CatBoost-5", "CatBoost", 5): {"alpha": [0.10, 0.20], "beta": [0.50, 0.50]},
        ("KNN-1", "KNN", 1): {"alpha": [0.90, 0.90], "beta": [4.00, 4.00]},
    }
    for (method, config_type, n_configs), per_dataset in errors.items():
        for dataset, values in per_dataset.items():
            for fold, value in enumerate(values):
                rows.append(
                    {
                        "method": method,
                        "config_type": config_type,
                        "n_configs": n_configs,
                        "dataset": dataset,
                        "fold": fold,
                        "metric_error": value,
                        "time_train_s": 4.0 * n_configs,
                        "time_infer_s": 0.25,
                        "time_train_s_per_1K": 2.0 * n_configs,
                        "time_infer_s_per_1K": 0.1,
                        "imputed": False,
                    },
                )
    return pd.DataFrame(rows)


def _methods_map() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["CatBoost-1", "CatBoost-5", "KNN-1"],
            "n_configs": [1, 5, 1],
            "config_type": ["CatBoost", "CatBoost", "KNN"],
        },
    ).set_index("method")


def _compute(**kwargs) -> pd.DataFrame:
    return compute_per_dataset_trajectories(
        _combined_data(),
        tabarena_context=_StubContext(),
        methods_map=_methods_map(),
        fillna_method=None,
        exclude_imputed=False,
        **kwargs,
    )


def test_improvability_is_relative_to_the_best_on_that_split():
    """Each dataset gets its own reference point, averaged over that dataset's splits.

    On ``alpha`` the best errors are 0.10 and 0.20, so the one-config CatBoost point is
    1 - 0.10/0.30 behind on the first split and 1 - 0.20/0.30 on the second.
    """
    out = _compute().set_index(["dataset", "method", "n_configs"])
    expected = ((1 - 0.10 / 0.30) + (1 - 0.20 / 0.30)) / 2 * 100
    assert out.loc[("alpha", "CatBoost", 1), "imp"] == pytest.approx(expected)
    # The best method on a dataset is 0% behind itself.
    assert out.loc[("alpha", "CatBoost", 5), "imp"] == pytest.approx(0.0)
    # `beta` has its own reference, so the same method has a different gap there.
    assert out.loc[("beta", "CatBoost", 1), "imp"] == pytest.approx(50.0)


def test_runtimes_and_errors_are_per_dataset():
    out = _compute().set_index(["dataset", "method", "n_configs"])
    assert out.loc[("alpha", "CatBoost", 5), "err"] == pytest.approx(0.15)
    assert out.loc[("alpha", "CatBoost", 5), "train_s"] == pytest.approx(20.0)
    assert out.loc[("alpha", "CatBoost", 5), "x_train"] == pytest.approx(10.0)


def test_hidden_methods_are_dropped_but_still_set_the_reference():
    """Hiding a method removes its row, not its results.

    The aggregate figure hides the weak baselines the same way, after the leaderboard has been
    computed over the full field, so the two surfaces report the same improvability.
    """
    out = _compute(hidden_methods=["KNN"])
    assert set(out["method"]) == {"CatBoost"}
    kept = out.set_index(["dataset", "method", "n_configs"])
    assert kept.loc[("beta", "CatBoost", 1), "imp"] == pytest.approx(50.0)


def test_rename_map_applies_after_the_hidden_filter():
    out = _compute(method_rename_map={"CatBoost": "CatBoost (renamed)"}, hidden_methods=["KNN"])
    assert set(out["method"]) == {"CatBoost (renamed)"}


def test_empty_input_returns_the_expected_columns():
    out = compute_per_dataset_trajectories(
        _combined_data().iloc[:0],
        tabarena_context=_StubContext(),
        methods_map=_methods_map(),
        fillna_method=None,
        exclude_imputed=False,
    )
    assert out.empty
    assert list(out.columns) == ["dataset", "method", "n_configs", "x_train", "x_infer", "err", "imp", "imputed"]
