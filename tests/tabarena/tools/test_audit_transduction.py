from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tabarena.tools.audit_transduction import (
    N_PROBE,
    PROBES,
    audit_transduction,
    diagnose,
    make_probe_tables,
    run_probes,
)

if TYPE_CHECKING:
    import pandas as pd


def _rowwise(X: pd.DataFrame) -> np.ndarray:
    return np.tanh(X["num0"].to_numpy() + 0.001 * X["num_large"].to_numpy())


def _batch_mean(X: pd.DataFrame) -> np.ndarray:
    """Normalizes with the mean of the rows being predicted."""
    return _rowwise(X) - X["num0"].mean()


def _count_seeded(X: pd.DataFrame) -> np.ndarray:
    """Adds noise drawn from an RNG whose stream depends on the number of rows."""
    rng = np.random.default_rng(0)
    rng.normal(size=len(X))
    return _rowwise(X) + 0.01 * rng.normal()


def _duplicate_count(X: pd.DataFrame) -> np.ndarray:
    """Marks a row that repeats an earlier row of the batch."""
    return _rowwise(X) + 0.1 * X.duplicated().to_numpy()


def _position(X: pd.DataFrame) -> np.ndarray:
    return _rowwise(X) + 1e-3 * np.arange(len(X))


def _failed(predict) -> set[str]:
    _, _, X_query, X_decoy = make_probe_tables("binary")
    diffs = run_probes(predict, X_query, X_decoy)
    return {name for name in PROBES if diffs[name] > 1e-5}


@pytest.mark.parametrize(
    ("predict", "expected"),
    [
        (_rowwise, set()),
        (_batch_mean, {"alone", "batch", "size", "content", "twin"}),
        (_count_seeded, {"alone", "batch", "size"}),
        (_duplicate_count, {"twin"}),
        (_position, {"alone", "twin", "order"}),
    ],
)
def test_each_probe_separates_its_cause(predict, expected):
    assert _failed(predict) == expected


def test_probe_tables_reach_the_known_paths():
    X_train, y_train, X_query, X_decoy = make_probe_tables("multiclass")
    assert set(y_train) == {0, 1, 2}
    assert X_train["const_train"].nunique() == 1 == X_query["const_train"].nunique()
    assert X_train["const_nan_train"].isna().any()
    assert X_train["const_nan_train"].nunique() == 1 == X_query["const_nan_train"].nunique()
    assert X_decoy["const_nan_train"].std() > 0
    assert set(X_query["cat"]) - set(X_train["cat"])
    assert X_query["nan_in_query"].isna().all()
    assert not X_query.iloc[:N_PROBE].duplicated().any()


def test_diagnose_names_the_cause():
    assert diagnose([]).startswith("OK")
    assert "number of rows" in diagnose(["alone", "batch", "size"])
    assert "duplicate" in diagnose(["twin"])
    assert "values reach" in diagnose(["content", "twin"])
    assert "not deterministic" in diagnose(["repeat", "alone"])


@pytest.mark.parametrize("problem_type", ["binary", "regression"])
def test_inductive_autogluon_model_passes(problem_type):
    from autogluon.tabular.models import LinearModel

    data = audit_transduction(LinearModel, problem_type=problem_type, hyperparameters={})
    assert data["failed"] == []
