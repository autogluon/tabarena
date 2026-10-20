from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
compacttab = pytest.importorskip("compacttab")

from tabarena.models.compacttab.model import CompactTabHybridModel, CompactTabModel


def test_compacttab_initialization():
    model = CompactTabModel(problem_type="binary")
    assert model.ag_key == "TA-COMPACTTAB"

    hybrid_model = CompactTabHybridModel(problem_type="regression")
    assert hybrid_model.ag_key == "TA-COMPACTTAB-HYBRID"


def test_compacttab_fit_predict_binary(tmp_path):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(40, 4)), columns=[f"feat_{i}" for i in range(4)])
    y = pd.Series((X["feat_0"] > 0).astype(int))

    model = CompactTabModel(
        path=str(tmp_path / "compacttab_bin"),
        name="test_compacttab",
        problem_type="binary",
        eval_metric="roc_auc",
        hyperparameters={"max_epochs": 5, "batch_size": 16},
    )

    model.fit(X=X, y=y)
    preds = model.predict(X)
    probs = model.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_compacttab_fit_predict_regression(tmp_path):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(40, 4)), columns=[f"feat_{i}" for i in range(4)])
    y = pd.Series(X["feat_0"] * 2.0 + X["feat_1"])

    model = CompactTabModel(
        path=str(tmp_path / "compacttab_reg"),
        name="test_compacttab_reg",
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        hyperparameters={"max_epochs": 5, "batch_size": 16},
    )

    model.fit(X=X, y=y)
    preds = model.predict(X)

    assert len(preds) == len(X)
    assert preds.ndim == 1


def test_compacttab_hybrid_fit_predict_binary(tmp_path):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(40, 4)), columns=[f"feat_{i}" for i in range(4)])
    y = pd.Series((X["feat_0"] > 0).astype(int))

    model = CompactTabHybridModel(
        path=str(tmp_path / "compacttab_hybrid_bin"),
        name="test_compacttab_hybrid",
        problem_type="binary",
        eval_metric="roc_auc",
        hyperparameters={"max_epochs": 5, "batch_size": 16, "cv_folds": 2},
    )

    model.fit(X=X, y=y)
    preds = model.predict(X)
    probs = model.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
