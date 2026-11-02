"""TabPack wrapper logic the registry fit test does not reach: its preprocessing, a class missing from
a fold's training rows, and the regression target scale. Cheap CPU fits (two members, one epoch).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from autogluon.core.metrics import get_metric
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

pytest.importorskip("tabpackage")

from tabarena.models.tabpack.model import TabPackModel


def _frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "num_nan": np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n) * 1000),
            "flag": rng.choice(["yes", "no"], n),
            "cat": rng.choice(["a", "b", "c", None], n).astype(object),
        }
    )


def _fit(tmp_path, X, y, problem_type, metric, X_val=None, y_val=None, **fit_kwargs) -> TabPackModel:
    """Fit on AutoGluon's default features, as the benchmark does (bool and category special types included)."""
    generator = AutoMLPipelineFeatureGenerator(verbosity=0)
    X = generator.fit_transform(X)
    X_val = None if X_val is None else generator.transform(X_val)
    model = TabPackModel(
        path=str(tmp_path / problem_type) + "/",
        name="TabPack",
        problem_type=problem_type,
        eval_metric=get_metric(metric, problem_type),
        hyperparameters={"n_models": 2, "max_epochs": 1},
    )
    model.generator = generator  # the tests transform new rows the same way
    model.fit(
        X=X,
        y=y,
        X_val=X_val,
        y_val=y_val,
        feature_metadata=generator.feature_metadata,
        num_cpus=1,
        num_gpus=0,
        **fit_kwargs,
    )
    return model


def test_preprocessing_meets_the_library_contract_on_new_rows(tmp_path):
    X = _frame(200)
    y = pd.Series((X["num"] > 0).astype(int))
    model = _fit(tmp_path, X.iloc[:150], y.iloc[:150], "binary", "roc_auc", X.iloc[150:], y.iloc[150:])

    X_new = _frame(5, seed=1)
    X_new.loc[0, "num"] = np.inf
    X_new.loc[1, "num_nan"] = np.nan
    X_new["cat"] = [None, "unseen", "b", "a", "b"]
    X_new = model.generator.transform(X_new)
    out = model.preprocess(X_new)

    num_cols = ["num", "num_nan"]
    assert (out[num_cols].dtypes == np.float32).all()
    assert np.isfinite(out[num_cols].to_numpy()).all()
    assert out["cat"].dtype == "string"
    # AutoGluon codes a missing training value as a category of its own and an unseen value as NaN.
    assert out["cat"].tolist()[:2] == ["0", "__missing__"]
    assert out["flag"].dtype == "string"  # bool features are categorical, as in the paper
    assert out["__tabpack_missing__num_nan"].tolist()[1] == "1"
    assert "__tabpack_missing__num" not in out  # no missing values in "num" during training
    assert model.predict_proba(X_new).shape == (5,)
    assert not model.model.tabpack_._checkpoints.state_dicts  # the Developer fix dropped them


def test_a_class_missing_from_the_training_rows_keeps_its_probability_column(tmp_path):
    X = _frame(300)
    y = pd.Series(np.digitize(X["num"], [-0.5, 0.5]))  # classes 0, 1, 2
    train = y != 2
    # In a bagged fit the parent hands every fold model the full class count.
    model = _fit(tmp_path, X[train], y[train], "multiclass", "log_loss", X.iloc[:60], y.iloc[:60], num_classes=3)

    proba = model.predict_proba(model.generator.transform(X.iloc[:10]))
    assert proba.shape == (10, 3)
    np.testing.assert_allclose(proba[:, 2], 0.0, atol=1e-6)  # AutoGluon clips probabilities away from 0
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


def test_regression_predictions_are_on_the_target_scale(tmp_path):
    X = _frame(300)
    y = pd.Series(5000.0 + 100.0 * X["num"])
    model = _fit(tmp_path, X.iloc[:200], y.iloc[:200], "regression", "rmse", X.iloc[200:], y.iloc[200:])

    pred = model.predict(model.generator.transform(X.iloc[200:]))
    assert abs(pred.mean() - 5000.0) < 200.0
