from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

PROBLEM_TYPES = pytest.mark.parametrize(
    ("problem_type", "n_classes"), [("binary", 2), ("multiclass", 4), ("multiclass", 12), ("regression", 0)]
)


def _table(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    X["cat"] = pd.Categorical(rng.choice(list("abc"), size=n))
    X.loc[rng.random(n) < 0.05, "f1"] = np.nan
    return X


def _target(X: pd.DataFrame, n_classes: int) -> pd.Series:
    signal = X["f0"] + 0.5 * X["f1"].fillna(0) - 0.3 * X["f2"]
    if n_classes == 0:
        return signal
    edges = np.quantile(signal, np.linspace(0, 1, n_classes + 1)[1:-1])
    return pd.Series(np.searchsorted(edges, signal), index=X.index)


def _fitted(tmp_path, name: str, problem_type: str, n_classes: int, **hyperparameters):
    pytest.importorskip("inpriora")
    from tabarena.models.inpriora.model import InprioraModel

    X = _table(300, seed=0)
    model = InprioraModel(
        path=str(tmp_path),
        name=name,
        problem_type=problem_type,
        hyperparameters={"n_estimators": 2, "ag_args_fit": {"num_gpus": 0}, **hyperparameters},
    )
    model.fit(X=X, y=_target(X, n_classes), num_gpus=0)
    return model


@PROBLEM_TYPES
def test_cached_and_uncached_predictions_agree(tmp_path, problem_type, n_classes):
    """The fold models predict uncached, the refit model from the context cache; both give the same predictions.

    The out-of-fold scores the ensembles are built on come from the first path and the test scores from the
    second, so they must measure the same predictor. The test also covers a save and load of each.
    """
    from tabarena.models.inpriora.model import InprioraModel

    X_query = _table(150, seed=1)
    predictions = {}
    for cache in (False, True, "repr"):
        model = _fitted(tmp_path / str(cache), "m", problem_type, n_classes, cache=cache)
        before = model.predict_proba(X_query)
        model.save()
        loaded = InprioraModel.load(path=model.path)
        np.testing.assert_allclose(loaded.predict_proba(X_query), before, atol=1e-6)
        predictions[cache] = before
    np.testing.assert_allclose(predictions[True], predictions[False], atol=1e-5)
    np.testing.assert_allclose(predictions["repr"], predictions[False], atol=1e-5)


def test_cpu_move_runs_preprocessing_on_the_cpu(tmp_path):
    """A model moved to the CPU runs the fitted power transform and ranks there."""
    model = _fitted(tmp_path, "m", "binary", 2, cache=True)
    ensemble = model.model.ensemble_
    model.set_device("cpu")
    assert ensemble.yeojohnson_device is None
    assert all(pipeline.yeojohnson_device is None for pipeline in ensemble.preprocessors_.values())
