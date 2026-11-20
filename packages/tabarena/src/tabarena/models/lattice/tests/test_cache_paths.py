from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("model_name", ["LatticeModel", "LatticeTinyModel"])
@pytest.mark.parametrize("problem_type", ["binary", "regression"])
def test_cache_and_reload_predictions(tmp_path, model_name, problem_type):
    """Fold and refit paths agree, and each survives AutoGluon serialization."""
    pytest.importorskip("lattice")
    from tabarena.models.lattice import model as lattice_model

    model_cls = getattr(lattice_model, model_name)
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    x["cat"] = pd.Categorical(rng.choice(["a", "b", "c"], size=len(x)))
    query = x.iloc[60:].copy()
    train = x.iloc[:60].copy()
    target = pd.Series((train.a > 0).astype(int) if problem_type == "binary" else train.a + train.b)
    predictions = {}
    for cache in (False, True):
        model = model_cls(
            path=str(tmp_path / str(cache)),
            name="model",
            problem_type=problem_type,
            hyperparameters={"n_estimators": 1, "kv_cache": cache},
        )
        model.fit(X=train, y=target, num_cpus=2, num_gpus=0)
        before = model.predict_proba(query)
        model.save()
        restored = model_cls.load(path=model.path)
        np.testing.assert_allclose(restored.predict_proba(query), before, rtol=1e-5, atol=1e-5)
        predictions[cache] = before
    np.testing.assert_allclose(predictions[True], predictions[False], rtol=1e-4, atol=1e-4)
