from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from autogluon.core.models import AbstractModel

pytest.importorskip("ftfm")

from tabarena.models.ftfm.model import FTFMModel


def _frames(n_rows: int = 160, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n_rows, 4)), columns=list("abcd"))
    X["cat"] = pd.Series(rng.choice(["u", "v", "w"], size=n_rows), dtype="category")
    y = pd.Series((X["a"] + (X["cat"] == "u") > 0).astype(int))
    return X.iloc[:120], y.iloc[:120], X.iloc[120:]


def _fit(tmp_path, name: str, X, y, cache_context: bool) -> FTFMModel:
    model = FTFMModel(
        path=str(tmp_path / name),
        name=name,
        problem_type="binary",
        hyperparameters={"n_estimators": 2, "cache_context": cache_context},
    )
    model.fit(X=X, y=y, num_cpus=2, num_gpus=0)
    return model


def test_uncached_predicts_like_cached_and_keeps_no_context(tmp_path):
    X, y, X_test = _frames()
    cached = _fit(tmp_path, "cached", X, y, cache_context=True)
    uncached = _fit(tmp_path, "uncached", X, y, cache_context=False)

    assert cached.model.contexts_ is not None
    assert getattr(uncached.model, "contexts_", None) is None
    np.testing.assert_allclose(uncached.predict_proba(X_test), cached.predict_proba(X_test), rtol=1e-5, atol=1e-6)
    # The contexts built for that predict are released again; a second call rebuilds them.
    assert uncached.model.contexts_ is None
    np.testing.assert_allclose(uncached.predict_proba(X_test), cached.predict_proba(X_test), rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    not hasattr(AbstractModel, "model_tensors_file_name"), reason="this AutoGluon keeps a model's tensors in model.pkl"
)
def test_view_batched_contexts_save_at_their_own_size(tmp_path):
    """On CUDA the library prefills views in one batch and keeps each member's context as a slice of it.

    A pickle wrote the whole batch once per slice; AutoGluon's tensor file writes it once.
    """
    X, y, X_test = _frames(n_rows=300)
    saved_bytes, predictions = {}, {}
    for view_batch_size in (4, 1):
        model = FTFMModel(
            path=str(tmp_path / f"views_{view_batch_size}"),
            name="m",
            problem_type="binary",
            hyperparameters={"n_estimators": 4, "view_batch_size": view_batch_size, "offload_contexts": True},
        )
        model.fit(X=X, y=y, num_cpus=2, num_gpus=0)
        predictions[view_batch_size] = model.predict_proba(X_test)
        path = model.save()
        saved_bytes[view_batch_size] = sum(os.path.getsize(os.path.join(path, name)) for name in os.listdir(path))

    np.testing.assert_allclose(predictions[4], predictions[1], rtol=1e-5, atol=1e-6)
    assert saved_bytes[4] < 1.1 * saved_bytes[1]
