from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

PROBLEM_TYPES = pytest.mark.parametrize("problem_type", ["binary", "regression"])


def _table(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    X["cat"] = pd.Categorical(rng.choice(list("abc"), size=n))
    return X


def _fitted(tmp_path, problem_type: str, **hyperparameters):
    pytest.importorskip("inference.v2_0.predictor")
    from tabarena.models.limix_2.model import LimiX2Model

    X = _table(200, seed=0)
    y = X["f0"] + 0.1 * X["f1"] if problem_type == "regression" else (X["f0"] > 0).astype(int)
    model = LimiX2Model(path=str(tmp_path), name="m", problem_type=problem_type, hyperparameters=hyperparameters)
    model.fit(X=X, y=y, num_gpus=0)
    return model


@PROBLEM_TYPES
def test_predict_leaves_no_query_state(tmp_path, problem_type):
    """A predict leaves the predictor as fitted: same pickle size, no per-row state, same predictions."""
    model = _fitted(tmp_path, problem_type, n_estimators=8)
    predictor = model.model
    size_before = len(pickle.dumps(predictor.preprocess_pipelines))
    X_query = _table(2000, seed=1)
    first = model.predict_proba(X_query)
    assert len(pickle.dumps(predictor.preprocess_pipelines)) == size_before
    assert all(getattr(predictor, attr, None) is None for attr in ("_reg_decode_x_train", "_reg_decode_y_train"))
    assert not any(hasattr(step, "train_hashes_") for pipe in predictor.preprocess_pipelines for step in pipe)
    np.testing.assert_allclose(model.predict_proba(X_query), first, atol=1e-6)


def test_use_data_cache_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="use_data_cache"):
        _fitted(tmp_path, "binary", n_estimators=1, use_data_cache=True)


def test_many_class_wrapper_restores_the_frame():
    """``ManyClassClassifier`` passes an object ndarray; the sub-estimator gets the typed frame back."""
    from tabarena.models.limix_2.model import _LimiX2SklearnWrapper

    X = _table(10, seed=0).astype({"cat": object})
    wrapper = _LimiX2SklearnWrapper(
        device_str="cpu",
        model_path=None,
        inference_config=None,
        preprocess_num_jobs=1,
        hps={},
        columns=list(X.columns),
        dtypes=X.dtypes.to_dict(),
    )
    restored = wrapper._as_frame(X.to_numpy(dtype=object))
    pd.testing.assert_frame_equal(restored, X)
    assert wrapper.get_params()["dtypes"] == X.dtypes.to_dict()
