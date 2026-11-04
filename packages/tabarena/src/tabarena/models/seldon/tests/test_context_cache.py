from __future__ import annotations

import os

import pytest

pytest.importorskip("neuralk_oss")


def _fitted(tmp_path):
    import numpy as np
    import pandas as pd

    from tabarena.models.seldon.model import SeldonMiniModel

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((X["f0"] > 0).astype(int))
    model = SeldonMiniModel(path=str(tmp_path), name="m", problem_type="binary", eval_metric="log_loss")
    model.fit(X=X, y=y, num_gpus=0)
    return model, X


def test_context_cache_is_saved_beside_the_pickle(tmp_path):
    import numpy as np

    from tabarena.models.seldon.model import SeldonMiniModel

    model, X = _fitted(tmp_path)
    expected = model.predict_proba(X.iloc[:20])
    cache_bytes = model.model.classifier.estimator._cache.size()
    size_estimate = model._get_pickled_size()
    path = model.save()

    # AutoGluon writes the model's tensors, the context cache among them, beside model.pkl.
    pickle_bytes = os.path.getsize(os.path.join(path, SeldonMiniModel.model_file_name))
    tensor_file_bytes = os.path.getsize(os.path.join(path, SeldonMiniModel.model_tensors_file_name))
    assert pickle_bytes < cache_bytes / 10
    assert tensor_file_bytes >= cache_bytes
    assert cache_bytes <= size_estimate <= pickle_bytes + tensor_file_bytes

    loaded = SeldonMiniModel.load(path)
    np.testing.assert_allclose(loaded.predict_proba(X.iloc[:20]), expected, atol=1e-6)


def test_resave_of_a_loaded_model(tmp_path):
    import numpy as np

    from tabarena.models.seldon.model import SeldonMiniModel

    model, X = _fitted(tmp_path)
    expected = model.predict_proba(X.iloc[:20])
    loaded = SeldonMiniModel.load(model.save())
    # refit_full re-saves a loaded model over the files it was loaded from.
    loaded.save()

    np.testing.assert_allclose(loaded.predict_proba(X.iloc[:20]), expected, atol=1e-6)
