from __future__ import annotations

import os
import pickle

import pytest

pytest.importorskip("neuralk_oss")


PROBLEM_TYPES = pytest.mark.parametrize("problem_type", ["binary", "regression"])


def _fitted(tmp_path, problem_type):
    import numpy as np
    import pandas as pd

    from tabarena.models.seldon.model import SeldonMiniModel

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 5)), columns=[f"f{i}" for i in range(5)])
    if problem_type == "regression":
        y, eval_metric = X["f0"] + 0.1 * rng.normal(size=300), "root_mean_squared_error"
    else:
        y, eval_metric = pd.Series((X["f0"] > 0).astype(int)), "log_loss"
    model = SeldonMiniModel(path=str(tmp_path), name="m", problem_type=problem_type, eval_metric=eval_metric)
    model.fit(X=X, y=y, num_gpus=0)
    return model, X


@PROBLEM_TYPES
def test_context_cache_is_saved_beside_the_pickle(tmp_path, problem_type):
    import numpy as np

    from tabarena.models.seldon.model import SeldonMiniModel

    model, X = _fitted(tmp_path, problem_type)
    expected = model.predict_proba(X.iloc[:20])
    cache_bytes = model.model.estimator.estimator._cache.size()
    path = model.save()

    assert os.path.exists(os.path.join(path, SeldonMiniModel.cache_file_name))
    assert os.path.getsize(os.path.join(path, "model.pkl")) < cache_bytes
    assert model.model.estimator.estimator._cache is not None  # restored after the save
    with model._without_cache():
        pickled = len(pickle.dumps(model, protocol=4))
    assert pickled < model._get_pickled_size() <= pickled + cache_bytes + 1024

    loaded = SeldonMiniModel.load(path)
    np.testing.assert_allclose(loaded.predict_proba(X.iloc[:20]), expected, atol=1e-6)


@PROBLEM_TYPES
def test_resave_over_a_memory_mapped_cache(tmp_path, problem_type):
    import numpy as np

    from tabarena.models.seldon.model import SeldonMiniModel

    model, X = _fitted(tmp_path, problem_type)
    expected = model.predict_proba(X.iloc[:20])
    loaded = SeldonMiniModel.load(model.save())
    # On the CPU the loaded cache is memory-mapped from the file this save replaces.
    loaded.save()

    np.testing.assert_allclose(loaded.predict_proba(X.iloc[:20]), expected, atol=1e-6)
