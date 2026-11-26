"""Checks of the Causilo wrapper's own logic, on the CPU: the shared 1.5 network, category labels, class columns.

Run with ``pytest packages/tabarena/src/tabarena/models/causilo/tests``; they need causilo and its weights
(downloaded on the first run) and stay out of the default suite and CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("causilo")

from tabarena.models import _weights as registry
from tabarena.models.causilo.model import Causilo15Model, CausiloModel, CausiloModelBase


def _fit(cls, tmp_path, name, X, y, problem_type, *, num_classes=None, n_estimators=1, **hyperparameters):
    if n_estimators is not None:
        hyperparameters["n_estimators"] = n_estimators
    model = cls(
        path=str(tmp_path / name) + "/",
        name=name,
        problem_type=problem_type,
        hyperparameters={"device": "cpu", **hyperparameters},
    )
    model.fit(X=X, y=y, num_classes=num_classes, num_cpus=4, num_gpus=0)
    return model


@pytest.fixture
def binary_table():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(96, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=96) > 0).astype(int))
    return X, y


def test_fits_share_one_network_and_match_an_unshared_fit(tmp_path, binary_table):
    X, y = binary_table
    registry.release()
    registry.reset_stats()
    first = _fit(Causilo15Model, tmp_path, "first", X, y, "binary")
    second = _fit(Causilo15Model, tmp_path, "second", X, y, "binary")
    assert registry.report()["stats"]["misses"] == 1
    assert registry.report()["stats"]["hits"] == 1
    assert first.model._engine.model is second.model._engine.model
    assert first.get_info()["shared_weights"]["present_before_fit"] is False
    assert second.get_info()["shared_weights"]["present_before_fit"] is True

    unshared = _fit(
        Causilo15Model, tmp_path, "unshared", X, y, "binary", ag_args_fit={"share_pretrained_weights": False}
    )
    assert unshared.get_info()["shared_weights"] is None
    assert unshared.model._engine.model is not first.model._engine.model
    expected = unshared.predict_proba(X)
    np.testing.assert_array_equal(first.predict_proba(X), expected)

    first.save()
    reloaded = Causilo15Model.load(first.path)
    assert reloaded.model._engine.model is first.model._engine.model
    np.testing.assert_array_equal(reloaded.predict_proba(X), expected)


def test_class_setting_switches_sharing_off(tmp_path, binary_table):
    X, y = binary_table
    assert Causilo15Model.get_class_settings() is not CausiloModel.get_class_settings()
    Causilo15Model.set_class_settings(share_weights=False)
    try:
        model = _fit(Causilo15Model, tmp_path, "off", X, y, "binary")
    finally:
        Causilo15Model.set_class_settings(share_weights=True)
    assert model.get_info()["shared_weights"] is None
    assert CausiloModel.get_class_settings().share_weights is True


def test_fold_unseen_category_encodes_like_a_missing_one(tmp_path):
    rng = np.random.default_rng(1)
    dtype = pd.CategoricalDtype(categories=list(range(12)))  # AutoGluon's integer codes, fewer than 50
    codes = rng.integers(0, 12, 200)
    codes[codes == 5] = 6  # code 5 is in the dtype but not in the training rows
    X = pd.DataFrame({"c": pd.Categorical(codes, dtype=dtype), "x": rng.normal(size=200)})
    y = pd.Series(((codes % 3 == 0) ^ (X["x"] > 0)).astype(int))
    query = pd.DataFrame({"c": pd.Categorical([5, np.nan, 3], dtype=dtype), "x": [0.4, 0.4, -0.1]})

    model = _fit(Causilo15Model, tmp_path, "labels", X, y, "binary")
    proba = model.predict_proba(query)
    # Without a frequency feature an unseen text category and a missing value share one encoding.
    assert proba[0] == proba[1]

    original = Causilo15Model._preprocess
    Causilo15Model._preprocess = CausiloModelBase._preprocess
    try:
        codes_as_numbers = _fit(Causilo15Model, tmp_path, "codes", X, y, "binary").predict_proba(query)
    finally:
        Causilo15Model._preprocess = original
    assert proba[2] == codes_as_numbers[2]  # a seen category encodes as before
    assert proba[0] != codes_as_numbers[0]  # the integer code would be interpolated between 4 and 6


@pytest.mark.parametrize("cls", [CausiloModel, Causilo15Model])
def test_a_class_missing_from_the_fold_keeps_its_column(cls, tmp_path):
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
    y = pd.Series(rng.choice([0, 1, 3], 120))
    proba = _fit(cls, tmp_path, "classes", X, y, "multiclass", num_classes=4).predict_proba(X.iloc[:5])
    assert proba.shape == (5, 4)
    assert (proba[:, 2] == 0).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1, rtol=1e-6)


def test_sampled_contexts_share_the_network_and_reload(tmp_path, binary_table):
    X, y = binary_table  # 96 rows, so a 64-row budget samples one context per member
    registry.release()
    registry.reset_stats()
    model = _fit(Causilo15Model, tmp_path, "sampled", X, y, "binary", n_estimators=2, max_context=64)
    assert type(model.model._engine.state).__name__ == "SampledFitState"
    assert model.get_info()["shared_weights"] is not None
    expected = model.predict_proba(X)
    model.save()
    reloaded = Causilo15Model.load(model.path)
    np.testing.assert_array_equal(reloaded.predict_proba(X), expected)


def test_default_context_cap_matches_no_cap_below_it(tmp_path, binary_table):
    X, y = binary_table
    capped = _fit(Causilo15Model, tmp_path, "capped", X, y, "binary")
    assert capped.params["max_context"] == 200_000
    uncapped = _fit(Causilo15Model, tmp_path, "uncapped", X, y, "binary", max_context=None)
    np.testing.assert_array_equal(capped.predict_proba(X), uncapped.predict_proba(X))


@pytest.mark.parametrize(("problem_type", "members"), [("binary", 4), ("regression", 8)])
def test_default_ensemble_uses_the_library_preset(tmp_path, binary_table, problem_type, members):
    X, y = binary_table
    if problem_type == "regression":
        y = X["a"] + X["b"]
    default = _fit(Causilo15Model, tmp_path, "default", X, y, problem_type, n_estimators=None)
    explicit = _fit(Causilo15Model, tmp_path, "explicit", X, y, problem_type, n_estimators=members)
    assert default.model.n_estimators is None
    np.testing.assert_array_equal(default.predict_proba(X), explicit.predict_proba(X))


def test_category_labels_do_not_depend_on_the_other_categories(tmp_path):
    """BeyondArena keeps unseen categories (here a longer code) in the dtype at predict time."""
    rng = np.random.default_rng(3)
    levels = [4675, 9999, 117961, 311867]
    codes = rng.choice(levels, 160)
    X = pd.DataFrame({"c": pd.Categorical(codes, categories=levels), "x": rng.normal(size=160)})
    y = pd.Series(((codes == 9999) | (codes == 311867)).astype(int))
    model = _fit(Causilo15Model, tmp_path, "labels", X, y, "binary")
    query = X.iloc[:20]
    widened = query.assign(c=pd.Categorical(query["c"].astype(int), categories=[*levels, 1234567]))
    np.testing.assert_array_equal(model.predict_proba(widened), model.predict_proba(query))
    assert model.predict_proba(query).std() > 0.05  # the categories carry the label


def test_category_labels_survive_another_dtype_at_predict(tmp_path):
    """BeyondArena's preprocessing hands predict an ``object`` index of Python ints for an ``int64`` column."""
    rng = np.random.default_rng(4)
    levels = [4675, 9999, 117961, 311867]
    codes = rng.choice(levels, 160)
    X = pd.DataFrame({"c": pd.Categorical(codes, categories=levels), "x": rng.normal(size=160)})
    y = pd.Series(((codes == 9999) | (codes == 311867)).astype(int))
    model = _fit(Causilo15Model, tmp_path, "dtypes", X, y, "binary")
    query = X.iloc[:20]
    as_objects = query.assign(
        c=pd.Categorical([int(v) for v in query["c"]], categories=pd.Index([int(v) for v in levels], dtype=object))
    )
    assert as_objects["c"].cat.categories.dtype == object
    np.testing.assert_array_equal(model.predict_proba(as_objects), model.predict_proba(query))
