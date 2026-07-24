from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabarena.models.tabpfn_3.model import TabPFN3Model, perturb_categorical_mask


def test_perturb_categorical_mask_extremes():
    cat_indices = [1, 3, 4]
    n_cols = 6

    mask, dropped = perturb_categorical_mask(cat_indices, n_cols, drop_prob=0.0, add_prob=0.0, seed=0)
    assert mask == cat_indices
    assert dropped == []

    mask, dropped = perturb_categorical_mask(cat_indices, n_cols, drop_prob=1.0, add_prob=0.0, seed=0)
    assert mask == []
    assert dropped == cat_indices

    mask, dropped = perturb_categorical_mask(cat_indices, n_cols, drop_prob=0.0, add_prob=1.0, seed=0)
    assert mask == list(range(n_cols))
    assert dropped == []


def test_perturb_categorical_mask_reproducible_and_valid():
    cat_indices = list(range(0, 40, 2))
    n_cols = 40

    mask_a, dropped_a = perturb_categorical_mask(cat_indices, n_cols, drop_prob=0.5, add_prob=0.3, seed=7)
    mask_b, dropped_b = perturb_categorical_mask(cat_indices, n_cols, drop_prob=0.5, add_prob=0.3, seed=7)
    assert mask_a == mask_b
    assert dropped_a == dropped_b

    mask_c, _ = perturb_categorical_mask(cat_indices, n_cols, drop_prob=0.5, add_prob=0.3, seed=8)
    assert mask_c != mask_a  # different seed -> different perturbation (w.h.p.)

    cat_set = set(cat_indices)
    assert set(dropped_a) <= cat_set
    assert not set(dropped_a) & set(mask_a)
    assert all(0 <= i < n_cols for i in mask_a)
    # true categoricals that were not dropped stay in the mask
    assert cat_set - set(dropped_a) <= set(mask_a)


def _make_mixed_df(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "num_a": rng.normal(size=n),
            "cat_a": pd.Categorical(rng.choice(["x", "y", "z"], size=n)),
            "num_b": rng.normal(size=n),
            "cat_b": pd.Categorical(rng.choice(["u", "v"], size=n)),
        },
    )


def _make_model(tmp_path, hyperparameters: dict) -> TabPFN3Model:
    model = TabPFN3Model(
        path=str(tmp_path),
        problem_type="binary",
        hyperparameters=hyperparameters,
    )
    # AutoGluon initializes before _fit/_preprocess(is_train=True) in a real fit;
    # mirror that here so the hyperparameters are visible to _get_model_params().
    model.initialize()
    return model


def test_preprocess_without_perturbation_detects_cats(tmp_path):
    model = _make_model(tmp_path, {})
    X = _make_mixed_df()
    X_out = model._preprocess(X, is_train=True)

    assert model._categorical_indices == [1, 3]
    assert not model._cat_mask_dropped_cols
    assert X_out["cat_a"].dtype == "category"  # untouched without perturbation


def test_preprocess_drop_encodes_columns_consistently(tmp_path):
    model = _make_model(tmp_path, {"cat_mask_drop_prob": 1.0, "cat_mask_seed": 0})
    X = _make_mixed_df()
    # inject a missing value to check NaN round-trips through the code encoding
    X.loc[X.index[0], "cat_a"] = np.nan

    X_train = model._preprocess(X, is_train=True)

    assert model._categorical_indices is None  # every categorical dropped from the mask
    assert model._cat_mask_dropped_cols == ["cat_a", "cat_b"]
    assert pd.api.types.is_float_dtype(X_train["cat_a"])
    assert pd.api.types.is_float_dtype(X_train["cat_b"])
    assert np.isnan(X_train["cat_a"].iloc[0])

    # predict-time preprocessing applies the same stateful encoding
    X_test = model._preprocess(_make_mixed_df(), is_train=False)
    assert pd.api.types.is_float_dtype(X_test["cat_a"])
    pd.testing.assert_series_equal(X_train["cat_b"], X_test["cat_b"], check_names=True)

    # numeric columns are untouched
    pd.testing.assert_series_equal(X_train["num_a"], X["num_a"], check_dtype=False)


def test_preprocess_add_marks_numeric_as_categorical(tmp_path):
    model = _make_model(tmp_path, {"cat_mask_add_prob": 1.0, "cat_mask_seed": 0})
    X = _make_mixed_df()
    X_out = model._preprocess(X, is_train=True)

    assert model._categorical_indices == [0, 1, 2, 3]  # numerics added to the mask
    assert not model._cat_mask_dropped_cols
    assert X_out["cat_a"].dtype == "category"  # values themselves untouched


@pytest.mark.parametrize("param", ["cat_mask_drop_prob", "cat_mask_add_prob", "cat_mask_seed"])
def test_cat_mask_params_are_popped_from_estimator_hps(tmp_path, param):
    """The wrapper-only params must never reach the tabpfn estimator constructor."""
    from tabarena.models.tabpfn_3.model import CAT_MASK_PARAM_NAMES

    assert param in CAT_MASK_PARAM_NAMES
    model = _make_model(tmp_path, {param: 0.5})
    hps = dict(model._get_model_params())
    for name in CAT_MASK_PARAM_NAMES:
        hps.pop(name, None)
    assert param not in hps
