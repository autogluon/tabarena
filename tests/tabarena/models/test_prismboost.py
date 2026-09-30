"""Unit tests for the PrismBoost wrapper's own logic.

The registry-driven fit test (``test_all_models.py``) covers that the model fits and predicts.
What it does not reach is the wrapper's round-selection ladder and its dense encoder, so those
are tested here. Every test needs the optional ``prismboost`` dependency and is skipped without
it; the encoder tests need only scikit-learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabarena.models.prismboost._internal.preprocessing import PrismBoostPreprocessor
from tabarena.models.prismboost.model import PrismBoostModel

prismboost = pytest.importorskip("prismboost")


def _frame(n_rows: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "num": rng.normal(size=n_rows),
            "num_with_gaps": np.where(rng.random(n_rows) < 0.2, np.nan, rng.normal(size=n_rows)),
            "low_card": pd.Categorical(rng.choice(list("abc"), size=n_rows)),
            "high_card": pd.Categorical(rng.integers(0, 80, size=n_rows).astype(str)),
        },
    )


def _binary_target(X: pd.DataFrame) -> pd.Series:
    return pd.Series((X["num"] > 0).astype(int), index=X.index)


# --- encoder ---------------------------------------------------------------------------


def test_preprocessor_splits_categoricals_by_cardinality():
    X = _frame()
    pre = PrismBoostPreprocessor(problem_type="binary", ohe_max_cardinality=32)
    pre.fit_transform(X, _binary_target(X))
    assert pre.onehot_columns_ == ["low_card"]
    assert pre.encoded_columns_ == ["high_card"]
    assert pre.numeric_columns_ == ["num", "num_with_gaps"]


def test_preprocessor_onehot_encoder_leaves_nothing_to_target_encode():
    X = _frame()
    pre = PrismBoostPreprocessor(problem_type="binary", categorical_encoder="onehot")
    pre.fit_transform(X, _binary_target(X))
    assert pre.encoded_columns_ == []
    assert set(pre.onehot_columns_) == {"low_card", "high_card"}


def test_preprocessor_output_is_finite_float64_and_width_stable():
    X = _frame()
    pre = PrismBoostPreprocessor(problem_type="binary")
    train = pre.fit_transform(X, _binary_target(X))
    # Unseen levels and an all-missing numeric column are what a held-out split really brings.
    unseen = X.iloc[:20].copy()
    unseen["high_card"] = pd.Categorical(["unseen"] * 20)
    unseen["num_with_gaps"] = np.nan
    test = pre.transform(unseen)
    assert train.dtype == np.float64
    assert np.isfinite(train).all()
    assert np.isfinite(test).all()
    assert train.shape[1] == test.shape[1]


def test_preprocessor_missing_indicator_is_binary():
    """The indicator block is 0/1: it is kept out of the pipeline the scaler fits."""
    X = _frame()
    pre = PrismBoostPreprocessor(problem_type="binary", numeric_scaler="standard")
    out = pre.fit_transform(X, _binary_target(X))
    indicator = out[:, 2]  # one numeric column has gaps, so exactly one indicator follows
    assert set(np.unique(indicator)) <= {0.0, 1.0}
    assert 0 < indicator.sum() < len(X)


def test_preprocessor_handles_missing_values_unseen_in_training():
    """A bagged fold can hold the only missing value in a column; transform must not raise."""
    X = _frame()
    X["num"] = X["num"].fillna(0.0)  # complete during fit
    pre = PrismBoostPreprocessor(problem_type="binary")
    train = pre.fit_transform(X, _binary_target(X))
    held_out = X.iloc[:20].copy()
    held_out.loc[held_out.index[:5], "num"] = np.nan  # missing only at transform time
    test = pre.transform(held_out)
    assert train.shape[1] == test.shape[1]
    assert np.isfinite(test).all()


def test_preprocessor_handles_an_integer_regression_target():
    """TabArena regression targets are not always float (Food_Delivery_Time is uint8)."""
    X = _frame()
    y = pd.Series(np.arange(len(X), dtype=np.uint8), index=X.index)
    pre = PrismBoostPreprocessor(problem_type="regression")
    out = pre.fit_transform(X, y)
    assert np.isfinite(out).all()


def test_preprocessor_rejects_an_unknown_scaler():
    X = _frame()
    pre = PrismBoostPreprocessor(problem_type="binary", numeric_scaler="not-a-scaler")
    with pytest.raises(ValueError, match="numeric_scaler"):
        pre.fit_transform(X, _binary_target(X))


# --- round-selection ladder ------------------------------------------------------------


def _fitted(hyperparameters: dict, **fit_kwargs) -> PrismBoostModel:
    X = _frame(n_rows=300)
    y = _binary_target(X)
    model = PrismBoostModel(
        problem_type="binary",
        eval_metric="roc_auc",
        hyperparameters=hyperparameters,
        name="prismboost-test",
    )
    model.fit(X=X.iloc[:240], y=y.iloc[:240], X_val=X.iloc[240:], y_val=y.iloc[240:], **fit_kwargs)
    return model


def test_ladder_records_the_chosen_round_count():
    model = _fitted({"max_n_estimators": 100})
    assert model.params_trained["n_estimators"] in (50, 100)


def test_explicit_round_count_skips_the_ladder():
    model = _fitted({"n_estimators": 40, "max_n_estimators": 100})
    assert "n_estimators" not in model.params_trained
    assert model.model.n_estimators_ == 40


def test_refitting_the_chosen_count_reproduces_the_selected_model():
    """The refit path replays `params_trained` as an explicit value, so it must be the same fit."""
    X = _frame(n_rows=300)
    y = _binary_target(X)
    selected = _fitted({"max_n_estimators": 100})
    chosen = selected.params_trained["n_estimators"]

    refit = PrismBoostModel(
        problem_type="binary",
        eval_metric="roc_auc",
        hyperparameters={"n_estimators": chosen, "max_n_estimators": 100},
        name="prismboost-refit",
    )
    refit.fit(X=X.iloc[:240], y=y.iloc[:240])
    np.testing.assert_allclose(selected.predict_proba(X), refit.predict_proba(X))


def test_a_tight_time_limit_stops_the_ladder_early():
    model = _fitted({"max_n_estimators": 1600}, time_limit=0.5)
    assert model.model is not None
    assert model.params_trained["n_estimators"] < 1600


# --- class alignment -------------------------------------------------------------------


def test_align_proba_widens_a_child_that_missed_a_class():
    model = PrismBoostModel(problem_type="multiclass", eval_metric="log_loss", name="prismboost-align")
    model.num_classes = 3

    class _Child:
        classes_ = np.array([0, 2])

    widened = model._align_proba(_Child(), np.array([[0.3, 0.7], [0.6, 0.4]]))
    np.testing.assert_allclose(widened, [[0.3, 0.0, 0.7], [0.6, 0.0, 0.4]])
