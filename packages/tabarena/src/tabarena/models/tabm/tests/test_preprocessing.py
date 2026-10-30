"""TabM's numeric preprocessing: near-constant columns are dropped and new data is clipped to the training range."""

from __future__ import annotations

import numpy as np
import pandas as pd
from autogluon.core.metrics import root_mean_squared_error

from tabarena.models.tabm._internal._tabm_internal import TabMImplementation


def _fit(X: pd.DataFrame, y: pd.Series) -> TabMImplementation:
    model = TabMImplementation(
        early_stopping_metric=root_mean_squared_error,
        problem_type="regression",
        device="cpu",
        verbosity=0,
        random_state=0,
        n_epochs=2,
        d_block=32,
        n_blocks=1,
        tabm_k=2,
    )
    model.fit(X_train=X.iloc[:80], y_train=y.iloc[:80], X_val=X.iloc[80:], y_val=y.iloc[80:], cat_col_names=[])
    return model


def test_values_outside_the_training_range_predict_like_the_nearest_training_extreme():
    rng = np.random.default_rng(0)
    # "age" is heavily tied at its minimum, like concrete_compressive_strength's Age (3 days).
    X = pd.DataFrame({"age": rng.choice([3.0, 3.0, 3.0, 7.0, 28.0, 90.0], 100), "x": rng.normal(size=100)})
    y = pd.Series(X["age"] / 10 + X["x"])
    model = _fit(X, y)

    below = X.iloc[:5].assign(age=1.0)  # below the training minimum
    at_min = X.iloc[:5].assign(age=3.0)
    np.testing.assert_allclose(model.predict(below), model.predict(at_min))
