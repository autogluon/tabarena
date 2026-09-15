from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabarena.utils.synthetic_data import make_synthetic_frames


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_shapes_and_dtypes(problem_type):
    X, y, X_predict = make_synthetic_frames(problem_type, n_rows=96, n_features=6, n_categorical=1, seed=0)
    assert isinstance(X, pd.DataFrame) and isinstance(X_predict, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (96, 6)
    assert X_predict.shape[1] == 6
    assert len(X_predict) == 24
    assert len(y) == 96
    assert isinstance(X.dtypes.iloc[-1], pd.CategoricalDtype)
    assert isinstance(X_predict.dtypes.iloc[-1], pd.CategoricalDtype)
    assert all(pd.api.types.is_float_dtype(dt) for dt in X.dtypes.iloc[:-1])


def test_labels_per_problem_type():
    _, y_bin, _ = make_synthetic_frames("binary")
    _, y_multi, _ = make_synthetic_frames("multiclass")
    _, y_reg, _ = make_synthetic_frames("regression")
    assert set(y_bin.unique()) == {0, 1}
    assert set(y_multi.unique()) == {0, 1, 2}
    assert pd.api.types.is_float_dtype(y_reg)
    assert y_reg.nunique() > 10
    # Quantile thresholds keep the classes balanced.
    assert y_bin.value_counts().min() >= 40
    assert y_multi.value_counts().min() >= 25


def test_fixed_seed_is_reproducible_and_seed_changes_data():
    X1, y1, P1 = make_synthetic_frames("binary", seed=0)
    X2, y2, P2 = make_synthetic_frames("binary", seed=0)
    X3, _, _ = make_synthetic_frames("binary", seed=1)
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)
    pd.testing.assert_frame_equal(P1, P2)
    assert not np.allclose(X1["f0"].to_numpy(), X3["f0"].to_numpy())


def test_overrides_and_validation():
    X, y, X_predict = make_synthetic_frames("regression", n_rows=20, n_features=3, n_categorical=0, n_rows_predict=5)
    assert X.shape == (20, 3) and len(X_predict) == 5
    assert not any(isinstance(dt, pd.CategoricalDtype) for dt in X.dtypes)
    with pytest.raises(ValueError, match="problem_type"):
        make_synthetic_frames("quantile")
    with pytest.raises(ValueError, match="n_categorical"):
        make_synthetic_frames("binary", n_features=2, n_categorical=3)
