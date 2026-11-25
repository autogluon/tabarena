"""Inductive predictions and save/load for the TabSil wrapper, on the CPU with the pinned weights.

A test row's prediction may depend only on the training rows and that row: predicted alone, inside
the full test set, in batches of other sizes, in another order or after an identical copy of itself, it
gets the same prediction up to floating-point rounding. The chunked forward the network takes on large
tables is checked the same way by lowering its chunk size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tabsil")

from tabarena.models.tabsil.model import TabSilModel

#: Four members: the regression members turn the target power transform off, off, on, on, so both
#: kinds are covered, and the CPU fits stay at a few seconds.
_HYPERPARAMETERS = {"n_estimators": 4}
_ATOL = 1e-5


def _table(n_rows: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n_rows, 5)), columns=[f"num{i}" for i in range(5)])
    X.loc[rng.random(n_rows) < 0.1, "num1"] = np.nan
    X["cat"] = pd.Categorical(rng.choice(["a", "b", "c", "d"], size=n_rows))
    return X


def _target(X: pd.DataFrame, problem_type: str) -> pd.Series:
    signal = X["num0"].to_numpy() + 0.5 * (X["cat"] == "b").to_numpy()
    if problem_type == "regression":
        return pd.Series(3.0 * signal + 10.0)
    if problem_type == "binary":
        return pd.Series((signal > 0).astype(int))
    return pd.Series(np.digitize(signal, [-0.5, 0.5]).astype(int))


def _fit(tmp_path, problem_type: str) -> tuple[TabSilModel, pd.DataFrame]:
    X = _table(260)
    y = _target(X, problem_type)
    model = TabSilModel(
        path=str(tmp_path),
        name="tabsil",
        problem_type=problem_type,
        eval_metric="root_mean_squared_error" if problem_type == "regression" else "log_loss",
        hyperparameters=_HYPERPARAMETERS,
    )
    model.fit(X=X.iloc[:200], y=y.iloc[:200], num_gpus=0, num_cpus=2)
    return model, X.iloc[200:].reset_index(drop=True)


def _predict(model: TabSilModel, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X), dtype=np.float64)


def _assert_inductive(model: TabSilModel, X_test: pd.DataFrame) -> None:
    reference = _predict(model, X_test)
    probes = range(8)
    alone = np.stack([_predict(model, X_test.iloc[[i]])[0] for i in probes])
    np.testing.assert_allclose(alone, reference[list(probes)], atol=_ATOL, err_msg="alone vs. full")
    for batch_size in (3, 7, 25):
        batched = np.concatenate(
            [_predict(model, X_test.iloc[i : i + batch_size]) for i in range(0, len(X_test), batch_size)]
        )
        np.testing.assert_allclose(batched, reference, atol=_ATOL, err_msg=f"batches of {batch_size}")
    reversed_rows = _predict(model, X_test.iloc[::-1])[::-1]
    np.testing.assert_allclose(reversed_rows, reference, atol=_ATOL, err_msg="reversed order")
    twin = _predict(model, pd.concat([X_test.iloc[[0]], X_test]).reset_index(drop=True))
    np.testing.assert_allclose(twin[0], twin[1], atol=_ATOL, err_msg="duplicate rows")
    np.testing.assert_allclose(twin[1:], reference, atol=_ATOL, err_msg="after a duplicate")


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_predictions_are_inductive(tmp_path, problem_type):
    model, X_test = _fit(tmp_path, problem_type)
    _assert_inductive(model, X_test)


@pytest.mark.parametrize("problem_type", ["binary", "regression"])
def test_chunked_forward_is_inductive(tmp_path, monkeypatch, problem_type):
    """Above ``inference_chunk_cells`` the network computes its stages in row chunks; force that here."""
    from tabsil.models.column_row_transformer import ColumnRowTransformer

    model, X_test = _fit(tmp_path, problem_type)
    unchunked = _predict(model, X_test)
    monkeypatch.setattr(ColumnRowTransformer, "inference_chunk_cells", 64 * 8)
    np.testing.assert_allclose(_predict(model, X_test), unchunked, atol=1e-4, err_msg="chunked vs. unchunked")
    _assert_inductive(model, X_test)


@pytest.mark.parametrize("problem_type", ["multiclass", "regression"])
def test_save_and_load_keep_the_predictions(tmp_path, problem_type):
    model, X_test = _fit(tmp_path, problem_type)
    expected = _predict(model, X_test)
    path = model.save()
    loaded = TabSilModel.load(path)
    np.testing.assert_allclose(_predict(loaded, X_test), expected, atol=1e-6)


@pytest.mark.parametrize("problem_type", ["binary", "regression"])
def test_float32_training_rows_keep_the_predictions(tmp_path, monkeypatch, problem_type):
    """``_compact_training_rows`` stores the members' training rows in float32 without changing a prediction."""
    model, X_test = _fit(tmp_path / "compact", problem_type)
    assert all(x.dtype == np.float32 for x in model.model.X_trains_)
    monkeypatch.setattr(TabSilModel, "_compact_training_rows", lambda self: None)
    full, _ = _fit(tmp_path / "full", problem_type)
    assert all(x.dtype == np.float64 for x in full.model.X_trains_)
    np.testing.assert_array_equal(_predict(model, X_test), _predict(full, X_test))


def test_a_class_missing_from_the_fold_gets_its_column(tmp_path):
    """AutoGluon's labels are 0..K-1 over the whole task; a fold without label 1 still returns K columns."""
    X = _table(200)
    y = pd.Series(np.where(X["num0"] > 0, 2, 0))
    model = TabSilModel(
        path=str(tmp_path),
        name="tabsil",
        problem_type="multiclass",
        eval_metric="log_loss",
        hyperparameters=_HYPERPARAMETERS,
    )
    model.fit(X=X.iloc[:150], y=y.iloc[:150], num_gpus=0, num_classes=3)
    proba = _predict(model, X.iloc[150:])
    assert proba.shape == (50, 3)
    # AutoGluon's probability post-processing leaves a floor of about 1e-7 on every column.
    np.testing.assert_allclose(proba[:, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_an_infinite_value_counts_as_missing(tmp_path):
    """One infinite training value would otherwise turn the column's outlier bounds into NaN."""
    X = _table(260)
    y = _target(X, "binary")
    with_inf, with_nan = X.copy(), X.copy()
    with_inf.loc[[3, 210], "num0"] = [np.inf, -np.inf]
    with_nan.loc[[3, 210], "num0"] = np.nan
    predictions = []
    for i, frame in enumerate((with_inf, with_nan)):
        model = TabSilModel(
            path=str(tmp_path / str(i)),
            name="tabsil",
            problem_type="binary",
            eval_metric="log_loss",
            hyperparameters=_HYPERPARAMETERS,
        )
        model.fit(X=frame.iloc[:200], y=y.iloc[:200], num_gpus=0)
        assert all(np.isfinite(prep.lower[0]) for prep in model.model.preprocessors_)
        predictions.append(_predict(model, frame.iloc[200:]))
    np.testing.assert_allclose(predictions[0], predictions[1], atol=1e-6)
