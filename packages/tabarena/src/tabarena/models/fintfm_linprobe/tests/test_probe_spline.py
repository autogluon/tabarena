# ruff: noqa: F811
"""``probe_spline``: spline features of the numeric raw columns in the readout (entry default: on, FINDINGS §290/§293)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

pytest.importorskip("fintfm")
pytest.importorskip("torch")

from sklearn.preprocessing import SplineTransformer

from tabarena.models.fintfm_linprobe.model import (
    SPLINE_DEGREE,
    SPLINE_MAX_COLUMNS,
    SPLINE_N_KNOTS,
    FinTFMLinearProbeModel,
)
from tabarena.models.fintfm_linprobe.tests.test_crossfit import (  # noqa: F401 (fixtures)
    N_TRAIN,
    PARTITION,
    checkpoint,
    clean_env,
    data,
    make,
)

PROBE = {"scale": True, "cv": True, "raw": True}


@pytest.fixture(autouse=True)
def clean_spline_env(monkeypatch):
    monkeypatch.delenv("FINTFM_PROBE_SPLINE", raising=False)


def fitted(tmp_path, checkpoint, problem_type="binary", *, categorical=False, partitioned=False, **hp):
    """A fitted model plus its training frame and the test frame."""
    X, y, X_test = data(problem_type, categorical=categorical)
    extra = {f"partition_{problem_type}": dict(PARTITION)} if partitioned else {}
    m = make(problem_type, tmp_path, checkpoint, **{f"probe_{problem_type}": PROBE, **extra, **hp})
    m.fit(X=X, y=y)
    return m, X, X_test


def predict(m, X_test) -> np.ndarray:
    return np.asarray(m.predict_proba(X_test))


def knots(m) -> list[np.ndarray]:
    """The fitted knot vector of every spline column of the current readout."""
    return [np.array(b.t) for b in m._raw_spline["tf"].bsplines_]


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_explicit_none_turns_the_entry_splines_off(problem_type, tmp_path, checkpoint):
    entry, _, X_test = fitted(tmp_path / "a", checkpoint, problem_type)
    off, _, _ = fitted(tmp_path / "b", checkpoint, problem_type, probe_spline=None)
    assert entry._spline_cfg is not None and entry._raw_spline is not None
    assert off._get_model_params()["probe_spline"] is None
    assert off._spline_cfg is None and off._raw_spline is None
    assert not np.array_equal(predict(entry, X_test), predict(off, X_test))


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("categorical", [False, True])
def test_on_fits_and_predicts(problem_type, categorical, tmp_path, checkpoint):
    m, _, X_test = fitted(tmp_path, checkpoint, problem_type, categorical=categorical, probe_spline=True)
    assert m._spline_cfg == {"n_knots": SPLINE_N_KNOTS, "degree": SPLINE_DEGREE, "max_columns": SPLINE_MAX_COLUMNS}
    assert m._raw_spline is not None
    out = predict(m, X_test)
    assert np.isfinite(out).all()
    if problem_type == "regression":
        assert out.shape == (len(X_test),)
        return
    n_classes = 2 if problem_type == "binary" else 3
    assert out.shape == (len(X_test), n_classes) if problem_type == "multiclass" else out.shape[0] == len(X_test)
    probs = out if out.ndim == 2 else np.column_stack([1 - out, out])
    assert ((probs >= 0) & (probs <= 1)).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_on_changes_the_readout_input_width(tmp_path, checkpoint):
    base, _, _ = fitted(tmp_path / "a", checkpoint, probe_spline=None)
    on, _, _ = fitted(tmp_path / "b", checkpoint, probe_spline={"n_knots": 4, "degree": 2})
    per_column = 4 + 2 - 1 - 1  # n_knots + degree - 1 basis functions, one dropped by include_bias=False
    assert base._linear_probe.n_features_in_ + 5 * (per_column - 1) == on._linear_probe.n_features_in_


def test_categorical_columns_are_never_expanded(tmp_path, checkpoint):
    m, X, _ = fitted(tmp_path, checkpoint, "multiclass", categorical=True, probe_spline=True)
    cat = set(m._cat_cols.tolist())
    assert cat == {X.columns.get_loc("cat")}
    assert cat.isdisjoint(m._raw_spline["sel"].tolist())
    assert cat <= set(m._raw_spline["keep"].tolist())


def test_low_cardinality_columns_stay_as_they_are(tmp_path, checkpoint):
    X, y, X_test = data("binary")
    X = X.copy()
    X["flag"] = (X["f2"] > 0).astype(float)  # two values: fewer than n_knots distinct knots
    X["const"] = 1.0
    X["mostly_zero"] = np.where(np.arange(len(X)) % 10 == 0, X["f3"], 0.0)  # tied quantile knots
    X_test = X_test.assign(flag=0.0, const=1.0, mostly_zero=0.0)
    m = make("binary", tmp_path, checkpoint, probe_binary=PROBE, probe_spline=True)
    m.fit(X=X, y=y)
    # f0..f4 are the first five columns; the two-valued flag and the mostly-zero column stay as they
    # are (AutoGluon already drops the constant column before the readout sees it).
    assert m._raw_spline["sel"].tolist() == [0, 1, 2, 3, 4]
    assert len(m._raw_spline["keep"]) >= 2
    assert np.isfinite(predict(m, X_test)).all()


def test_all_columns_low_cardinality_gives_no_spline_stage(tmp_path, checkpoint):
    X, y, X_test = data("binary")
    X = (X > 0).astype(float)
    X_test = (X_test > 0).astype(float)
    m = make("binary", tmp_path, checkpoint, probe_binary=PROBE, probe_spline=True)
    m.fit(X=X, y=y)
    assert m._raw_spline is None
    assert np.isfinite(predict(m, X_test)).all()


def test_knots_depend_only_on_training_rows(tmp_path, checkpoint, monkeypatch):
    fits: list[np.ndarray] = []
    original = SplineTransformer.fit

    def spy(self, X, *args, **kwargs):
        fits.append(np.array(X, copy=True))
        return original(self, X, *args, **kwargs)

    monkeypatch.setattr(SplineTransformer, "fit", spy)
    m, X, X_test = fitted(tmp_path, checkpoint, probe_spline=True)
    assert len(fits) == 1
    assert fits[0].shape[0] == len(X) == N_TRAIN, "the transformer must see the training rows, and only them"
    before = [k.copy() for k in knots(m)]
    state = pickle.dumps(m._raw_spline)
    base = predict(m, X_test)
    # Wildly different query rows, then the originals again.
    predict(m, X_test * 1000.0 + 50.0)
    predict(m, X_test.iloc[:3])
    assert len(fits) == 1, "predict refitted the transformer"
    assert all(np.array_equal(a, b) for a, b in zip(before, knots(m), strict=True))
    assert pickle.dumps(m._raw_spline) == state
    assert np.array_equal(base, predict(m, X_test))


def test_fitted_knots_are_quantiles_of_the_training_rows_alone(tmp_path, checkpoint):
    a, X, X_test = fitted(tmp_path / "a", checkpoint, probe_spline=True)
    # The same training rows fitted again: the knots do not depend on anything else around.
    b, _, _ = fitted(tmp_path / "b", checkpoint, probe_spline=True)
    assert all(np.array_equal(p, q) for p, q in zip(knots(a), knots(b), strict=True))


def test_state_pickles_and_stays_small(tmp_path, checkpoint):
    m, X, X_test = fitted(tmp_path, checkpoint, probe_spline=True)
    blob = pickle.dumps(m._raw_spline)
    assert len(blob) < 100_000
    assert len(blob) < X.to_numpy().nbytes
    restored = pickle.loads(blob)
    Z = np.random.default_rng(1).normal(size=(7, len(m._raw_spline["sel"])))
    assert np.allclose(restored["tf"].transform(Z), m._raw_spline["tf"].transform(Z))


@pytest.mark.parametrize("partitioned", [False, True])
def test_works_with_crossfit_and_partition(partitioned, tmp_path, checkpoint, monkeypatch):
    fits: list[int] = []
    original = SplineTransformer.fit

    def spy(self, X, *args, **kwargs):
        fits.append(len(X))
        return original(self, X, *args, **kwargs)

    monkeypatch.setattr(SplineTransformer, "fit", spy)
    m, _, X_test = fitted(tmp_path, checkpoint, partitioned=partitioned, probe_crossfit_folds=5, probe_spline=True)
    assert m._crossfit_on
    # One transformer per readout, fitted on the training rows, shared by folds and leaves as
    # the raw conditioner is; no fit ever sees a query row.
    assert fits == [N_TRAIN]
    assert np.isfinite(predict(m, X_test)).all()
    assert fits == [N_TRAIN]


def test_env_parsing(monkeypatch):
    over = FinTFMLinearProbeModel._env_entry_overrides
    assert "probe_spline" not in over()
    monkeypatch.setenv("FINTFM_PROBE_SPLINE", "")
    assert "probe_spline" not in over()
    for text in ("off", "OFF", "none", "false", "no"):
        monkeypatch.setenv("FINTFM_PROBE_SPLINE", text)
        assert over()["probe_spline"] is None  # turns the binary entry's splines off
    for text in ("on", "ON", "true", "yes"):
        monkeypatch.setenv("FINTFM_PROBE_SPLINE", text)
        assert over()["probe_spline"] is True
    monkeypatch.setenv("FINTFM_PROBE_SPLINE", '{"n_knots": 7, "degree": 2}')
    assert over()["probe_spline"] == {"n_knots": 7, "degree": 2}
    monkeypatch.setenv("FINTFM_PROBE_SPLINE", '{"n_knots": 7}')
    assert FinTFMLinearProbeModel._check_spline(over()["probe_spline"], source="t") == {
        "n_knots": 7,
        "degree": 3,
        "max_columns": SPLINE_MAX_COLUMNS,
    }
    monkeypatch.setenv("FINTFM_PROBE_SPLINE", '{"max_columns": 12}')
    assert over()["probe_spline"] == {"max_columns": 12}


@pytest.mark.parametrize(
    "text",
    [
        "maybe",
        "{not json",
        "[3, 3]",
        "7",
        '{"knots": 5}',
        '{"n_knots": 2}',
        '{"n_knots": 5.0}',
        '{"n_knots": true}',
        '{"degree": 0}',
        '{"degree": 4}',
        '{"degree": "3"}',
        '{"max_columns": 0}',
        '{"max_columns": 2.5}',
        '{"max_columns": true}',
    ],
)
def test_env_invalid_values_are_errors(text, monkeypatch):
    monkeypatch.setenv("FINTFM_PROBE_SPLINE", text)
    with pytest.raises(ValueError, match="FINTFM_PROBE_SPLINE"):
        FinTFMLinearProbeModel._env_entry_overrides()


def test_hyperparameter_validation(tmp_path, checkpoint):
    X, y, _ = data("binary")
    m = make("binary", tmp_path, checkpoint, probe_binary=PROBE, probe_spline={"n_knots": 2})
    with pytest.raises(ValueError, match="probe_spline"):
        m.fit(X=X, y=y)


def test_env_variable_reaches_the_fit(tmp_path, checkpoint, monkeypatch):
    monkeypatch.setenv("FINTFM_PROBE_SPLINE", '{"n_knots": 4, "degree": 1}')
    m, _, _ = fitted(tmp_path, checkpoint)
    assert m._spline_cfg == {"n_knots": 4, "degree": 1, "max_columns": SPLINE_MAX_COLUMNS}


def test_moot_without_the_raw_block(tmp_path, checkpoint):
    base, _, X_test = fitted(tmp_path / "a", checkpoint, probe_binary={"scale": True})
    on, _, _ = fitted(tmp_path / "b", checkpoint, probe_binary={"scale": True}, probe_spline=True)
    assert on._spline_cfg is None
    assert on._raw_spline is None
    assert np.array_equal(predict(base, X_test), predict(on, X_test))


def test_max_columns_binds_by_distinct_count_then_column_index(tmp_path, checkpoint):
    X, y, X_test = data("binary")
    X, X_test = X.copy(), X_test.copy()
    # Distinct training values: f0..f4 have 240 each (ties); the two coarse columns have 8 and 6.
    for frame in (X, X_test):
        frame["coarse8"] = np.round(frame["f0"] * 2).clip(-4, 3)
        frame["coarse6"] = np.round(frame["f1"] * 2).clip(-3, 2)
    assert X["coarse8"].nunique() == 8
    assert X["coarse6"].nunique() == 6
    m = make("binary", tmp_path, checkpoint, probe_binary=PROBE, probe_spline={"max_columns": 3})
    m.fit(X=X, y=y)
    # All seven qualify; the five 240-value columns win on count, the three lowest indices on the tie.
    assert m._raw_spline["sel"].tolist() == [0, 1, 2]
    assert m._raw_spline["keep"].tolist() == [3, 4, 5, 6]
    assert np.isfinite(predict(m, X_test)).all()
    # A cap of 6 takes all five fine columns plus the 8-value one, not the 6-value one.
    m6 = make("binary", tmp_path / "b", checkpoint, probe_binary=PROBE, probe_spline={"max_columns": 6})
    m6.fit(X=X, y=y)
    assert m6._raw_spline["sel"].tolist() == [0, 1, 2, 3, 4, 5]
    assert m6._raw_spline["keep"].tolist() == [6]


def test_cap_not_binding_changes_nothing(tmp_path, checkpoint):
    a, _, X_test = fitted(tmp_path / "a", checkpoint, probe_spline=True)
    b, _, _ = fitted(tmp_path / "b", checkpoint, probe_spline={"max_columns": 5})
    assert a._raw_spline["sel"].tolist() == b._raw_spline["sel"].tolist() == [0, 1, 2, 3, 4]
    assert np.array_equal(predict(a, X_test), predict(b, X_test))
