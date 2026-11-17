# ruff: noqa: F811
"""``probe_onehot``: one-hot level indicators of categorical columns in the multiclass readout."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fintfm")
pytest.importorskip("torch")

from tabarena.models.fintfm_linprobe.model import (
    ONEHOT_MAX_COLUMNS,
    ONEHOT_MIN_COUNT,
    FinTFMLinearProbeModel,
    _AugmentedMatrix,
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
def clean_onehot_env(monkeypatch):
    monkeypatch.delenv("FINTFM_PROBE_ONEHOT", raising=False)


def frame(seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Multiclass data with a 3-level, a 6-level (two rare levels) and a 2-level categorical column."""
    X, y, X_test = data("multiclass", seed=seed, categorical=True)
    rng = np.random.default_rng(seed + 1)
    full = pd.concat([X, X_test], ignore_index=True)
    levels = np.array(["p", "q", "r", "s", "t", "u"])
    codes = rng.choice(6, size=len(full), p=[0.35, 0.3, 0.2, 0.1, 0.03, 0.02])
    full["many"] = pd.Categorical(levels[codes])
    full["flag"] = pd.Categorical(rng.choice(["yes", "no"], size=len(full), p=[0.8, 0.2]))
    return full.iloc[:N_TRAIN].reset_index(drop=True), y, full.iloc[N_TRAIN:]


def fitted(tmp_path, checkpoint, *, partitioned=False, problem_type="multiclass", **hp):
    X, y, X_test = frame() if problem_type == "multiclass" else (*data(problem_type, categorical=True)[:2], None)
    if problem_type != "multiclass":
        X_test = data(problem_type, categorical=True)[2]
    extra = {f"partition_{problem_type}": dict(PARTITION)} if partitioned else {}
    m = make(problem_type, tmp_path, checkpoint, **{f"probe_{problem_type}": PROBE, **extra, **hp})
    m.fit(X=X, y=y)
    return m, X, X_test


def predict(m, X_test) -> np.ndarray:
    return np.asarray(m.predict_proba(X_test))


ENTRY_ONEHOT = {"min_count": 3, "max_columns": 256, "max_levels": 64}


@pytest.mark.parametrize("problem_type", ["binary", "regression"])
@pytest.mark.parametrize("partitioned", [False, True])
def test_default_is_off_and_bit_identical_for_binary_and_regression(problem_type, partitioned, tmp_path, checkpoint):
    base, _, X_test = fitted(tmp_path / "a", checkpoint, partitioned=partitioned, problem_type=problem_type)
    explicit, _, _ = fitted(
        tmp_path / "b", checkpoint, partitioned=partitioned, problem_type=problem_type, probe_onehot=None
    )
    assert base._get_model_params()["probe_onehot"] is None
    assert base._onehot_cfg is None
    assert base._oh_spec is None
    assert np.array_equal(predict(base, X_test), predict(explicit, X_test))


@pytest.mark.parametrize("partitioned", [False, True])
def test_multiclass_default_is_the_entry_and_off_is_explicit(partitioned, tmp_path, checkpoint):
    base, _, X_test = fitted(tmp_path / "a", checkpoint, partitioned=partitioned)
    entry, _, _ = fitted(tmp_path / "b", checkpoint, partitioned=partitioned, probe_onehot=dict(ENTRY_ONEHOT))
    off, _, _ = fitted(tmp_path / "c", checkpoint, partitioned=partitioned, probe_onehot=None)
    assert base._get_model_params()["probe_onehot"] == ENTRY_ONEHOT
    assert base._onehot_cfg == ENTRY_ONEHOT
    assert base._oh_spec is not None
    assert off._onehot_cfg is None
    assert off._oh_spec is None
    assert np.array_equal(predict(base, X_test), predict(entry, X_test))
    assert not np.array_equal(predict(base, X_test), predict(off, X_test))


@pytest.mark.parametrize("problem_type", ["binary", "regression"])
def test_other_problem_types_ignore_the_option(problem_type, tmp_path, checkpoint):
    base, _, X_test = fitted(tmp_path / "a", checkpoint, problem_type=problem_type)
    on, _, _ = fitted(tmp_path / "b", checkpoint, problem_type=problem_type, probe_onehot=True)
    assert on._onehot_cfg is None
    assert np.array_equal(predict(base, X_test), predict(on, X_test))


def test_ignored_without_the_raw_block(tmp_path, checkpoint):
    base, _, X_test = fitted(tmp_path / "a", checkpoint, probe_multiclass={"scale": True})
    on, _, _ = fitted(tmp_path / "b", checkpoint, probe_multiclass={"scale": True}, probe_onehot=True)
    assert on._onehot_cfg is None
    assert np.array_equal(predict(base, X_test), predict(on, X_test))


@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize("folds", [0, 5])
def test_readout_gets_aligned_indicators_and_the_classifier_matrix_is_unchanged(
    partitioned, folds, tmp_path, checkpoint, monkeypatch
):
    seen = []
    train_mats = []
    original = FinTFMLinearProbeModel._probe_input
    original_pre = FinTFMLinearProbeModel.preprocess

    def keep_train(self, X, **kwargs):
        out = original_pre(self, X, **kwargs)
        if kwargs.get("is_train"):
            train_mats.append(out)
        return out

    monkeypatch.setattr(FinTFMLinearProbeModel, "preprocess", keep_train)

    def record(self, est, Xq, fit_idx=None):
        out = original(self, est, Xq, fit_idx)
        seen.append((np.asarray(Xq).copy(), None if getattr(Xq, "oh", None) is None else Xq.oh.copy(), out))
        return out

    monkeypatch.setattr(FinTFMLinearProbeModel, "_probe_input", record)
    base, X, X_test = fitted(
        tmp_path / "a", checkpoint, partitioned=partitioned, probe_crossfit_folds=folds, probe_onehot=None
    )
    seen_base = list(seen)
    seen.clear()
    on, _, _ = fitted(
        tmp_path / "b", checkpoint, partitioned=partitioned, probe_crossfit_folds=folds, probe_onehot=True
    )
    cols, levels = on._oh_spec
    # 3 levels of "cat", 5 of "many" (min_count 3 drops the level with 2 rows, 4 rows kept above), 1 of "flag".
    assert len(cols) == len(levels)
    assert ONEHOT_MIN_COUNT == 3
    assert set(cols) == {0, 1, 2}
    assert sum(cols == 2) == 1  # the two-level column keeps only its rarer level
    width_on = max(out.shape[1] for _, _, out in seen)
    width_base = max(out.shape[1] for _, _, out in seen_base)
    assert width_on - width_base == len(cols)
    # Every row the readout saw carries indicators equal to the codes of that very row.
    # Cross-fitted rows are the out-of-fold encoded training matrix, the others the replay.
    replay = train_mats[-1] if folds >= 2 else on.preprocess(X, is_train=False)
    train_mats.clear()
    for Xq, oh, _ in seen:
        assert oh is not None
        assert oh.shape == (len(Xq), len(cols))
        for r in range(0, len(Xq), 7):
            hit = np.flatnonzero((np.asarray(replay) == Xq[r]).all(axis=1))
            assert hit.size >= 1
            assert any(np.array_equal(replay.oh[h], oh[r]) for h in hit)
    # The matrix the classifier works on is the same as without the option.
    base_replay = np.asarray(base.preprocess(X, is_train=False))
    assert np.array_equal(np.asarray(on.preprocess(X, is_train=False)), base_replay)
    assert predict(on, X_test).shape == predict(base, X_test).shape
    assert np.allclose(predict(on, X_test).sum(axis=1), 1.0)
    assert not np.array_equal(predict(on, X_test), predict(base, X_test))


def test_indicator_values_match_the_raw_categories(tmp_path, checkpoint):
    on, X, _ = fitted(tmp_path / "a", checkpoint, probe_onehot=True, probe_crossfit_folds=0)
    cols, levels = on._oh_spec
    mat = on.preprocess(X, is_train=False)
    assert isinstance(mat, _AugmentedMatrix)
    assert mat.shape[1] == X.shape[1]  # the classifier's width: base columns only
    # "flag" (the third categorical in frame order) keeps its rarer level, "no".
    flag_col = [i for i, c in enumerate(cols) if c == 2]
    assert len(flag_col) == 1
    assert np.array_equal(mat.oh[:, flag_col[0]], (X["flag"] == "no").to_numpy(dtype=np.float32))


def test_unseen_level_at_predict_is_all_zero_and_missing_is_handled(tmp_path, checkpoint):
    on, X, X_test = fitted(tmp_path / "a", checkpoint, probe_onehot=True)
    odd = X_test.copy()
    # AutoGluon's pipeline maps a level unseen in training to NaN on the training categories.
    odd["many"] = pd.Categorical(["zzz"] * len(odd), categories=X["many"].cat.categories)
    odd.loc[odd.index[0], "cat"] = np.nan
    mat = on.preprocess(odd)
    cols, _ = on._oh_spec
    many_cols = [i for i, c in enumerate(cols) if c == 1]
    assert not mat.oh[:, many_cols].any()
    proba = predict(on, odd)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_caps(tmp_path, checkpoint):
    capped, _, _ = fitted(tmp_path / "a", checkpoint, probe_onehot={"max_columns": 2})
    assert len(capped._oh_spec[0]) == 2
    # The two most frequent qualifying levels win, whichever column they are in.
    none, X, X_test = fitted(tmp_path / "b", checkpoint, probe_onehot={"min_count": 10_000})
    assert none._oh_spec is None
    base, _, _ = fitted(tmp_path / "c", checkpoint, probe_onehot=None)
    assert np.array_equal(predict(none, X_test), predict(base, X_test))


def test_choose_indicators_ranks_by_frequency_and_orders_by_column():
    codes = np.array(
        [[0, 0, 5]] * 6 + [[1, 1, 5]] * 3 + [[2, 2, 7]] * 1 + [[2, -1, 5]] * 2,
        dtype=np.float64,
    )
    cfg = {"min_count": 3, "max_columns": 3}
    cols, levels = FinTFMLinearProbeModel._choose_indicators(codes, cfg)
    # col 0: levels 0 (6), 1 (3), 2 (3); col 1: 0 (6), 1 (3), 2 (1), -1 (2); col 2: two levels, rarer is 7 (1 row) -> dropped.
    # Top 3 by count: (6, col0, 0), (6, col1, 0), then (3, col0, 1).
    assert list(zip(cols.tolist(), levels.tolist(), strict=True)) == [(0, 0.0), (0, 1.0), (1, 0.0)]
    assert FinTFMLinearProbeModel._choose_indicators(codes, {"min_count": 99, "max_columns": 3}) is None


def test_many_categoricals_are_capped():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 3, size=(500, 1617)).astype(np.float64)
    cols, levels = FinTFMLinearProbeModel._choose_indicators(
        codes, {"min_count": ONEHOT_MIN_COUNT, "max_columns": ONEHOT_MAX_COLUMNS}
    )
    assert len(cols) == len(levels) == ONEHOT_MAX_COLUMNS


def test_augmented_matrix_slicing_and_pickle():
    arr = np.arange(12, dtype=np.float32).reshape(4, 3)
    oh = np.arange(8, dtype=np.float32).reshape(4, 2)
    mat = _AugmentedMatrix.attach(arr, oh)
    assert np.array_equal(mat[np.array([2, 0])].oh, oh[[2, 0]])
    assert np.array_equal(mat[np.array([True, False, True, False])].oh, oh[[0, 2]])
    assert np.array_equal(mat[1:3].oh, oh[1:3])
    assert not isinstance(mat[:, :2], _AugmentedMatrix)
    assert not isinstance(mat[1], _AugmentedMatrix)
    back = pickle.loads(pickle.dumps(mat))
    assert type(back) is np.ndarray
    assert np.array_equal(back, arr)


def test_env_override(monkeypatch):
    monkeypatch.setenv("FINTFM_PROBE_ONEHOT", "on")
    assert FinTFMLinearProbeModel._env_entry_overrides()["probe_onehot"] is True
    monkeypatch.setenv("FINTFM_PROBE_ONEHOT", '{"min_count": 5, "max_columns": 64}')
    assert FinTFMLinearProbeModel._env_entry_overrides()["probe_onehot"] == {"min_count": 5, "max_columns": 64}
    # An explicit off overrides the multiclass entry's indicators; empty leaves the entry alone.
    for off in ("off", "none"):
        monkeypatch.setenv("FINTFM_PROBE_ONEHOT", off)
        assert FinTFMLinearProbeModel._env_entry_overrides()["probe_onehot"] is None
    monkeypatch.setenv("FINTFM_PROBE_ONEHOT", "")
    assert "probe_onehot" not in FinTFMLinearProbeModel._env_entry_overrides()
    for bad in ("{", '{"min_count": 0}', '{"bogus": 1}', '"on"', "[1]"):
        monkeypatch.setenv("FINTFM_PROBE_ONEHOT", bad)
        with pytest.raises(ValueError, match="FINTFM_PROBE_ONEHOT"):
            FinTFMLinearProbeModel._env_entry_overrides()


def test_env_override_reaches_the_fit(monkeypatch, tmp_path, checkpoint):
    monkeypatch.setenv("FINTFM_PROBE_ONEHOT", '{"max_columns": 4}')
    X, y, _ = frame()
    m = make("multiclass", tmp_path, checkpoint, probe_multiclass=PROBE)
    assert m._get_model_params()["probe_onehot"] == {"max_columns": 4}
    m.fit(X=X, y=y)
    assert len(m._oh_spec[0]) == 4


@pytest.mark.parametrize("bad", [1, "on", [1], {"min_count": 1.5}, {"min_count": True}, {"x": 1}])
def test_check_rejects_bad_values(bad):
    with pytest.raises(ValueError, match="probe_onehot"):
        FinTFMLinearProbeModel._check_onehot(bad, source="probe_onehot")
