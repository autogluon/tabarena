# ruff: noqa: F811
"""Size gates of the entry levers: ``probe_onehot`` ``max_levels`` and ``reg_rff_max_rows``."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fintfm")
pytest.importorskip("torch")

from tabarena.models.fintfm_linprobe.model import FinTFMLinearProbeModel
from tabarena.models.fintfm_linprobe.tests.test_crossfit import (  # noqa: F401 (fixtures)
    N_TRAIN,
    checkpoint,
    clean_env,
    data,
    make,
)
from tabarena.models.fintfm_linprobe.tests.test_probe_onehot import (  # noqa: F401 (fixtures)
    PROBE,
    clean_onehot_env,
    fitted,
    frame,
    predict,
)


@pytest.fixture(autouse=True)
def clean_rff_env(monkeypatch):
    monkeypatch.delenv("FINTFM_REG_RFF_MAX_ROWS", raising=False)
    monkeypatch.delenv("FINTFM_REG_READOUT", raising=False)


# ---- gate 1: probe_onehot max_levels -------------------------------------------------------


def test_max_levels_none_reproduces_ungated(tmp_path, checkpoint):
    base, _, X_test = fitted(tmp_path / "a", checkpoint, probe_onehot=True)
    explicit, _, _ = fitted(tmp_path / "b", checkpoint, probe_onehot={"max_levels": None})
    assert base._onehot_cfg["max_levels"] is None
    assert np.array_equal(base._oh_spec[0], explicit._oh_spec[0])
    assert np.array_equal(base._oh_spec[1], explicit._oh_spec[1])
    assert np.array_equal(predict(base, X_test), predict(explicit, X_test))


def test_max_levels_drops_the_high_cardinality_column_only(tmp_path, checkpoint, caplog):
    # Categorical columns in frame order: cat (3 levels), many (6), flag (2).
    on, _, _ = fitted(tmp_path / "a", checkpoint, probe_onehot=True)
    assert {0, 1, 2} <= set(on._oh_spec[0].tolist())
    with caplog.at_level(logging.WARNING):
        gated, _, X_test = fitted(tmp_path / "b", checkpoint, probe_onehot={"max_levels": 4})
    cols = set(gated._oh_spec[0].tolist())
    assert 1 not in cols  # "many" has 6 levels > 4
    assert {0, 2} <= cols  # the 3- and 2-level columns keep theirs
    msgs = [r.getMessage() for r in caplog.records if "max_levels" in r.getMessage()]
    assert len(msgs) == 1
    assert "many=6" in msgs[0]
    assert np.isfinite(predict(gated, X_test)).all()


def test_max_levels_at_the_boundary_keeps_the_column():
    codes = np.array([[0, 0]] * 4 + [[1, 1]] * 4 + [[2, 2]] * 4 + [[0, 3]] * 4, dtype=np.float64)
    cfg = {"min_count": 3, "max_columns": 10, "max_levels": 3}
    cols, _ = FinTFMLinearProbeModel._choose_indicators(codes, cfg)
    assert set(cols.tolist()) == {0}  # column 1 has 4 levels > 3; column 0 has exactly 3
    assert FinTFMLinearProbeModel._choose_indicators(codes, {**cfg, "max_levels": 2}) is None


def test_max_levels_env(monkeypatch):
    monkeypatch.setenv("FINTFM_PROBE_ONEHOT", '{"min_count": 3, "max_columns": 256, "max_levels": 64}')
    over = FinTFMLinearProbeModel._env_entry_overrides()["probe_onehot"]
    assert over == {"min_count": 3, "max_columns": 256, "max_levels": 64}
    monkeypatch.setenv("FINTFM_PROBE_ONEHOT", "on")
    assert FinTFMLinearProbeModel._env_entry_overrides()["probe_onehot"] is True
    assert FinTFMLinearProbeModel._check_onehot(True, source="x")["max_levels"] is None


@pytest.mark.parametrize(
    "bad", [{"max_levels": 1}, {"max_levels": 0}, {"max_levels": 2.5}, {"max_levels": True}, {"max_levels": "8"}]
)
def test_max_levels_rejects_bad_values(bad):
    with pytest.raises(ValueError, match="max_levels"):
        FinTFMLinearProbeModel._check_onehot(bad, source="probe_onehot")


# ---- gate 2: reg_rff_max_rows --------------------------------------------------------------


def fit_reg(tmp_path, checkpoint, **hp):
    """Fit a regression model on a nonlinear target, where rff differs from plain ridge."""
    X, y, X_test = data("regression", categorical=True)
    y = pd.Series(np.sin(2.5 * X["f0"].to_numpy()) * 2.0 + X["f1"].to_numpy() ** 2, index=X.index)
    m = make("regression", tmp_path, checkpoint, probe_regression=PROBE, **hp)
    m.fit(X=X, y=y)
    return m, X_test


def test_rff_cap_none_reproduces_plain_rff(tmp_path, checkpoint):
    plain, X_test = fit_reg(tmp_path / "a", checkpoint, reg_readout="rff")
    explicit, _ = fit_reg(tmp_path / "b", checkpoint, reg_readout="rff", reg_rff_max_rows=None)
    assert plain._reg_readout_eff == explicit._reg_readout_eff == "rff"
    assert np.array_equal(plain.predict(X_test), explicit.predict(X_test))


def test_rff_cap_fires_above_and_not_below(tmp_path, checkpoint, caplog):
    ridge, X_test = fit_reg(tmp_path / "r", checkpoint, reg_readout="ridge")
    rff, _ = fit_reg(tmp_path / "f", checkpoint, reg_readout="rff")
    assert not np.array_equal(ridge.predict(X_test), rff.predict(X_test))  # the arms differ at all
    with caplog.at_level(logging.WARNING):
        above, _ = fit_reg(tmp_path / "a", checkpoint, reg_readout="rff", reg_rff_max_rows=N_TRAIN - 1)
    assert above._reg_readout_eff == "ridge"
    assert np.array_equal(above.predict(X_test), ridge.predict(X_test))
    # The entry cap (5,000) also logs "kept" for the plain rff fit above; only the fall-back is counted.
    assert len([r for r in caplog.records if "exceed reg_rff_max_rows" in r.getMessage()]) == 1
    at_cap, _ = fit_reg(tmp_path / "b", checkpoint, reg_readout="rff", reg_rff_max_rows=N_TRAIN)
    assert at_cap._reg_readout_eff == "rff"
    assert np.array_equal(at_cap.predict(X_test), rff.predict(X_test))


def test_rff_cap_is_inert_for_ridge(tmp_path, checkpoint):
    base, X_test = fit_reg(tmp_path / "a", checkpoint, reg_readout="ridge")
    capped, _ = fit_reg(tmp_path / "b", checkpoint, reg_readout="ridge", reg_rff_max_rows=1)
    assert capped._reg_readout_eff == "ridge"
    assert np.array_equal(base.predict(X_test), capped.predict(X_test))


def test_rff_cap_env_reaches_the_fit(monkeypatch, tmp_path, checkpoint):
    monkeypatch.setenv("FINTFM_REG_READOUT", "rff")
    monkeypatch.setenv("FINTFM_REG_RFF_MAX_ROWS", "10")
    m, _ = fit_reg(tmp_path, checkpoint)
    assert m._get_model_params()["reg_rff_max_rows"] == 10
    assert m._reg_readout == "rff"
    assert m._reg_readout_eff == "ridge"


@pytest.mark.parametrize("bad", ["0", "-3", "1.5", "abc"])
def test_rff_cap_env_rejects_bad_values(monkeypatch, bad):
    monkeypatch.setenv("FINTFM_REG_RFF_MAX_ROWS", bad)
    with pytest.raises(ValueError, match="FINTFM_REG_RFF_MAX_ROWS"):
        FinTFMLinearProbeModel._env_entry_overrides()


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "10"])
def test_rff_cap_rejects_bad_values(bad):
    with pytest.raises(ValueError, match="reg_rff_max_rows"):
        FinTFMLinearProbeModel._check_reg_rff_max_rows(bad, source="reg_rff_max_rows")
    assert FinTFMLinearProbeModel._check_reg_rff_max_rows(None, source="x") is None
