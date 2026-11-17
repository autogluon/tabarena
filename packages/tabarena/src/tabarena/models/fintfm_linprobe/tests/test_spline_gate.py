# ruff: noqa: F811
"""Row gate ``spline_max_rows``: ``probe_spline`` is off above the cap (binary entry: 10,000 rows)."""

from __future__ import annotations

import logging

import numpy as np
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

PROBE = {"scale": True, "cv": True, "raw": True}


@pytest.fixture(autouse=True)
def clean_spline_gate_env(monkeypatch):
    monkeypatch.delenv("FINTFM_SPLINE_MAX_ROWS", raising=False)
    monkeypatch.delenv("FINTFM_PROBE_SPLINE", raising=False)


def fit_binary(tmp_path, checkpoint, problem_type="binary", **hp):
    X, y, X_test = data(problem_type)
    m = make(problem_type, tmp_path, checkpoint, **{f"probe_{problem_type}": PROBE, **hp})
    m.fit(X=X, y=y)
    return m, np.asarray(m.predict_proba(X_test) if problem_type != "regression" else m.predict(X_test))


def gate_msgs(caplog):
    return [r.getMessage() for r in caplog.records if "spline_max_rows" in r.getMessage()]


def test_default_none_never_gates(tmp_path, checkpoint, caplog):
    with caplog.at_level(logging.WARNING):
        # The binary entry gates at 10,000 rows, above this test's 240; the explicit None is the gate-off case.
        base, p_base = fit_binary(tmp_path / "a", checkpoint, probe_spline=True, spline_max_rows=None)
        explicit, p_explicit = fit_binary(tmp_path / "b", checkpoint, probe_spline=True, spline_max_rows=None)
    assert base._spline_max_rows is None
    assert base._raw_spline is not None and explicit._raw_spline is not None
    assert np.array_equal(p_base, p_explicit)
    assert not gate_msgs(caplog)


def test_above_cap_matches_no_spline(tmp_path, checkpoint, caplog):
    with caplog.at_level(logging.WARNING):
        above, p_above = fit_binary(tmp_path / "a", checkpoint, probe_spline=True, spline_max_rows=N_TRAIN - 1)
    off, p_off = fit_binary(tmp_path / "b", checkpoint, probe_spline=None)
    on, p_on = fit_binary(tmp_path / "c", checkpoint, probe_spline=True, spline_max_rows=None)
    assert not np.array_equal(p_off, p_on)  # the arms differ at all
    assert above._spline_cfg is None and above._raw_spline is None
    assert np.array_equal(p_above, p_off)
    msgs = gate_msgs(caplog)
    assert len(msgs) == 1 and "gated off" in msgs[0]


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_at_cap_matches_ungated(tmp_path, checkpoint, caplog, problem_type):
    on, p_on = fit_binary(tmp_path / "c", checkpoint, problem_type, probe_spline=True, spline_max_rows=None)
    with caplog.at_level(logging.WARNING):
        at_cap, p_at = fit_binary(tmp_path / "a", checkpoint, problem_type, probe_spline=True, spline_max_rows=N_TRAIN)
    assert at_cap._raw_spline is not None
    assert np.array_equal(p_at, p_on)
    msgs = gate_msgs(caplog)
    assert len(msgs) == 1 and "kept" in msgs[0]


def test_gate_without_probe_spline_is_a_no_op(tmp_path, checkpoint, caplog):
    with caplog.at_level(logging.WARNING):
        gated, p_gated = fit_binary(tmp_path / "a", checkpoint, probe_spline=None, spline_max_rows=N_TRAIN - 1)
    plain, p_plain = fit_binary(tmp_path / "b", checkpoint, probe_spline=None)
    assert np.array_equal(p_gated, p_plain)
    assert not gate_msgs(caplog)


def test_env_var_sets_the_gate(tmp_path, checkpoint, monkeypatch):
    monkeypatch.setenv("FINTFM_SPLINE_MAX_ROWS", str(N_TRAIN - 1))
    assert FinTFMLinearProbeModel._env_entry_overrides()["spline_max_rows"] == N_TRAIN - 1
    m, _ = fit_binary(tmp_path, checkpoint, probe_spline=True)
    assert m._spline_cfg is None and m._raw_spline is None
    for off in ("off", "none", "OFF"):
        monkeypatch.setenv("FINTFM_SPLINE_MAX_ROWS", off)
        assert FinTFMLinearProbeModel._env_entry_overrides()["spline_max_rows"] is None


@pytest.mark.parametrize("good", [None, 1, 5000])
def test_validator_accepts(good):
    assert FinTFMLinearProbeModel._check_spline_max_rows(good, source="spline_max_rows") == good


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, "100"])
def test_bad_values_raise(bad, tmp_path, checkpoint):
    with pytest.raises(ValueError, match="spline_max_rows"):
        FinTFMLinearProbeModel._check_spline_max_rows(bad, source="spline_max_rows")
    with pytest.raises(ValueError, match="spline_max_rows"):
        fit_binary(tmp_path, checkpoint, probe_spline=True, spline_max_rows=bad)


@pytest.mark.parametrize("bad", ["abc", "0", "-1", "2.5"])
def test_bad_env_values_raise(bad, monkeypatch):
    monkeypatch.setenv("FINTFM_SPLINE_MAX_ROWS", bad)
    with pytest.raises(ValueError, match="FINTFM_SPLINE_MAX_ROWS"):
        FinTFMLinearProbeModel._env_entry_overrides()
