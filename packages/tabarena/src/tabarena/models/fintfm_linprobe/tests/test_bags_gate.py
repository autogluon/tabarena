# ruff: noqa: F811
"""Default-off row gate ``partition_bags_max_rows``: one partition bag above the cap."""

from __future__ import annotations

import logging

import numpy as np
import pytest

pytest.importorskip("fintfm")
pytest.importorskip("torch")

from tabarena.models.fintfm_linprobe.model import FinTFMLinearProbeModel
from tabarena.models.fintfm_linprobe.tests.test_crossfit import (  # noqa: F401 (fixtures)
    N_TRAIN,
    PARTITION,
    checkpoint,
    clean_env,
    data,
    make,
)

BAGGED = {**PARTITION, "n_bags": 2}
PLAIN = {k: v for k, v in PARTITION.items() if k != "n_bags"}


@pytest.fixture(autouse=True)
def clean_bags_env(monkeypatch):
    monkeypatch.delenv("FINTFM_PARTITION_BAGS_MAX_ROWS", raising=False)


def fit_binary(tmp_path, checkpoint, partition, **hp):
    X, y, X_test = data("binary")
    m = make("binary", tmp_path, checkpoint, partition_binary=partition, **hp)
    m.fit(X=X, y=y)
    return m, m.predict_proba(X_test)


def test_default_none_keeps_n_bags(tmp_path, checkpoint, caplog):
    # The binary entry of the PR branch gates at 5,000 rows; the explicit None is the gate-off case.
    with caplog.at_level(logging.WARNING):
        gated, p_gated = fit_binary(tmp_path / "a", checkpoint, BAGGED, partition_bags_max_rows=None)
        explicit, p_explicit = fit_binary(tmp_path / "b", checkpoint, BAGGED, partition_bags_max_rows=None)
    assert gated._bags_max_rows is None
    assert gated.model.n_bags == explicit.model.n_bags == 2
    assert np.array_equal(p_gated, p_explicit)
    assert not [r for r in caplog.records if "partition_bags_max_rows" in r.getMessage()]


def test_above_cap_matches_no_n_bags(tmp_path, checkpoint, caplog):
    with caplog.at_level(logging.WARNING):
        above, p_above = fit_binary(tmp_path / "a", checkpoint, BAGGED, partition_bags_max_rows=N_TRAIN - 1)
    single, p_single = fit_binary(tmp_path / "b", checkpoint, PLAIN)
    bagged, p_bagged = fit_binary(tmp_path / "c", checkpoint, BAGGED, partition_bags_max_rows=None)
    assert not np.array_equal(p_single, p_bagged)  # the arms differ at all
    assert above.model.n_bags == single.model.n_bags == 1
    assert np.array_equal(p_above, p_single)
    msgs = [r.getMessage() for r in caplog.records if "partition_bags_max_rows" in r.getMessage()]
    assert len(msgs) == 1 and "n_bags=2 -> 1" in msgs[0]


def test_at_or_below_cap_matches_plain_n_bags(tmp_path, checkpoint, caplog):
    bagged, p_bagged = fit_binary(tmp_path / "c", checkpoint, BAGGED, partition_bags_max_rows=None)
    with caplog.at_level(logging.WARNING):
        at_cap, p_at = fit_binary(tmp_path / "a", checkpoint, BAGGED, partition_bags_max_rows=N_TRAIN)
    assert at_cap.model.n_bags == 2
    assert np.array_equal(p_at, p_bagged)
    msgs = [r.getMessage() for r in caplog.records if "partition_bags_max_rows" in r.getMessage()]
    assert len(msgs) == 1 and "kept" in msgs[0]


def test_env_var_sets_the_gate(tmp_path, checkpoint, monkeypatch):
    monkeypatch.setenv("FINTFM_PARTITION_BAGS_MAX_ROWS", str(N_TRAIN - 1))
    assert FinTFMLinearProbeModel._env_entry_overrides()["partition_bags_max_rows"] == N_TRAIN - 1
    m, _ = fit_binary(tmp_path, checkpoint, BAGGED)
    assert m.model.n_bags == 1
    for off in ("off", "none", "OFF"):
        monkeypatch.setenv("FINTFM_PARTITION_BAGS_MAX_ROWS", off)
        assert FinTFMLinearProbeModel._env_entry_overrides()["partition_bags_max_rows"] is None


def test_gate_without_n_bags_is_a_no_op():
    m = FinTFMLinearProbeModel.__new__(FinTFMLinearProbeModel)
    m._bags_max_rows, m._n_fit_rows = 10, 1000
    assert m._gate_partition_bags(None) is None
    assert m._gate_partition_bags({"max_depth": 1}) == {"max_depth": 1}


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, "100"])
def test_bad_values_raise(bad, tmp_path, checkpoint):
    with pytest.raises(ValueError, match="partition_bags_max_rows"):
        FinTFMLinearProbeModel._check_bags_max_rows(bad, source="partition_bags_max_rows")
    with pytest.raises(ValueError, match="partition_bags_max_rows"):
        fit_binary(tmp_path, checkpoint, BAGGED, partition_bags_max_rows=bad)


@pytest.mark.parametrize("bad", ["abc", "0", "-1", "2.5"])
def test_bad_env_values_raise(bad, monkeypatch):
    monkeypatch.setenv("FINTFM_PARTITION_BAGS_MAX_ROWS", bad)
    with pytest.raises(ValueError, match="FINTFM_PARTITION_BAGS_MAX_ROWS"):
        FinTFMLinearProbeModel._env_entry_overrides()
