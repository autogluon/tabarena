"""``blas_threads``: fit and predict run with BLAS limited to ``min(4, num_cpus)`` threads (FINDINGS §298)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_info

from tabarena.models.fintfm_linprobe.model import FinTFMLinearProbeModel


def blas_threads_now() -> list[int]:
    """Thread counts of every BLAS library loaded in this process."""
    return [p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas"]


def make(tmp_path, **hp) -> FinTFMLinearProbeModel:
    """A binary model with the entry's defaults and the given overrides, not fitted."""
    model = FinTFMLinearProbeModel(
        path=str(tmp_path), name="blas", problem_type="binary", eval_metric="roc_auc", hyperparameters=hp
    )
    model._init_params()
    return model


@pytest.mark.parametrize(
    ("setting", "num_cpus", "expected"),
    [("auto", 8, 4), ("auto", 12, 4), ("auto", 4, 4), ("auto", 2, 2), ("auto", 1, 1), (2, 8, 2), (None, 8, None)],
)
def test_limit_is_min_of_four_and_the_fits_cpus(setting, num_cpus: int, expected: int | None) -> None:
    """``auto`` never raises the thread count above the CPUs the fit was given, nor above 4."""
    assert FinTFMLinearProbeModel._blas_limit_for(setting, num_cpus) == expected


def test_auto_without_a_cpu_count_uses_the_machines() -> None:
    """A missing or non-integer ``num_cpus`` (``"auto"`` from a caller) falls back to ``os.cpu_count``."""
    assert FinTFMLinearProbeModel._blas_limit_for("auto", "auto") == min(4, os.cpu_count() or 1)


@pytest.mark.parametrize("bad", [0, -1, 2.5, "four", True])
def test_malformed_setting_is_refused(bad) -> None:
    """Anything but ``"auto"``, ``None`` or a positive int raises."""
    with pytest.raises(ValueError, match="blas_threads"):
        FinTFMLinearProbeModel._check_blas_threads(bad, source="blas_threads")


def test_entry_default_is_auto(tmp_path) -> None:
    """Every problem type's entry limits BLAS by default."""
    assert make(tmp_path)._get_model_params()["blas_threads"] == "auto"


def test_fit_and_predict_run_under_the_limit(tmp_path, monkeypatch) -> None:
    """The readout's fit and predict bodies see BLAS at the limit; the limit is lifted afterwards."""
    if not blas_threads_now():
        pytest.skip("no BLAS library visible to threadpoolctl")
    seen: dict[str, list[int]] = {}

    def fake_fit(self, X, y, **kwargs):
        seen["fit"] = blas_threads_now()

    def fake_predict(self, X, **kwargs):
        seen["predict"] = blas_threads_now()
        return np.zeros(len(X))

    monkeypatch.setattr(FinTFMLinearProbeModel, "_fit_readout", fake_fit)
    monkeypatch.setattr(FinTFMLinearProbeModel, "_predict_proba_readout", fake_predict)
    before = blas_threads_now()
    model = make(tmp_path, blas_threads=1)
    X = pd.DataFrame({"a": [0.0, 1.0]})
    model._fit(X, pd.Series([0, 1]), num_cpus=8)
    model._predict_proba(X)
    assert model._blas_limit == 1
    assert seen["fit"] and all(n == 1 for n in seen["fit"])
    assert seen["predict"] and all(n == 1 for n in seen["predict"])
    assert blas_threads_now() == before


def test_none_leaves_blas_alone(tmp_path, monkeypatch) -> None:
    """``blas_threads=None`` runs the bodies with whatever thread count the process has."""
    seen: dict[str, list[int]] = {}
    monkeypatch.setattr(
        FinTFMLinearProbeModel, "_fit_readout", lambda self, X, y, **kw: seen.setdefault("fit", blas_threads_now())
    )
    before = blas_threads_now()
    model = make(tmp_path, blas_threads=None)
    model._fit(pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0, 1]), num_cpus=8)
    assert model._blas_limit is None
    assert seen["fit"] == before
