"""FinTFM declares binary only unless ``FINTFM_ALLOW_MULTICLASS=1`` or ``FINTFM_ALLOW_REGRESSION=1`` is set before import."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_PROBE = "from tabarena.models.fintfm.model import FinTFMModel; print(','.join(FinTFMModel._supported_problem_types))"


@pytest.mark.parametrize(("flag", "expected"), [(None, "binary"), ("0", "binary"), ("1", "binary,multiclass")])
def test_multiclass_is_declared_only_when_opted_in(flag: str | None, expected: str) -> None:
    """The submitted declaration stays binary; only the explicit opt-in adds multiclass."""
    env = {k: v for k, v in os.environ.items() if k not in ("FINTFM_ALLOW_MULTICLASS", "FINTFM_ALLOW_REGRESSION")}
    if flag is not None:
        env["FINTFM_ALLOW_MULTICLASS"] = flag
    out = subprocess.run([sys.executable, "-P", "-c", _PROBE], env=env, capture_output=True, text=True, check=True)  # noqa: S603
    assert out.stdout.strip().splitlines()[-1] == expected


@pytest.mark.parametrize(
    ("multiclass", "regression", "expected"),
    [("0", "1", "binary,regression"), ("1", "1", "binary,multiclass,regression"), ("1", "0", "binary,multiclass")],
)
def test_regression_is_declared_only_when_opted_in(multiclass: str, regression: str, expected: str) -> None:
    """Regression joins the declaration only with its own opt-in, independently of multiclass."""
    env = {k: v for k, v in os.environ.items() if k not in ("FINTFM_ALLOW_MULTICLASS", "FINTFM_ALLOW_REGRESSION")}
    env["FINTFM_ALLOW_MULTICLASS"] = multiclass
    env["FINTFM_ALLOW_REGRESSION"] = regression
    out = subprocess.run([sys.executable, "-P", "-c", _PROBE], env=env, capture_output=True, text=True, check=True)  # noqa: S603
    assert out.stdout.strip().splitlines()[-1] == expected
