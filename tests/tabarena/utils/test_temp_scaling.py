from __future__ import annotations

import numpy as np
import pytest

from tabarena.utils.temp_scaling.numpy_calibrators import NumpyTemperatureScalingCalibrator


def _toy_probs_labels(n: int = 200, c: int = 4, seed: int = 0):
    """Random but valid (n, c) probabilities + integer labels, all float64."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((n, c))
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)
    labels = rng.integers(0, c, size=n)
    return probs, labels


def test_numpy_temperature_scaling_returns_valid_probs():
    """The torch-free calibrator produces a valid, finite calibrated distribution."""
    probs, labels = _toy_probs_labels()
    cal = NumpyTemperatureScalingCalibrator().fit(probs, labels)

    out = cal.predict_proba(probs)
    assert out.shape == probs.shape
    np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-6)
    assert np.isfinite(cal.invtemp_)
    assert cal.invtemp_ > 0


def test_numpy_temperature_scaling_identity_on_calibrated_input():
    """Labels drawn exactly from the predicted distribution => ~no rescaling (invtemp ~ 1)."""
    rng = np.random.default_rng(0)
    probs, _ = _toy_probs_labels(n=4000, c=3, seed=1)
    labels = np.array([rng.choice(probs.shape[1], p=row) for row in probs])
    cal = NumpyTemperatureScalingCalibrator().fit(probs, labels)
    assert cal.invtemp_ == pytest.approx(1.0, abs=0.25)


def test_numpy_matches_torch_temperature_scaling():
    """The numpy fallback matches the torch v2 (bisection) calibrator it replaces."""
    pytest.importorskip("torch")
    from tabarena.utils.temp_scaling.calibrators import TemperatureScalingCalibrator

    probs, labels = _toy_probs_labels()

    np_cal = NumpyTemperatureScalingCalibrator().fit(probs, labels)

    torch_cal = TemperatureScalingCalibrator(max_iter=200, lr=0.1)  # opt="bisection" by default
    torch_cal.fit(X=probs, y=labels)

    np.testing.assert_allclose(np_cal.invtemp_, torch_cal.invtemp_, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np_cal.predict_proba(probs),
        torch_cal.predict_proba(probs),
        rtol=1e-4,
        atol=1e-6,
    )
