"""Torch-free temperature scaling.

A numpy-only reimplementation of the default (``opt="bisection"``)
:class:`tabarena.utils.temp_scaling.calibrators.TemperatureScalingCalibrator`.
Selected explicitly via ``ConfigResult.temp_scale(method="v2_numpy")`` (e.g. on
the minimal CPU/eval install without torch) — it is an opt-in method, never a
silent fallback. It is numerically equivalent to the torch "v2" path: the same
single inverse-temperature found by bisection on the analytic NLL gradient, the
same ``log(p + 1e-30)`` probability->logit transform, and the same
``softmax(invtemp * logits)`` output.
"""

from __future__ import annotations

import numpy as np

# Matches CategoricalProbs.get_logits() in distributions.py (torch path).
_LOGIT_EPS = 1e-30
# Matches bisection_search bounds/steps in calibrators.py.
_BISECTION_LO = -16.0
_BISECTION_HI = 16.0
_BISECTION_STEPS = 30


def _softmax(z: np.ndarray) -> np.ndarray:
    """Numerically-stable row-wise softmax (equivalent to ``torch.softmax(..., dim=-1)``)."""
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def _bisection(f, a: float, b: float, n_steps: int) -> float:
    """Root-find by bisection, mirroring ``calibrators.bisection_search``."""
    for _ in range(n_steps):
        c = a + 0.5 * (b - a)
        if f(c) > 0:
            b = c
        else:
            a = c
    return 0.5 * (a + b)


class NumpyTemperatureScalingCalibrator:
    """Torch-free temperature-scaling calibrator (numpy only).

    Drop-in for ``TemperatureScalingCalibrator(opt="bisection")``: ``fit`` takes
    validation probabilities ``X`` (shape ``(n, n_classes)``) and integer labels
    ``y``, and ``predict_proba`` returns temperature-scaled probabilities.
    """

    def __init__(self, max_bisection_steps: int = _BISECTION_STEPS, max_iter: int = 200, lr: float = 0.1):
        # ``max_iter``/``lr`` are accepted only for signature parity with the
        # torch calibrator; the bisection optimiser does not use them.
        self.max_bisection_steps = max_bisection_steps
        self.max_iter = max_iter
        self.lr = lr
        self.invtemp_: float = 1.0
        self.classes_: list[int] = []

    def _to_logits(self, X: np.ndarray) -> np.ndarray:
        return np.log(np.asarray(X, dtype=np.float64) + _LOGIT_EPS)

    def _loss_grad(self, invtemp: float, logits: np.ndarray, y: np.ndarray) -> float:
        # d/d(invtemp) of the mean cross-entropy of softmax(invtemp * logits).
        probs = _softmax(invtemp * logits)
        part_1 = np.mean(np.sum(logits * probs, axis=-1))
        part_2 = np.mean(logits[np.arange(logits.shape[0]), y])
        return float(part_1 - part_2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> NumpyTemperatureScalingCalibrator:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        self.classes_ = list(range(X.shape[-1]))
        logits = self._to_logits(X)
        u = _bisection(
            lambda u: self._loss_grad(np.exp(u), logits, y),
            a=_BISECTION_LO,
            b=_BISECTION_HI,
            n_steps=self.max_bisection_steps,
        )
        self.invtemp_ = float(np.exp(u))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _softmax(self.invtemp_ * self._to_logits(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.classes_)[np.argmax(self.predict_proba(X), axis=-1)]
