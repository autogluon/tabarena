"""Ridge on random Fourier features: a nonlinear regression readout over the FinTFM representation.

The ``reg_readout="rff"`` option of :class:`~tabarena.models.fintfm_linprobe.model.FinTFMLinearProbeModel`
swaps the linear ridge for :class:`RFFRidgeCV`. It is still a readout on the representation (plus the
squashed raw features): the input is the same standardised matrix the ridge reads. The estimator
maps it to ``[x, sqrt(2) * cos(x W + b)]`` (the linear part is kept, so a linear target costs
nothing), with ``W`` Gaussian and fixed by the seed, and fits a ridge on that. Bandwidth and alpha
are chosen by an inner K-fold on the training rows only, from one Gram matrix per fold, and
plain ridge (no random features) is one of the candidates, so the choice cannot be worse than
the linear readout on the inner folds.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

#: Bandwidth multipliers; ``gamma = multiplier / n_features`` of the RBF ``exp(-gamma |x - x'|^2)``.
DEFAULT_GAMMA_SCALES = (0.1, 0.4, 1.6)
#: Largest total column count (linear + random features) the readout builds; bounds the
#: ``O(n p^2 + K p^3)`` cost per candidate.
MAX_TOTAL_COLUMNS = 2048
#: Fewest random features kept when the linear part is wide.
MIN_RANDOM_FEATURES = 128
_CHUNK = 4096


class RFFRidgeCV(RegressorMixin, BaseEstimator):
    """Ridge on ``[x, random Fourier features of x]``, bandwidth and alpha chosen by inner K-fold.

    Args:
        alphas: Ridge penalties to choose from.
        n_features: Random Fourier features wanted (reduced so the total column count stays
            within ``MAX_TOTAL_COLUMNS``, and to at most ``n_rows // 2``).
        gamma_scales: Bandwidth candidates, as multipliers of ``1 / n_input_columns``.
        cv: Inner folds.
        random_state: Seed of the features and the folds.

    Attributes:
        alpha_: The chosen penalty.
        gamma_scale_: The chosen bandwidth multiplier; ``0.0`` when plain ridge won.
        cv_scores_: Inner-fold mean squared error per ``(gamma_scale, alpha)`` candidate.
    """

    def __init__(
        self,
        alphas=(1.0,),
        n_features: int = 1000,
        gamma_scales=DEFAULT_GAMMA_SCALES,
        cv: int = 4,
        random_state: int = 0,
    ):
        self.alphas = alphas
        self.n_features = n_features
        self.gamma_scales = gamma_scales
        self.cv = cv
        self.random_state = random_state

    def _design(self, X: np.ndarray, scale: float) -> np.ndarray:
        """The design matrix for bandwidth multiplier ``scale`` (``0`` is the linear part alone)."""
        if scale == 0.0 or self._d_rff == 0:
            return X
        out = np.empty((X.shape[0], X.shape[1] + self._d_rff), dtype=np.float64)
        out[:, : X.shape[1]] = X
        w = self._w0 * np.sqrt(2.0 * scale / X.shape[1])
        out[:, X.shape[1] :] = np.sqrt(2.0) * np.cos(X @ w + self._b0)
        return out

    @staticmethod
    def _gram(Z: np.ndarray, y: np.ndarray) -> tuple:
        """Sufficient statistics ``(Z'Z, Z'y, sum Z, sum y, n)`` of a block, in chunks."""
        p = Z.shape[1]
        g = np.zeros((p, p))
        c = np.zeros(p)
        for i in range(0, len(Z), _CHUNK):
            zi = Z[i : i + _CHUNK]
            g += zi.T @ zi
            c += zi.T @ y[i : i + _CHUNK]
        return g, c, Z.sum(axis=0), float(y.sum()), len(y)

    @staticmethod
    def _solve_all(stats: tuple, alphas: np.ndarray) -> tuple:
        """Centred ridge solutions for every alpha from one eigendecomposition.

        Returns:
            ``(coefs (n_alpha, p), intercept, mean_z (p,))``; a prediction is
            ``(z - mean_z) @ coef + intercept``.
        """
        g, c, sz, sy, n = stats
        mz = sz / n
        my = sy / n
        gc = g - n * np.outer(mz, mz)
        cc = c - n * mz * my
        evals, evecs = np.linalg.eigh(gc)
        evals = np.clip(evals, 0.0, None)
        proj = evecs.T @ cc
        coefs = np.stack([evecs @ (proj / (evals + a)) for a in alphas])
        return coefs, my, mz

    def fit(self, X, y):
        """Fit on standardised inputs ``X`` and targets ``y``; choose bandwidth and alpha by inner K-fold."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, d = X.shape
        alphas = np.asarray(self.alphas, dtype=np.float64)
        rng = np.random.default_rng(int(self.random_state))
        room = max(MIN_RANDOM_FEATURES, MAX_TOTAL_COLUMNS - d)
        self._d_rff = int(min(self.n_features, room, max(n // 2, 1)))
        self._w0 = rng.standard_normal((d, self._d_rff))
        self._b0 = rng.uniform(0.0, 2.0 * np.pi, self._d_rff)
        k = int(self.cv)
        folds = np.arange(n) % k
        rng.shuffle(folds)
        scales = [0.0, *[float(s) for s in self.gamma_scales]] if self._d_rff > 0 else [0.0]
        self.cv_scores_ = {}
        for s in scales:
            Z = self._design(X, s)
            parts = [self._gram(Z[folds == f], y[folds == f]) for f in range(k)]
            tot = tuple(sum(p[i] for p in parts) for i in range(5))
            sse = np.zeros(len(alphas))
            for f in range(k):
                tr = tuple(tot[i] - parts[f][i] for i in range(5))
                coefs, icpt, mz = self._solve_all(tr, alphas)
                te = folds == f
                pred = (Z[te] - mz) @ coefs.T + icpt
                sse += ((pred - y[te, None]) ** 2).sum(axis=0)
            for a, e in zip(alphas, sse / n, strict=False):
                self.cv_scores_[(s, float(a))] = float(e)
        (self.gamma_scale_, self.alpha_) = min(self.cv_scores_, key=self.cv_scores_.get)
        Z = self._design(X, self.gamma_scale_)
        coefs, icpt, mz = self._solve_all(self._gram(Z, y), np.asarray([self.alpha_]))
        self.coef_, self.intercept_, self._mz = coefs[0], float(icpt), mz
        return self

    def predict(self, X) -> np.ndarray:
        """Predict targets for standardised inputs ``X``."""
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(len(X))
        for i in range(0, len(X), _CHUNK):
            out[i : i + _CHUNK] = (self._design(X[i : i + _CHUNK], self.gamma_scale_) - self._mz) @ self.coef_
        return out + self.intercept_
