from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("inference.v2_0.preprocess", reason="needs the LimiX package (limix_2 extra)")

from scipy.sparse.linalg import ArpackError
from sklearn.decomposition import TruncatedSVD

from tabarena.models.limix_2.model import _patch_svd_fallback


def test_arpack_failure_falls_back_to_the_randomized_solver(monkeypatch):
    from inference.v2_0 import preprocess

    _patch_svd_fallback()
    _patch_svd_fallback()  # idempotent
    patched = preprocess.TruncatedSVD
    assert patched is not TruncatedSVD and issubclass(patched, TruncatedSVD)

    calls: list[str] = []
    original = TruncatedSVD.fit_transform

    def flaky_fit_transform(self, X, y=None):
        calls.append(self.algorithm)
        if self.algorithm == "arpack":
            raise ArpackError(3, {3: "No shifts could be applied"})
        return original(self, X, y)

    monkeypatch.setattr(TruncatedSVD, "fit_transform", flaky_fit_transform)
    X = np.random.default_rng(0).standard_normal((30, 8))
    out = patched(algorithm="arpack", n_components=3, random_state=0).fit_transform(X)
    assert out.shape == (30, 3)
    assert calls == ["arpack", "randomized"]
