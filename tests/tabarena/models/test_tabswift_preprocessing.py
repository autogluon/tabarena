from __future__ import annotations

import numpy as np
import pytest

from tabarena.models.tabswift._vendor.preprocessing import PreprocessingPipeline, _clip_to_finite_range


def test_clip_keeps_finite_values_and_nans_and_bounds_infinities():
    X = np.array([[1.0, np.inf], [-np.inf, np.nan]], dtype=np.float32)
    out = _clip_to_finite_range(X)
    assert out[0, 0] == 1.0
    assert np.isnan(out[1, 1])
    assert np.isfinite(out[0, 1]) and np.isfinite(out[1, 0])
    assert out[0, 1] > 0 > out[1, 0]


@pytest.mark.parametrize("method", ["power", "none"])
def test_pipeline_transform_survives_a_normalizer_overflow(method):
    """A normalizer output with inf (float32 overflow of a heavy-tailed column) must not fail the outlier remover."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4)).astype(np.float32)
    pipe = PreprocessingPipeline(normalization_method=method).fit(X)
    if pipe.normalizer_ is not None:
        original = pipe.normalizer_.transform

        def overflowing_transform(Z):
            out = original(Z)
            out[0, 0] = np.inf
            out[1, 1] = -np.inf
            return out

        pipe.normalizer_.transform = overflowing_transform
    out = pipe.transform(X)
    assert np.isfinite(out).all()
