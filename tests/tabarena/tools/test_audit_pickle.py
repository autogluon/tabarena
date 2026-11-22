"""Tests for ``tabarena.tools.audit_pickle``.

Tiny ``AbstractModel`` subclasses stand in for an in-context model: one keeps a per-row tensor cache
(AutoGluon writes it beside ``model.pkl``), one keeps the same cache behind a ``__reduce__`` that
serializes it to bytes (so it lands in the pickle), one grows its pickle without tensors (like tree
nodes), one stores only the training table.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autogluon.core.models import AbstractModel

from tabarena.tools.audit_pickle import audit_pickle, tensor_nbytes

#: Bytes per training row of the dummy cache (1024 float32).
CACHE_BYTES_PER_ROW = 4096


class _TensorCacheModel(AbstractModel):
    """Keeps a cache that grows with the training rows as a tensor on ``self.model``."""

    ag_key = "_TENSOR_CACHE"
    ag_name = "_TensorCache"

    def _fit(self, X, y, **kwargs):
        X = self.preprocess(X, is_train=True)
        self.model = {"cache": self._cache(torch.ones(len(X), CACHE_BYTES_PER_ROW // 4)), "mean": float(y.mean())}

    def _cache(self, tensor):
        return tensor

    def _predict_proba(self, X, **kwargs):
        return np.full(len(X), self.model["mean"])


class _BytesCache:
    """A cache whose ``__reduce__`` turns its tensor into bytes, as some libraries' pickle protocols do."""

    def __init__(self, tensor):
        self.tensor = tensor

    def __reduce__(self):
        return _bytes_cache_from_bytes, (self.tensor.numpy().tobytes(), tuple(self.tensor.shape))


def _bytes_cache_from_bytes(data: bytes, shape: tuple[int, ...]) -> _BytesCache:
    return _BytesCache(torch.frombuffer(bytearray(data), dtype=torch.float32).reshape(shape))


class _HiddenTensorCacheModel(_TensorCacheModel):
    """The same cache behind a ``__reduce__`` that serializes it to bytes, so it stays in the pickle."""

    ag_key = "_HIDDEN_TENSOR_CACHE"
    ag_name = "_HiddenTensorCache"

    def _cache(self, tensor):
        return _BytesCache(tensor)


class _NodesModel(AbstractModel):
    """Grows its pickle with the training rows without tensors, like the nodes of a forest."""

    ag_key = "_NODES"
    ag_name = "_Nodes"

    def _fit(self, X, y, **kwargs):
        self.model = [bytes(CACHE_BYTES_PER_ROW) for _ in range(len(X))]


class _TableModel(AbstractModel):
    """Stores only the training table, like an in-context model without a cache."""

    ag_key = "_TABLE"
    ag_name = "_Table"

    def _fit(self, X, y, **kwargs):
        self.model = (self.preprocess(X, is_train=True), y.copy())


def test_tensor_nbytes_counts_a_shared_storage_once():
    base = torch.zeros(4, 256)
    assert tensor_nbytes({"a": base[:2], "b": [base[2:], base]}) == base.untyped_storage().nbytes()


class _RefusingMapping(Mapping):
    """A mapping that refuses iteration, like the urllib3 object a walk through ``sys.modules`` hit."""

    def __getitem__(self, key):
        raise KeyError(key)

    def __iter__(self):
        raise TypeError("refuses iteration")

    def __len__(self):
        return 0


def test_tensor_nbytes_skips_modules():
    """A module on the model (SciPy's ``BSpline._xp``) is not walked: its state is not the model's."""
    module = ModuleType("_array_namespace")
    module.headers = _RefusingMapping()
    module.weights = torch.zeros(1024)
    cache = torch.zeros(4, 256)
    assert tensor_nbytes(SimpleNamespace(xp=module, cache=cache)) == cache.untyped_storage().nbytes()


def test_tensor_nbytes_walks_past_a_scipy_spline():
    interpolate = pytest.importorskip("scipy.interpolate")
    spline = interpolate.make_interp_spline(np.arange(10.0), np.arange(10.0) ** 2)
    cache = torch.zeros(4, 256)
    assert tensor_nbytes({"spline": spline, "cache": cache}) == cache.untyped_storage().nbytes()


@pytest.mark.parametrize(
    ("model_cls", "flagged", "noted"),
    [
        (_TensorCacheModel, False, False),
        (_HiddenTensorCacheModel, True, False),
        (_NodesModel, False, True),
        (_TableModel, False, False),
    ],
)
def test_audit_pickle_flags_tensors_inside_the_pickle(model_cls, flagged, noted):
    data = audit_pickle(model_cls, problem_type="binary", hyperparameters={}, n_rows=64)
    per_row = data["bytes_per_row"]
    assert data["flag"] is flagged
    assert data["note"] is noted
    if flagged:
        assert per_row["model_pkl_bytes"] >= CACHE_BYTES_PER_ROW
    if model_cls is _TensorCacheModel:
        assert per_row["tensors_file_bytes"] >= CACHE_BYTES_PER_ROW
        assert per_row["model_pkl_bytes"] < CACHE_BYTES_PER_ROW / 16
