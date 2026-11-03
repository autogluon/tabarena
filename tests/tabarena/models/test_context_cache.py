"""Tests for :mod:`tabarena.models.context_cache` and the ``audit_pickle`` tool that checks for it.

Tiny ``AbstractModel`` subclasses stand in for an in-context model: one keeps a per-row "cache" inside
the pickle, one keeps the same cache in its own file through ``ContextCacheFileMixin``, one grows its pickle
without tensors (like tree nodes), one stores only the training table.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from autogluon.core.models import AbstractModel

from tabarena.models.context_cache import ContextCacheFileMixin, tensor_nbytes
from tabarena.tools.audit_pickle import audit_pickle

#: Bytes per training row of the dummy cache (1024 float32).
CACHE_BYTES_PER_ROW = 4096


class _CacheInPickleModel(AbstractModel):
    """Keeps a cache that grows with the training rows on ``self.model``, inside the pickle."""

    ag_key = "_CACHE_IN_PICKLE"
    ag_name = "_CacheInPickle"

    def _fit(self, X, y, **kwargs):
        X = self.preprocess(X, is_train=True)
        self.model = {"cache": torch.ones(len(X), CACHE_BYTES_PER_ROW // 4), "mean": float(y.mean())}

    def _predict_proba(self, X, **kwargs):
        mean = self.model["mean"] + float(self.model["cache"].mean()) - 1.0
        return np.full(len(X), mean)


class _CacheFileModel(ContextCacheFileMixin, _CacheInPickleModel):
    """The same cache, saved beside ``model.pkl``."""

    ag_key = "_CACHE_FILE"
    ag_name = "_CacheFile"

    def _get_context_cache(self):
        return None if self.model is None else self.model.get("cache")

    def _set_context_cache(self, cache) -> None:
        self.model["cache"] = cache


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


def _fit(model_cls, tmp_path, n_rows: int = 64):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n_rows, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] > 0).astype(int))
    model = model_cls(path=str(tmp_path), name="m", problem_type="binary", eval_metric="log_loss")
    model.fit(X=X, y=y)
    return model, X


def test_cache_is_saved_beside_the_pickle_and_counted_in_the_size(tmp_path):
    model, X = _fit(_CacheFileModel, tmp_path)
    expected = model.predict_proba(X)
    cache_bytes = tensor_nbytes(model._get_context_cache())
    path = model.save()

    assert os.path.getsize(os.path.join(path, "model.pkl")) < cache_bytes
    assert os.path.exists(os.path.join(path, _CacheFileModel.context_cache_file_name))
    assert model._get_context_cache() is not None  # put back after the save
    with model._without_context_cache():
        pickled = len(pickle.dumps(model, protocol=4))
    assert pickled + cache_bytes <= model._get_pickled_size() <= pickled + cache_bytes + 1024

    loaded = _CacheFileModel.load(path)
    np.testing.assert_allclose(loaded.predict_proba(X), expected)
    # A re-save writes over the file the loaded cache is memory-mapped from.
    loaded.save()
    np.testing.assert_allclose(_CacheFileModel.load(path).predict_proba(X), expected)


def test_a_save_without_a_cache_removes_a_stale_cache_file(tmp_path):
    model, _ = _fit(_CacheFileModel, tmp_path)
    path = model.save()
    model._set_context_cache(None)
    model.save()
    assert not os.path.exists(os.path.join(path, _CacheFileModel.context_cache_file_name))


def test_tensor_nbytes_counts_a_shared_storage_once():
    base = torch.zeros(4, 256)
    assert tensor_nbytes({"a": base[:2], "b": [base[2:], base]}) == base.untyped_storage().nbytes()


@pytest.mark.parametrize(
    ("model_cls", "flagged", "noted"),
    [
        (_CacheInPickleModel, True, False),
        (_CacheFileModel, False, False),
        (_NodesModel, False, True),
        (_TableModel, False, False),
    ],
)
def test_audit_pickle_flags_a_cache_inside_the_pickle(model_cls, flagged, noted):
    data = audit_pickle(model_cls, problem_type="binary", hyperparameters={}, n_rows=64)
    assert data["flag"] is flagged
    assert data["note"] is noted
    if flagged:
        assert data["bytes_per_row"]["model_pkl_bytes"] >= CACHE_BYTES_PER_ROW
    if model_cls is _CacheFileModel:
        assert data["bytes_per_row"]["other_files_bytes"] >= CACHE_BYTES_PER_ROW
