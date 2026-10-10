"""TabFM+ output-codes label sets wider than the checkpoint's head, as the TabFM model does.

The TabFM network is replaced by a nearest-centroid stand-in with the same ten-class limit, so the
test checks the wiring (which estimator is built, with which preset, and that the probabilities
cover every class) without loading the checkpoint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("tabpfn_extensions.many_class")

from tabarena.models.tabfm import model as tabfm_model
from tabarena.systems.tabfm_plus import system as tabfm_plus_system
from tabarena.systems.tabfm_plus.system import TabFMPlusSystemModel

HEAD = 10


class _Network:
    def parameters(self):
        return iter(())


class _CentroidEstimator:
    """Nearest-centroid classifier that, like TabFM's, rejects more than ``HEAD`` classes."""

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) > HEAD:
            raise ValueError(f"The number of classes ({len(self.classes_)}) exceeds the maximum number of classes")
        X = np.asarray(X, dtype=float)
        self._centroids = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict_proba(self, X):
        d = ((np.asarray(X, dtype=float)[:, None, :] - self._centroids[None]) ** 2).sum(-1)
        p = np.exp(-(d - d.min(axis=1, keepdims=True)))
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


@pytest.fixture
def fake_tabfm(monkeypatch):
    """Record every TabFM estimator the fit builds, with its preset."""
    built = []

    def build(*, problem_type, device, interface, network=None, **hps):
        built.append(interface)
        estimator = _CentroidEstimator()
        estimator.model = _Network()
        return estimator

    monkeypatch.setattr(tabfm_model, "_build_tabfm_estimator", build)
    monkeypatch.setattr(tabfm_plus_system, "_build_tabfm_estimator", build)
    monkeypatch.setattr(tabfm_plus_system, "_load_tabfm_network", lambda **_: _Network())
    return built


def _fit(n_classes: int, **init):
    rng = np.random.default_rng(0)
    y = pd.Series(np.repeat([f"c{i}" for i in range(n_classes)], 20), name="target")
    X = pd.DataFrame(rng.normal(size=(len(y), 4)), columns=[f"f{i}" for i in range(4)])
    X["f0"] += np.repeat(np.arange(n_classes), 20) * 8.0
    model = TabFMPlusSystemModel(problem_type="multiclass", eval_metric=None, device="cpu", **init)
    model._fit_system(
        X,
        y,
        target_name="target",
        problem_type="multiclass",
        eval_metric=None,
        validation_metadata=None,
        num_cpus=1,
        num_gpus=0,
        memory_limit=None,
        time_limit=None,
        random_state=0,
    )
    return model, X, y


@pytest.mark.parametrize("interface", ["ensemble", "default"])
def test_more_classes_than_the_head_are_output_coded(fake_tabfm, interface):
    model, X, y = _fit(15, interface=interface)
    proba = model._predict_proba(X)
    # One estimator per code row (fit when predicting, as for an in-context model), all with the preset.
    assert len(fake_tabfm) > 1 and set(fake_tabfm) == {interface}
    assert sorted(proba.columns) == sorted(y.unique())
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba.idxmax(axis=1) == y).mean() > 0.9
    assert len(model._predict(X)) == len(X)


def test_within_the_head_fits_one_estimator(fake_tabfm):
    model, X, y = _fit(5)
    assert fake_tabfm == ["ensemble"]
    assert sorted(model._predict_proba(X).columns) == sorted(y.unique())


def test_threshold_is_an_init_knob(fake_tabfm):
    model, X, _ = _fit(5, many_class_threshold=3)
    model._predict_proba(X)
    assert len(fake_tabfm) > 1
