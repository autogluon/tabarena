"""``FINTFM_PROBE`` configures the linear probe's readout and leaves it unchanged when unset.

The options exist to improve FinTFM's multiclass and regression entries on TabArena-Lite
(fintfm DR7.6, 2026-11-11) without changing the binary entry, so the unset case must give the
same readout as before. These tests stand in for the network with a fixed representation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fintfm")

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline

from tabarena.models.fintfm.model import FinTFMModel
from tabarena.models.fintfm_linprobe.model import FinTFMLinearProbeModel


class _Network:
    """A stand-in network whose representation is two fixed columns of the row."""

    def transform_representation(self, X: np.ndarray, stage: str) -> np.ndarray:
        """Return a two-column representation of ``X``.

        Args:
            X: Query rows.
            stage: Ignored; accepted for signature compatibility.

        Returns:
            ``(n, 2)``: the first feature and a constant.
        """
        return np.column_stack([X[:, 0], np.ones(len(X))])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return a uniform three-class distribution, the head's stand-in output.

        Args:
            X: Query rows.

        Returns:
            ``(n, 3)`` probabilities.
        """
        return np.full((len(X), 3), 1.0 / 3.0)


def _fitted(monkeypatch: pytest.MonkeyPatch, probe: str | None) -> FinTFMLinearProbeModel:
    """A probe model fitted on three features with the parent fit stubbed and ``FINTFM_PROBE`` set.

    Cross-fitting is off (``probe_crossfit_folds=0``): these tests cover the readout options on a
    fixed representation, and the stand-in network has no context to restrict. The stubbed parent
    fit resolves the entry settings from the model's hyperparameters as the real one does.
    """
    model = FinTFMLinearProbeModel(
        path="",
        name="fintfm_probe_opts",
        problem_type="multiclass",
        eval_metric="log_loss",
        hyperparameters={"probe_crossfit_folds": 0},
    )
    model.random_seed = 0

    def parent_fit(self, X, y, **kwargs) -> None:
        self._resolve_entry(self._get_model_params())
        self.model = _Network()

    monkeypatch.setattr(FinTFMModel, "_fit", parent_fit)
    monkeypatch.setattr(FinTFMLinearProbeModel, "preprocess", lambda self, X, **kw: X.to_numpy(dtype=np.float32))
    monkeypatch.delenv("FINTFM_PARTITION", raising=False)
    monkeypatch.delenv("FINTFM_CHECKPOINT", raising=False)
    monkeypatch.delenv("FINTFM_CHECKPOINT_MULTICLASS", raising=False)
    if probe is None:
        monkeypatch.delenv("FINTFM_PROBE", raising=False)
    else:
        monkeypatch.setenv("FINTFM_PROBE", probe)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300), "c": rng.normal(size=300)})
    X.loc[::7, "c"] = np.nan
    y = pd.Series(np.digitize(X["b"], [-0.5, 0.5]))
    model._init_params()  # AutoGluon calls this when a fit starts; it reads FINTFM_PROBE
    model._fit(X, y)
    return model


def test_empty_object_is_the_plain_logistic_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    """``FINTFM_PROBE={}`` is a bare ``C=1`` logistic regression on the representation."""
    model = _fitted(monkeypatch, "{}")
    assert type(model._linear_probe) is LogisticRegression
    assert model._linear_probe.n_features_in_ == 2


def test_unset_takes_the_entry_default_for_the_problem_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``FINTFM_PROBE`` the multiclass entry's own readout applies (scale, cv and raw)."""
    model = _fitted(monkeypatch, None)
    assert model._probe_opts == {"scale": True, "cv": True, "raw": True}
    assert isinstance(model._linear_probe, Pipeline)
    assert model._linear_probe.n_features_in_ == 5


def test_probe_hyperparameter_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``probe_multiclass`` is used over ``FINTFM_PROBE``."""
    model = FinTFMLinearProbeModel(
        path="",
        name="fintfm_probe_opts",
        problem_type="multiclass",
        eval_metric="log_loss",
        hyperparameters={"probe_crossfit_folds": 0, "probe_multiclass": {"scale": True}},
    )
    monkeypatch.setenv("FINTFM_PROBE", '{"raw": true}')
    model._init_params()
    assert model._resolve_entry(model._get_model_params())["probe"] == {"scale": True}


def test_raw_appends_the_features_and_scale_wraps_a_scaler(monkeypatch: pytest.MonkeyPatch) -> None:
    """``raw`` widens the probe's input by the three features, missing values included."""
    model = _fitted(monkeypatch, '{"scale": true, "cv": true, "raw": true}')
    assert isinstance(model._linear_probe, Pipeline)
    assert model._linear_probe.n_features_in_ == 5
    X = np.random.default_rng(1).normal(size=(10, 3)).astype(np.float32)
    X[0, 2] = np.nan
    proba = model._linear_probe.predict_proba(model._probe_input(model.model, X))
    assert np.isfinite(proba).all()


def test_head_appends_the_head_log_probabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    """``head`` widens the probe's input by one column per class."""
    model = _fitted(monkeypatch, '{"scale": true, "head": true}')
    assert model._linear_probe.n_features_in_ == 5


def test_unknown_key_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A misspelled option fails the fit rather than running the default readout."""
    with pytest.raises(ValueError, match="unknown key"):
        _fitted(monkeypatch, '{"sclae": true}')


def _probe_for(y: np.ndarray, opts: dict):
    """The unfitted readout the model builds for labels ``y`` under probe options ``opts``."""
    model = FinTFMLinearProbeModel(path="", name="fintfm_probe_cv", problem_type="multiclass", eval_metric="log_loss")
    model._probe_opts = opts
    return model._new_classifier_probe(y)


def test_cv_keeps_three_folds_when_every_class_has_three_rows() -> None:
    """On data with at least three rows per class the cross-validated readout is as configured."""
    probe = _probe_for(np.array([0, 0, 0, 1, 1, 1, 1]), {"cv": True})
    assert isinstance(probe, LogisticRegressionCV)
    assert probe.cv == 3


def test_cv_folds_shrink_to_the_rarest_class() -> None:
    """A class with two rows leaves room for two stratified folds, not three."""
    probe = _probe_for(np.array([0, 0, 0, 0, 1, 1]), {"cv": True})
    assert isinstance(probe, LogisticRegressionCV)
    assert probe.cv == 2


def test_cv_falls_back_to_the_fixed_readout_with_a_single_row_class() -> None:
    """With a one-row class there is nothing to select ``C`` on, so the readout is plain and fits."""
    y = np.array([0, 0, 0, 1])
    probe = _probe_for(y, {"cv": True, "scale": True})
    assert isinstance(probe, Pipeline)
    assert type(probe[-1]) is LogisticRegression
    probe.fit(np.arange(8, dtype=float).reshape(4, 2), y)
    assert probe.predict_proba(np.zeros((1, 2))).shape == (1, 2)
