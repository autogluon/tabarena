"""sklearn-style meta-model adapters around AutoGluon ``AbstractModel`` classes.

Lets :class:`~tabarena.simulation.ensemble.StackingEnsembler` stack with any AutoGluon
model (LightGBM, CatBoost, RealMLP, ...) instead of an sklearn estimator::

    StackingEnsembler(
        problem_type=..., metric=...,
        classifier_cls=AutoGluonStackerClassifier,
        classifier_kwargs={"model_cls": LGBModel, "hyperparameters": {"n_estimators": 100}},
        regressor_cls=AutoGluonStackerRegressor,
        regressor_kwargs={"model_cls": LGBModel},
    )

The adapters bridge the interface differences: numpy features become named DataFrames,
labels become Series, the classifier tracks ``classes_`` and always returns sklearn-style
2-D probabilities (AutoGluon returns the 1-D positive-class vector for binary), and the
classification problem type is inferred from the fit labels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from autogluon.core.models import AbstractModel


class _AutoGluonStackerBase:
    def __init__(
        self,
        *,
        model_cls: type[AbstractModel],
        hyperparameters: dict | None = None,
        eval_metric: str | None = None,
        model_kwargs: dict | None = None,
    ):
        self.model_cls = model_cls
        self.hyperparameters = hyperparameters
        self.eval_metric = eval_metric
        self.model_kwargs = model_kwargs
        self._model: AbstractModel | None = None
        self._n_features: int | None = None

    def _to_frame(self, X) -> pd.DataFrame:
        X = np.asarray(X)
        if self._n_features is None:
            self._n_features = X.shape[1]
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    def _fit_model(self, X, y, problem_type: str) -> None:
        model = self.model_cls(
            problem_type=problem_type,
            eval_metric=self.eval_metric,
            hyperparameters=dict(self.hyperparameters) if self.hyperparameters else None,
            **(self.model_kwargs or {}),
        )
        model.fit(X=self._to_frame(X), y=pd.Series(np.asarray(y)))
        self._model = model

    def predict(self, X) -> np.ndarray:
        return np.asarray(self._model.predict(self._to_frame(X)))


class AutoGluonStackerClassifier(_AutoGluonStackerBase):
    """Classification adapter: infers binary/multiclass from the fit labels, exposes
    ``classes_``, and returns sklearn-style ``(n, n_classes)`` probabilities.
    """

    classes_: np.ndarray

    def fit(self, X, y) -> AutoGluonStackerClassifier:
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        problem_type = "binary" if len(self.classes_) == 2 else "multiclass"
        # AbstractModel expects contiguous 0..k-1 labels; map and remember the order
        y_encoded = np.searchsorted(self.classes_, y)
        self._fit_model(X, y_encoded, problem_type=problem_type)
        return self

    def predict_proba(self, X) -> np.ndarray:
        proba = np.asarray(self._model.predict_proba(self._to_frame(X)))
        if proba.ndim == 1:  # AutoGluon binary: positive-class vector -> sklearn 2-D
            proba = np.column_stack([1.0 - proba, proba])
        return proba

    def predict(self, X) -> np.ndarray:
        encoded = super().predict(X)
        return self.classes_[np.asarray(encoded, dtype=int)]


class AutoGluonStackerRegressor(_AutoGluonStackerBase):
    """Regression adapter."""

    def fit(self, X, y) -> AutoGluonStackerRegressor:
        self._fit_model(X, y, problem_type="regression")
        return self
