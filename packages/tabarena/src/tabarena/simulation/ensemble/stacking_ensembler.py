from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tabarena.simulation.ensemble.abstract_ensembler import AbstractEnsembler

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer


class StackingEnsembler(AbstractEnsembler):
    """Cross-validated meta-learner stacking over base-model predictions.

    A meta-model is trained to predict the label from the concatenated per-model
    predictions (for multiclass: ``n_models * n_classes`` probability features). The
    meta-model is any sklearn-style estimator (``fit`` / ``predict_proba`` or
    ``predict``); by default multinomial :class:`~sklearn.linear_model.LogisticRegression`
    for classification and :class:`~sklearn.linear_model.Ridge` for regression. Pass
    e.g. ``classifier_cls=TabPFNClassifier`` to stack with a foundation model.

    Honest validation error via inner CV
    ------------------------------------
    Fitting runs a K-fold cross-validation over the optimization split: the final
    meta-model is refit on all rows (used for unseen data, e.g. the test split), while
    out-of-fold meta-predictions are cached and returned when predicting on the fit
    input itself. The simulation evaluates ``metric_error_val`` on the same rows the
    ensembler was fitted on, so without this the stacker would score its own training
    predictions; with it, the reported validation error is an out-of-sample estimate
    (the same val-OOF / test-full-fit split :class:`EnsembleScorerCalibratedCV` uses
    for calibrators). The fit input is recognized by object identity; any other input
    uses the full refit model.

    Note this ensembler is non-linear (``linear = False``): it cannot consume
    metric-preprocessed prediction spaces, so run it with ``use_fast_metrics=False``.

    Parameters
    ----------
    classifier_cls, regressor_cls : type, optional
        sklearn-style meta-model class per problem kind. Defaults to
        ``LogisticRegression`` / ``Ridge``.
    classifier_kwargs, regressor_kwargs : dict, optional
        Constructor kwargs for the meta-model.
    n_splits : int, default 5
        Inner CV folds for the out-of-fold validation predictions. Reduced when the
        data cannot support it (few rows, or classes rarer than ``n_splits``); with
        fewer than 2 usable splits the CV step is skipped and validation predictions
        fall back to in-sample.
    random_state : int, default 0
        Seed for the CV split shuffling.
    """

    linear = False

    def __init__(
        self,
        *,
        problem_type: str,
        metric: Scorer,
        classifier_cls: type | None = None,
        classifier_kwargs: dict | None = None,
        regressor_cls: type | None = None,
        regressor_kwargs: dict | None = None,
        n_splits: int = 5,
        random_state: int = 0,
    ):
        super().__init__(problem_type=problem_type, metric=metric)
        assert n_splits >= 2
        self.classifier_cls = classifier_cls
        self.classifier_kwargs = classifier_kwargs
        self.regressor_cls = regressor_cls
        self.regressor_kwargs = regressor_kwargs
        self.n_splits = n_splits
        self.random_state = random_state
        self._n_classes: int | None = None

    @property
    def _is_classification(self) -> bool:
        return self.problem_type in ("binary", "multiclass")

    def _make_model(self):
        if self._is_classification:
            if self.classifier_cls is not None:
                return self.classifier_cls(**(self.classifier_kwargs or {}))
            from sklearn.linear_model import LogisticRegression

            kwargs = dict(self.classifier_kwargs or {})
            kwargs.setdefault("max_iter", 1000)
            return LogisticRegression(**kwargs)
        if self.regressor_cls is not None:
            return self.regressor_cls(**(self.regressor_kwargs or {}))
        from sklearn.linear_model import Ridge

        return Ridge(**(self.regressor_kwargs or {}))

    @staticmethod
    def _to_features(predictions) -> np.ndarray:
        arr = np.asarray(predictions)
        if arr.ndim == 2:  # (n_models, n_samples): binary pos-class proba or regression preds
            return arr.T
        # (n_models, n_samples, n_classes) -> (n_samples, n_models * n_classes)
        return arr.transpose(1, 0, 2).reshape(arr.shape[1], -1)

    def _predict_with(self, model, X: np.ndarray) -> np.ndarray:
        """Meta-model prediction in the same space as the per-model inputs."""
        if not self._is_classification:
            return model.predict(X)
        proba = model.predict_proba(X)
        classes = np.asarray(model.classes_).astype(int)
        if len(classes) != self._n_classes:
            # the training rows missed some classes: place columns by fitted class labels
            full = np.zeros((proba.shape[0], self._n_classes))
            full[:, classes] = proba
            proba = full
        if self.problem_type == "binary":
            return proba[:, 1]
        return proba

    def _compute_oof(self, X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
        """Out-of-fold meta-predictions over the fit rows, or None when the data cannot
        support at least 2 CV splits.
        """
        from sklearn.model_selection import KFold, StratifiedKFold

        if self._is_classification:
            _, counts = np.unique(y, return_counts=True)
            n_splits = min(self.n_splits, int(counts.min()))
        else:
            n_splits = min(self.n_splits, len(y))
        self._n_splits_used = n_splits
        if n_splits < 2:
            return None

        splitter_cls = StratifiedKFold if self._is_classification else KFold
        splitter = splitter_cls(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        if self.problem_type == "multiclass":
            oof = np.zeros((len(y), self._n_classes))
        else:
            oof = np.zeros(len(y))
        for train_idx, holdout_idx in splitter.split(X, y):
            model = self._make_model()
            model.fit(X[train_idx], y[train_idx])
            oof[holdout_idx] = self._predict_with(model, X[holdout_idx])
        return oof

    def _fit(self, *, predictions, labels, time_limit=None) -> None:
        X = self._to_features(predictions)
        y = np.asarray(labels)
        if self._is_classification:
            arr = np.asarray(predictions)
            self._n_classes = arr.shape[2] if arr.ndim == 3 else 2

        self._oof_prediction = self._compute_oof(X, y)
        self._fit_predictions_ref = predictions

        self._model = self._make_model()
        self._model.fit(X, y)

    def predict_proba(self, predictions) -> np.ndarray:
        if self._oof_prediction is not None and predictions is self._fit_predictions_ref:
            return self._oof_prediction
        return self._predict_with(self._model, self._to_features(predictions))

    def info(self) -> dict:
        return {"n_splits_used": self._n_splits_used}
