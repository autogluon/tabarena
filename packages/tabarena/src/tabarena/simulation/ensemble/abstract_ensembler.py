"""The ensembler interface used by TabArena's ensemble simulation.

An *ensembler* combines the predictions of already-fitted base models on a single task.
It is the swappable unit of :class:`~tabarena.simulation.ensemble_selection_config_scorer.EnsembleScorer`:
implement :class:`AbstractEnsembler` (or, for weighted combinations,
:class:`WeightedEnsembler`) and pass the class via ``ensembler_cls`` to plug an
alternative post-hoc ensembling method into the simulation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from autogluon.core.utils import get_pred_from_proba

if TYPE_CHECKING:
    from typing import Self

    from autogluon.core.metrics import Scorer


class AbstractEnsembler(ABC):
    """Fits an ensemble over base-model *predictions* (not features) for one task.

    Data format
    -----------
    ``predictions`` is an array-like of per-model predictions, shape
    ``(n_models, n_samples)`` or ``(n_models, n_samples, n_classes)``, in the space the
    fit ``metric`` consumes. Note that this space can be a *metric-preprocessed view* of
    the original problem: e.g. for log_loss the simulation feeds ground-truth-class
    probabilities (1-D per model) with ``problem_type="binary"``. Such transformed views
    are only valid to combine linearly, which is what the :attr:`linear` flag declares.

    Lifecycle
    ---------
    Construct with ``(problem_type, metric, **method_kwargs)``, then ``fit`` once on the
    optimization split and ``predict``/``predict_proba`` on any split. ``model_weights``
    / ``models_used`` / ``info`` report the fitted ensemble back to the simulation (they
    drive the ``ensemble_weights`` result column and the inference-time accounting).

    Parameters
    ----------
    problem_type : str
        The problem type the predictions are in (after any metric preprocessing).
    metric : Scorer
        The metric to optimize, in AutoGluon ``Scorer`` format. ``metric.error`` is the
        lower-is-better objective.
    """

    linear: bool = True
    """Whether :meth:`predict_proba` is a fixed linear combination of the input
    predictions. Non-linear ensemblers (e.g. stacking) must set this to ``False``; they
    are rejected when the simulation feeds metric-preprocessed prediction spaces, where
    only linear combinations are meaningful."""

    def __init__(self, *, problem_type: str, metric: Scorer):
        self.problem_type = problem_type
        self.metric = metric
        self.n_models_: int | None = None

    def fit(self, *, predictions: np.ndarray, labels: np.ndarray, time_limit: float | None = None) -> Self:
        """Fit the ensemble on the optimization split. Returns ``self``."""
        self.n_models_ = len(predictions)
        self._fit(predictions=predictions, labels=labels, time_limit=time_limit)
        return self

    @abstractmethod
    def _fit(self, *, predictions: np.ndarray, labels: np.ndarray, time_limit: float | None = None) -> None: ...

    @abstractmethod
    def predict_proba(self, predictions: np.ndarray) -> np.ndarray:
        """Combined prediction, in the same space as the per-model inputs."""
        ...

    def predict(self, predictions: np.ndarray, *, problem_type: str | None = None) -> np.ndarray:
        """Combined prediction converted to label space.

        ``problem_type`` overrides the fit-time problem type: the caller may have fitted
        on a metric-preprocessed view (e.g. ``"binary"`` ground-truth-class
        probabilities) but need label-space predictions for the original problem.
        """
        y_pred_proba = self.predict_proba(predictions)
        if problem_type is None:
            problem_type = self.problem_type
        return get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=problem_type)

    # -------------------------
    # Reporting surface
    # -------------------------
    def model_weights(self) -> np.ndarray | None:
        """Per-model linear weights of the fitted ensemble, or ``None`` if the method is
        not weight-based. Drives the ``ensemble_weights`` result column.
        """
        return None

    def models_used(self) -> np.ndarray:
        """Boolean mask of models required at inference time, aligned with the fit-time
        model order. Drives the ensemble's ``time_infer_s`` accounting. Defaults to
        ``model_weights() != 0``; ensemblers without weights report all models used and
        should override when they can do better.
        """
        weights = self.model_weights()
        if weights is None:
            return np.ones(self.n_models_, dtype=bool)
        return np.asarray(weights) != 0

    def info(self) -> dict:
        """Optional extra payload describing the fitted ensemble (surfaced as
        ``ensemble_info`` in task results when non-empty).
        """
        return {}

    # -------------------------
    # Helpers for subclasses
    # -------------------------
    def _score_error(self, labels: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """The metric error (lower is better) of a combined prediction, converting to
        label space first when the metric requires class predictions.
        """
        metric = self.metric
        if metric.needs_pred or metric.needs_quantile:
            y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
            return metric.error(labels, y_pred)
        return metric.error(labels, y_pred_proba)


class WeightedEnsembler(AbstractEnsembler):
    """Base class for ensemblers whose prediction is a fixed weighted sum.

    Subclasses set ``self.weights_`` (one weight per model) in ``_fit``.
    """

    weights_: np.ndarray

    def predict_proba(self, predictions: np.ndarray) -> np.ndarray:
        # Identical arithmetic to AutoGluon's AbstractWeightedEnsemble.weight_pred_probas,
        # so results are bit-for-bit interchangeable with the historical implementation.
        preds_norm = [pred * weight for pred, weight in zip(predictions, self.weights_, strict=False) if weight != 0]
        return np.sum(preds_norm, axis=0)

    def model_weights(self) -> np.ndarray | None:
        return np.asarray(self.weights_)


class LegacyEnsemblerAdapter(AbstractEnsembler):
    """Adapts a pre-interface ensemble class (AutoGluon ``AbstractWeightedEnsemble``
    style: ``cls(problem_type=..., metric=..., **kwargs)`` with ``fit(predictions,
    labels)``, ``predict``/``predict_proba`` reading ``self.problem_type``, and an
    optional ``weights_`` attribute) to :class:`AbstractEnsembler`.

    Kept so ``ensemble_method``-style callers (deprecated) continue to work unchanged.
    """

    def __init__(self, *, problem_type: str, metric: Scorer, ensemble_cls: type, ensemble_kwargs: dict | None = None):
        super().__init__(problem_type=problem_type, metric=metric)
        self._ensemble = ensemble_cls(problem_type=problem_type, metric=metric, **(ensemble_kwargs or {}))

    def _fit(self, *, predictions: np.ndarray, labels: np.ndarray, time_limit: float | None = None) -> None:
        # list() guard: AutoGluon's EnsembleSelection._fit mutates the sequence in place
        # when subsampling is triggered.
        self._ensemble.fit(predictions=list(predictions), labels=labels)

    def predict_proba(self, predictions: np.ndarray) -> np.ndarray:
        return self._ensemble.predict_proba(predictions)

    def predict(self, predictions: np.ndarray, *, problem_type: str | None = None) -> np.ndarray:
        # Legacy classes read self.problem_type inside predict; override it for the call.
        if problem_type is None or problem_type == self._ensemble.problem_type:
            return self._ensemble.predict(predictions)
        original_problem_type = self._ensemble.problem_type
        self._ensemble.problem_type = problem_type
        try:
            return self._ensemble.predict(predictions)
        finally:
            self._ensemble.problem_type = original_problem_type

    def model_weights(self) -> np.ndarray | None:
        weights = getattr(self._ensemble, "weights_", None)
        if weights is None:
            return None
        return np.asarray(weights)
