"""Reference :class:`~tabarena.simulation.ensemble.AbstractEnsembler` implementations.

These are small, useful-in-practice methods that double as templates for plugging custom
post-hoc ensembling into the simulation: pass one via ``ensembler_cls`` (plus
``ensembler_kwargs``) to :class:`~tabarena.simulation.ensemble_selection_config_scorer.EnsembleScorer`
or ``repo.evaluate_ensemble(ensemble_kwargs={"ensembler_cls": ..., "ensembler_kwargs": ...})``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np

from tabarena.simulation.ensemble.abstract_ensembler import AbstractEnsembler, WeightedEnsembler

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer


class SingleBestEnsembler(WeightedEnsembler):
    """Selects the single model with the best validation metric error (no ensembling)."""

    def _fit(self, *, predictions: np.ndarray, labels: np.ndarray, time_limit: float | None = None) -> None:
        errors = [self._score_error(labels, pred) for pred in predictions]
        self.best_index_ = int(np.nanargmin(errors))
        weights = np.zeros(len(predictions))
        weights[self.best_index_] = 1.0
        self.weights_ = weights

    def info(self) -> dict:
        return {"best_index": self.best_index_}


class TopKAverageEnsembler(WeightedEnsembler):
    """Uniform average of the ``k`` models with the best validation metric error.

    Parameters
    ----------
    k : int, default 5
        Number of models to average; capped at the number of available models.
    """

    def __init__(self, *, problem_type: str, metric: Scorer, k: int = 5):
        super().__init__(problem_type=problem_type, metric=metric)
        assert k >= 1
        self.k = k

    def _fit(self, *, predictions: np.ndarray, labels: np.ndarray, time_limit: float | None = None) -> None:
        errors = np.array([self._score_error(labels, pred) for pred in predictions])
        k = min(self.k, len(predictions))
        top_k = np.argsort(errors, kind="stable")[:k]
        weights = np.zeros(len(predictions))
        weights[top_k] = 1.0 / k
        self.weights_ = weights


class FixedWeightsEnsembler(WeightedEnsembler):
    """Applies user-provided per-model weights; nothing is fitted.

    Parameters
    ----------
    weights : array-like of float
        One weight per model, in the model order the simulation passes predictions in.
    """

    def __init__(self, *, problem_type: str, metric: Scorer, weights):
        super().__init__(problem_type=problem_type, metric=metric)
        self.weights_ = np.asarray(weights, dtype=np.float64)

    def _fit(self, *, predictions: np.ndarray, labels: np.ndarray, time_limit: float | None = None) -> None:
        if len(self.weights_) != len(predictions):
            raise ValueError(f"FixedWeightsEnsembler got {len(self.weights_)} weights for {len(predictions)} models")


class MedianEnsembler(AbstractEnsembler):
    """Applies the median for every prediction; nothing is fitted.

    The medians are normalized to produce probabilities in the multiclass case.
    """

    @override
    def _fit(self, *, predictions: np.ndarray, labels: np.ndarray, time_limit: float | None = None) -> None:
        if predictions.ndim < 3 and self.problem_type == "multiclass":
            raise ValueError(
                f"{self.__class__.__name__} has predictions with only {predictions.ndim} dimensions when there should be 3 for multiclass classification, to allow normalization. It is likely wrong classes were wrongly extracted before predicting."
            )

    @override
    def predict_proba(self, predictions: np.ndarray) -> np.ndarray:
        median_predictions = np.median(predictions, axis=0)
        if self.problem_type in ["binary", "regression"]:
            # Normalization unnecessary for binary as the other class median corresponds to the same sample.
            return median_predictions
        if predictions.ndim != 3:
            raise ValueError(
                f"Expected predictions with 3 dimensions (n_models, n_samples, n_classes) in the multiclass case, got {predictions.ndim} dimensions."
            )
        # Normalization for multiclass
        return median_predictions / median_predictions.sum(axis=1, keepdims=True)


class HardVotingEnsembler(AbstractEnsembler):
    """Applies a vote between different classifiers; nothing is fitted.

    It only works with classification, not regression. Class probabilities are estimated through the probability that a base learner would vote for a class.
    """

    def __init__(self, *, problem_type: str, metric: Scorer):
        if problem_type not in ["binary", "multiclass"]:
            raise ValueError(
                f"{self.__class__.__name__} only works with classification problems, not with {problem_type}."
            )
        super().__init__(problem_type=problem_type, metric=metric)

    @override
    def _fit(self, *, predictions: np.ndarray, labels: np.ndarray, time_limit: float | None = None) -> None:
        if predictions.ndim < 3 and self.problem_type == "multiclass":
            raise ValueError(
                f"{self.__class__.__name__} has predictions with only {predictions.ndim} dimensions when there should be 3 for multiclass classification, to allow a vote. It is likely wrong classes were wrongly extracted before predicting."
            )

    @override
    def predict_proba(self, predictions: np.ndarray) -> np.ndarray:
        if self.problem_type == "binary":
            return ((predictions > 0.5) + 0.5 * (predictions == 0.5)).mean(axis=0)
        # Binary matrix with 1 for the highest predicted probability, 0 otherwise.
        is_max = predictions == np.max(predictions, axis=2, keepdims=True)
        # For equalities, the vote is divided among all concerned classes.
        is_max = (is_max / is_max.sum(axis=2, keepdims=True)).astype(np.float32)
        # Proportion of vote used as probability estimation.
        return is_max.mean(axis=0)
