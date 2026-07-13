"""Reference :class:`~tabarena.simulation.ensemble.AbstractEnsembler` implementations.

These are small, useful-in-practice methods that double as templates for plugging custom
post-hoc ensembling into the simulation: pass one via ``ensembler_cls`` (plus
``ensembler_kwargs``) to :class:`~tabarena.simulation.ensemble_selection_config_scorer.EnsembleScorer`
or ``repo.evaluate_ensemble(ensemble_kwargs={"ensembler_cls": ..., "ensembler_kwargs": ...})``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tabarena.simulation.ensemble.abstract_ensembler import WeightedEnsembler

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
