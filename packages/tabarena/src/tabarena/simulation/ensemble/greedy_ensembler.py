from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.simulation.ensemble.abstract_ensembler import WeightedEnsembler

if TYPE_CHECKING:
    import numpy as np
    from autogluon.core.metrics import Scorer


class GreedyEnsembler(WeightedEnsembler):
    """Greedy weighted ensemble selection (Caruana et al., 2004) — the default TabArena
    post-hoc ensembling method.

    Thin adapter around AutoGluon's
    :class:`~autogluon.core.models.greedy_ensemble.ensemble_selection.EnsembleSelection`,
    which performs the actual selection; results are identical to using that class
    directly.

    Parameters
    ----------
    ensemble_size : int, default 100
        Maximum number of greedy selection iterations.
    **ensemble_selection_kwargs
        Forwarded to ``EnsembleSelection`` (e.g. ``tie_breaker``, ``subsample_size``,
        ``random_state``).
    """

    def __init__(self, *, problem_type: str, metric: Scorer, ensemble_size: int = 100, **ensemble_selection_kwargs):
        super().__init__(problem_type=problem_type, metric=metric)
        self.ensemble_size = ensemble_size
        self._ensemble_selection_kwargs = ensemble_selection_kwargs
        self._selection = None

    def _fit(self, *, predictions: np.ndarray, labels: np.ndarray, time_limit: float | None = None) -> None:
        from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection

        self._selection = EnsembleSelection(
            ensemble_size=self.ensemble_size,
            problem_type=self.problem_type,
            metric=self.metric,
            **self._ensemble_selection_kwargs,
        )
        # list() guard: EnsembleSelection._fit mutates the sequence in place when
        # subsampling is triggered.
        self._selection.fit(predictions=list(predictions), labels=labels, time_limit=time_limit)
        self.weights_ = self._selection.weights_
