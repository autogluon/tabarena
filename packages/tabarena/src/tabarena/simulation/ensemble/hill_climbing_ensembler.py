"""Kaggle-style hill-climbing ensemble (Matt-OP / Deotte family).

This is the definition of "hill climbing" that matches tilii7's AutoGluon feedback and
the community pointer on autogluon/autogluon#4505
(https://github.com/Matt-OP/hillclimbers/) — **not** continuous black-box HPO
(arxiv:2307.00286).

Compared to :class:`GreedyEnsembler` (Caruana ensemble selection, arxiv:1502.04759 /
AutoGluon ``EnsembleSelection``):

* **Caruana / GreedyEnsembler:** iteratively *append* a model so the uniform average of
  the multiset improves; weights are integer counts / ensemble_size.
* **Hill climbing (this class):** start from the best single model, then repeatedly try
  convex blends ``(1 - w) * ensemble + w * model`` over a weight grid; accept any
  improvement. Weights are continuous on a precision grid (default ``0.01``).

Both are greedy local search over linear combinations of OOF predictions; they can
diverge on the selected support and weight magnitudes. #4505 asks whether this family
improves TabRepo/TabArena simulation score vs current ensemble selection — plug this
class in via ``ensembler_cls`` / ``ensemble_kwargs``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from tabarena.simulation.ensemble.abstract_ensembler import WeightedEnsembler

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer


class HillClimbingEnsembler(WeightedEnsembler):
    """Iterative convex blending of base-model predictions (Kaggle hill climbing).

    Parameters
    ----------
    precision : float, default 0.01
        Weight step on ``(0, 1]``. Smaller is slower and can overfit small OOF sets.
    max_rounds : int, default 100
        Maximum outer passes over all models after initialization. Stops early when a
        full pass finds no improvement.
    allow_negative_weights : bool, default False
        If True, also tries ``w`` on ``[-1, 0)`` (Matt-OP ``negative_weights``). Safer
        off when OOF is small.
    max_models : int | None, default None
        Optional cap on how many models may receive non-zero weight (sparsity preference).
        ``None`` means no extra cap.
    random_state : int | np.random.RandomState | None, default None
        Used only for tie-breaking among equal-error starts / blends.
    """

    def __init__(
        self,
        *,
        problem_type: str,
        metric: Scorer,
        precision: float = 0.01,
        max_rounds: int = 100,
        allow_negative_weights: bool = False,
        max_models: int | None = None,
        random_state: int | np.random.RandomState | None = None,
    ):
        super().__init__(problem_type=problem_type, metric=metric)
        if precision <= 0 or precision > 1:
            raise ValueError(f"precision must be in (0, 1], got {precision}")
        if max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {max_rounds}")
        if max_models is not None and max_models < 1:
            raise ValueError(f"max_models must be >= 1 or None, got {max_models}")
        self.precision = float(precision)
        self.max_rounds = int(max_rounds)
        self.allow_negative_weights = bool(allow_negative_weights)
        self.max_models = max_models
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(0 if random_state is None else int(random_state))

        self.trajectory_: list[float] = []
        self.n_rounds_: int = 0

    def _weight_grid(self) -> np.ndarray:
        # Exclude 0 (no-op). Include 1.0.
        pos = np.arange(self.precision, 1.0 + self.precision * 0.5, self.precision)
        pos = np.clip(pos, self.precision, 1.0)
        # unique stable
        pos = np.unique(np.round(pos / self.precision) * self.precision)
        if self.allow_negative_weights:
            neg = -pos[::-1]
            return np.concatenate([neg, pos])
        return pos

    def _combine(self, predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
        # Same linear combo semantics as WeightedEnsembler.predict_proba / AG.
        preds_norm = [pred * w for pred, w in zip(predictions, weights, strict=True) if w != 0]
        if not preds_norm:
            # Degenerate: fall back to uniform over all (should not happen after init).
            return np.mean(predictions, axis=0)
        return np.sum(preds_norm, axis=0)

    def _n_nonzero(self, weights: np.ndarray) -> int:
        return int(np.sum(np.abs(weights) > 0))

    def _fit(self, *, predictions: np.ndarray, labels: np.ndarray, time_limit: float | None = None) -> None:
        start = time.time()
        predictions = np.asarray(predictions)
        n_models = len(predictions)
        if n_models == 0:
            raise ValueError("HillClimbingEnsembler requires at least one model")

        # --- Initialize with best single model ---
        single_errors = np.array([self._score_error(labels, pred) for pred in predictions], dtype=np.float64)
        best_error = float(np.nanmin(single_errors))
        candidates = np.flatnonzero(np.isclose(single_errors, best_error, atol=0, rtol=1e-12))
        start_idx = int(self.random_state.choice(candidates))

        weights = np.zeros(n_models, dtype=np.float64)
        weights[start_idx] = 1.0
        ensemble_pred = predictions[start_idx].copy()
        self.trajectory_ = [best_error]

        weight_grid = self._weight_grid()

        for round_i in range(self.max_rounds):
            if time_limit is not None and (time.time() - start) >= time_limit:
                break

            improved = False
            # Randomize model order each round for mild exploration under ties.
            order = self.random_state.permutation(n_models)
            for j in order:
                if time_limit is not None and (time.time() - start) >= time_limit:
                    break

                pred_j = predictions[j]
                best_local_error = best_error
                best_local_w = None
                best_local_pred = None

                for w in weight_grid:
                    # Convex blend against current ensemble (standard Kaggle HC step).
                    trial = (1.0 - w) * ensemble_pred + w * pred_j
                    # Optional multiclass renormalize if this is a probability simplex view.
                    if trial.ndim == 2 and self.problem_type in ("multiclass", "softclass"):
                        row_sum = trial.sum(axis=1, keepdims=True)
                        row_sum = np.where(row_sum == 0, 1.0, row_sum)
                        trial = trial / row_sum

                    # Enforce max_models: if adding a new model would exceed cap, skip
                    # unless it already has weight.
                    if self.max_models is not None and weights[j] == 0 and w != 0:
                        if self._n_nonzero(weights) >= self.max_models:
                            continue

                    err = self._score_error(labels, trial)
                    if err < best_local_error - 1e-15:
                        best_local_error = err
                        best_local_w = float(w)
                        best_local_pred = trial

                if best_local_w is not None:
                    # Update latent weights: ensemble := (1-w)*ensemble + w*model_j
                    # ⇒ scale existing weights by (1-w), add w to model j.
                    weights *= 1.0 - best_local_w
                    weights[j] += best_local_w
                    ensemble_pred = best_local_pred
                    best_error = best_local_error
                    self.trajectory_.append(best_error)
                    improved = True

            self.n_rounds_ = round_i + 1
            if not improved:
                break

        # Numerical cleanup: drop tiny weights, renormalize for stable reporting.
        weights[np.abs(weights) < 1e-12] = 0.0
        if not self.allow_negative_weights:
            weights = np.maximum(weights, 0.0)
        total = float(np.sum(weights))
        if abs(total) > 1e-12:
            weights = weights / total
        else:
            # Should not happen after best-single init; fall back to that model.
            weights = np.zeros(n_models, dtype=np.float64)
            weights[start_idx] = 1.0

        self.weights_ = weights

    def info(self) -> dict:
        return {
            "n_rounds": self.n_rounds_,
            "trajectory_len": len(self.trajectory_),
            "final_val_error": self.trajectory_[-1] if self.trajectory_ else None,
            "precision": self.precision,
            "allow_negative_weights": self.allow_negative_weights,
        }
