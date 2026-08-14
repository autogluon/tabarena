"""Kaggle-style hill-climbing ensemble (Matt-OP / Deotte family).

Same approach family as :class:`GreedyEnsembler` (Caruana GES / AG ``EnsembleSelection``):
greedy local search over linear combinations of OOF predictions. Differences are the step
parameterization (continuous ``w`` grid vs discrete multiset votes), not a different problem.

Each step evaluates all ``(model, w)`` candidates and accepts the single best improvement.
Optionally includes the Caruana step weight ``w = 1/(n_support+1)`` on the grid.

This class is optional research / post-hoc; TabArena's default remains
:class:`GreedyEnsembler`.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from tabarena.simulation.ensemble.abstract_ensembler import WeightedEnsembler

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer


class HillClimbingEnsembler(WeightedEnsembler):
    """Iterative convex blending of base-model predictions (hill climbing).

    Starts from the best single model (or optional warm-start weights), then repeatedly
    blends ``(1 - w) * ensemble + w * model`` and keeps the best improving step.

    Parameters
    ----------
    precision : float, default 0.01
        Weight step on ``(0, 1]``. Smaller is slower and can overfit small OOF sets.
    max_rounds : int, default 100
        Maximum accepted improvement steps. Stops early when a step finds no improvement.
    include_caruana_step : bool, default True
        Also try ``w = 1/(n_support + 1)`` each step (GES-style blend weight when treating
        the current ensemble as one unit).
    allow_negative_weights : bool, default False
        If True, also tries ``w`` on ``[-1, 0)``. Safer off on small OOF.
    max_models : int | None, default None
        Optional cap on how many models may receive non-zero weight. ``None`` means no cap.
    initial_weights : array-like | None, default None
        Optional warm-start weights (one entry per model). When set, skip best-single
        initialization. Must be finite; non-negative unless ``allow_negative_weights``;
        renormalized to sum to 1.
    random_state : int | np.random.RandomState | None, default None
        Tie-breaking when selecting the initial best single model.
    """

    def __init__(
        self,
        *,
        problem_type: str,
        metric: Scorer,
        precision: float = 0.01,
        max_rounds: int = 100,
        include_caruana_step: bool = True,
        allow_negative_weights: bool = False,
        max_models: int | None = None,
        initial_weights: np.ndarray | list[float] | None = None,
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
        self.include_caruana_step = bool(include_caruana_step)
        self.allow_negative_weights = bool(allow_negative_weights)
        self.max_models = max_models
        self.initial_weights = None if initial_weights is None else np.asarray(initial_weights, dtype=np.float64)
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(0 if random_state is None else int(random_state))

        self.trajectory_: list[float] = []
        self.n_rounds_: int = 0
        self.init_val_error_: float | None = None

    def _weight_grid(self, *, n_support: int | None = None) -> np.ndarray:
        pos = np.arange(self.precision, 1.0 + self.precision * 0.5, self.precision)
        pos = np.clip(pos, self.precision, 1.0)
        pos = np.unique(np.round(pos / self.precision) * self.precision)
        if self.include_caruana_step and n_support is not None and n_support >= 1:
            w_c = 1.0 / float(n_support + 1)
            if 0 < w_c <= 1:
                pos = np.unique(np.concatenate([pos, np.asarray([w_c], dtype=np.float64)]))
        if self.allow_negative_weights:
            return np.concatenate([-pos[::-1], pos])
        return pos

    def _allowed_model(self, j: int, w: float, weights: np.ndarray) -> bool:
        if self.max_models is None or weights[j] != 0 or w == 0:
            return True
        return self._n_nonzero(weights) < self.max_models

    def _try_blend(
        self,
        *,
        ensemble_pred: np.ndarray,
        pred_j: np.ndarray,
        w: float,
        labels: np.ndarray,
        best_error: float,
    ) -> tuple[float, float, np.ndarray] | None:
        # The running ensemble stays the raw linear blend so `weights_` describe it
        # exactly (negative weights can push rows off the simplex, and clamping the
        # running state would silently divorce the tracked weights from the scored
        # ensemble). Only the copy that is scored gets renormalized, matching what
        # `predict_proba` applies to the final linear combination.
        trial = (1.0 - w) * ensemble_pred + w * pred_j
        err = self._score_error(labels, self._renormalize_proba(trial))
        if err < best_error - 1e-15:
            return err, float(w), trial
        return None

    def _combine(self, predictions: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
        preds_norm = [pred * w for pred, w in zip(predictions, weights, strict=True) if w != 0]
        if not preds_norm:
            return np.mean(predictions, axis=0)
        return np.sum(preds_norm, axis=0)

    def _n_nonzero(self, weights: np.ndarray) -> int:
        return int(np.sum(np.abs(weights) > 0))

    def _renormalize_proba(self, trial: np.ndarray) -> np.ndarray:
        """Keep multiclass / softclass rows on the probability simplex."""
        if trial.ndim == 2 and self.problem_type in ("multiclass", "softclass"):
            row_sum = trial.sum(axis=1, keepdims=True)
            row_sum = np.where(row_sum == 0, 1.0, row_sum)
            trial = trial / row_sum
            np.maximum(trial, 0.0, out=trial)
            row_sum = trial.sum(axis=1, keepdims=True)
            row_sum = np.where(row_sum == 0, 1.0, row_sum)
            trial = trial / row_sum
        return trial

    def _fit(self, *, predictions: np.ndarray, labels: np.ndarray, time_limit: float | None = None) -> None:
        start = time.time()
        # float64 for stable blends; renorm multiclass so float32 OOF stays on the simplex.
        predictions = [self._renormalize_proba(np.asarray(p, dtype=np.float64)) for p in predictions]
        n_models = len(predictions)
        if n_models == 0:
            raise ValueError("HillClimbingEnsembler requires at least one model")

        if self.initial_weights is not None:
            weights = np.asarray(self.initial_weights, dtype=np.float64).copy()
            if weights.shape != (n_models,):
                raise ValueError(f"initial_weights length {weights.shape} != n_models={n_models}")
            if not np.isfinite(weights).all():
                raise ValueError("initial_weights must be finite")
            if not self.allow_negative_weights:
                weights = np.maximum(weights, 0.0)
            total0 = float(np.sum(weights))
            if abs(total0) <= 1e-12:
                raise ValueError("initial_weights sum to ~0; cannot warm-start")
            weights = weights / total0
            ensemble_pred = self._combine(predictions, weights)
            best_error = float(self._score_error(labels, self._renormalize_proba(ensemble_pred)))
            start_idx = int(np.argmax(np.abs(weights)))
        else:
            single_errors = np.array([self._score_error(labels, pred) for pred in predictions], dtype=np.float64)
            best_error = float(np.nanmin(single_errors))
            candidates = np.flatnonzero(np.isclose(single_errors, best_error, atol=0, rtol=1e-12))
            start_idx = int(self.random_state.choice(candidates))
            weights = np.zeros(n_models, dtype=np.float64)
            weights[start_idx] = 1.0
            ensemble_pred = predictions[start_idx].copy()

        self.init_val_error_ = best_error
        self.trajectory_ = [best_error]

        for round_i in range(self.max_rounds):
            if time_limit is not None and (time.time() - start) >= time_limit:
                break

            n_support = max(self._n_nonzero(weights), 1)
            weight_grid = self._weight_grid(n_support=n_support)
            best_local_error = best_error
            best_j = None
            best_local_w = None
            best_local_pred = None
            for j in range(n_models):
                if time_limit is not None and (time.time() - start) >= time_limit:
                    break
                pred_j = predictions[j]
                for w in weight_grid:
                    if not self._allowed_model(j, float(w), weights):
                        continue
                    hit = self._try_blend(
                        ensemble_pred=ensemble_pred,
                        pred_j=pred_j,
                        w=float(w),
                        labels=labels,
                        best_error=best_local_error,
                    )
                    if hit is not None:
                        best_local_error, best_local_w, best_local_pred = hit
                        best_j = j

            self.n_rounds_ = round_i + 1
            if best_j is None or best_local_w is None:
                break

            weights *= 1.0 - best_local_w
            weights[best_j] += best_local_w
            ensemble_pred = best_local_pred
            best_error = best_local_error
            self.trajectory_.append(best_error)

        weights[np.abs(weights) < 1e-12] = 0.0
        if not self.allow_negative_weights:
            weights = np.maximum(weights, 0.0)
        total = float(np.sum(weights))
        if abs(total) > 1e-12:
            weights = weights / total
        else:
            weights = np.zeros(n_models, dtype=np.float64)
            weights[start_idx] = 1.0

        self.weights_ = weights

    def predict_proba(self, predictions: np.ndarray) -> np.ndarray:
        out = super().predict_proba(predictions)
        return self._renormalize_proba(np.asarray(out, dtype=np.float64))

    def info(self) -> dict:
        return {
            "n_rounds": self.n_rounds_,
            "trajectory_len": len(self.trajectory_),
            "init_val_error": self.init_val_error_,
            "final_val_error": self.trajectory_[-1] if self.trajectory_ else None,
            "warm_started": self.initial_weights is not None,
            "include_caruana_step": self.include_caruana_step,
            "precision": self.precision,
            "allow_negative_weights": self.allow_negative_weights,
        }
