"""Ensemble scorer that makes post-hoc decisions on only part of the validation data.

:class:`EnsembleScorerValSubsample` mirrors :class:`EnsembleScorerMaxModels` but, before the weighted
ensemble is fit, it restricts the validation set. This simulates a practitioner who can only afford a
smaller validation set for post-hoc decisions (ensemble weighting, ``metric_error_val``); the
**test** evaluation is untouched, so ``metric_error`` still reflects the full test data — only the
*quality of the decisions made on validation* degrades.

Two ways to size the kept validation set (mutually exclusive):

* ``val_fraction`` — keep this fraction of the rows.
* ``max_val_samples`` — keep at most this many rows total: all of them if the validation set is
  already that small, otherwise exactly ``max_val_samples`` (proportionally across classes).

Subsampling is **stratified** for classification (every class keeps at least one row) so that metrics
which need all classes present stay computable (e.g. ``roc_auc`` needs both classes for binary and
every class for multiclass). For regression it is a plain random subsample. The draw is deterministic
per ``(dataset, fold)`` (seeded from ``val_subsample_seed``), so a run is reproducible while different
tasks get independent draws.
"""

from __future__ import annotations

import hashlib

import numpy as np

from tabarena.simulation.ensemble_selection_config_scorer import EnsembleScorerMaxModels


class EnsembleScorerValSubsample(EnsembleScorerMaxModels):
    """:class:`EnsembleScorerMaxModels` that fits the ensemble on a stratified subset of validation.

    Parameters
    ----------
    val_fraction: float, default = 1.0
        Fraction of validation rows to keep for post-hoc decisions, in ``(0, 1]``. ``1.0`` is a
        no-op (identical to :class:`EnsembleScorerMaxModels`). Ignored if ``max_val_samples`` is set.
    max_val_samples: int | None, default = None
        Absolute cap on validation rows: use all of them when the validation set has no more than
        ``max_val_samples`` rows, otherwise keep exactly ``max_val_samples`` (stratified). Takes
        precedence over ``val_fraction`` when set.
    val_subsample_seed: int, default = 0
        Base seed for the per-task draw. The actual seed is derived deterministically from
        ``(dataset, fold, val_subsample_seed)`` so each task is subsampled independently but
        reproducibly.
    """

    def __init__(
        self,
        *,
        val_fraction: float = 1.0,
        max_val_samples: int | None = None,
        val_subsample_seed: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 0.0 < val_fraction <= 1.0:
            raise ValueError(f"val_fraction must be in (0, 1], got {val_fraction}.")
        if max_val_samples is not None and max_val_samples < 1:
            raise ValueError(f"max_val_samples must be >= 1, got {max_val_samples}.")
        self.val_fraction = val_fraction
        self.max_val_samples = max_val_samples
        self.val_subsample_seed = val_subsample_seed

    def _is_active(self) -> bool:
        """Whether any subsampling happens (else this is a pass-through scorer)."""
        return self.max_val_samples is not None or self.val_fraction < 1.0

    def _task_seed(self, dataset: str, fold: int) -> int:
        """Deterministic per-task seed (stable across processes, unlike ``hash()``)."""
        digest = hashlib.md5(f"{dataset}|{fold}|{self.val_subsample_seed}".encode(), usedforsecurity=False).hexdigest()
        return int(digest[:8], 16)

    def _class_counts(self, y_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.unique(y_val, return_counts=True)

    def _per_class_targets_fraction(self, counts: np.ndarray) -> np.ndarray:
        """Keep ``val_fraction`` of each class, rounded, with at least one row per class."""
        return np.minimum(counts, np.maximum(1, np.round(self.val_fraction * counts).astype(int)))

    def _per_class_targets_total(self, counts: np.ndarray, total: int) -> np.ndarray:
        """Allocate ``total`` rows across classes proportionally, >=1 each, summing to ``total``.

        Uses largest-remainder rounding. If ``total`` is smaller than the number of classes the
        per-class floor of 1 wins (class presence is required for some metrics), so the returned sum
        can exceed ``total`` only in that degenerate case.
        """
        n = int(counts.sum())
        ideal = counts / n * total
        k = np.maximum(1, np.floor(ideal).astype(int))
        k = np.minimum(k, counts)
        diff = total - int(k.sum())
        remainder = ideal - np.floor(ideal)
        if diff > 0:  # hand out the leftover to the largest fractional remainders that have room
            for c in np.argsort(-remainder):
                if diff == 0:
                    break
                add = min(int(counts[c] - k[c]), diff)
                k[c] += add
                diff -= add
        elif diff < 0:  # over-allocated by the >=1 floors; trim smallest remainders, keeping >=1
            for c in np.argsort(remainder):
                if diff == 0:
                    break
                remove = min(int(k[c] - 1), -diff)
                k[c] -= remove
                diff += remove
        return k

    def _select_indices(self, y_val: np.ndarray, problem_type: str, seed: int) -> np.ndarray:
        """Indices of the kept validation rows (stratified for classification)."""
        rng = np.random.default_rng(seed)
        n = len(y_val)

        # Determine the absolute target size; None means "use the per-class fraction targets".
        if self.max_val_samples is not None:
            if n <= self.max_val_samples:
                return np.arange(n)  # already small enough -> use all
            total = self.max_val_samples
        else:
            total = None

        if problem_type in ("binary", "multiclass"):
            classes, counts = self._class_counts(y_val)
            if total is None:
                targets = self._per_class_targets_fraction(counts)
            else:
                targets = self._per_class_targets_total(counts, total)
            selected = [
                rng.choice(np.flatnonzero(y_val == cls), size=int(k), replace=False)
                for cls, k in zip(classes, targets, strict=True)
            ]
            idx = np.concatenate(selected)
        else:  # regression: plain random sample
            k = total if total is not None else min(n, max(1, round(self.val_fraction * n)))
            idx = rng.choice(n, size=int(k), replace=False)

        idx.sort()  # keep original row order for stable downstream behaviour
        return idx

    def subsample_val_data(
        self,
        *,
        dataset: str,
        fold: int,
        problem_type: str,
        y_val: np.ndarray,
        pred_val: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._is_active():
            return y_val, pred_val
        if pred_val.shape[1] != len(y_val):
            raise ValueError(
                f"pred_val row axis ({pred_val.shape[1]}) does not match y_val length "
                f"({len(y_val)}) for ({dataset}, fold {fold}).",
            )
        idx = self._select_indices(y_val=y_val, problem_type=problem_type, seed=self._task_seed(dataset, fold))
        # pred_val is (n_models, n_rows) or (n_models, n_rows, n_classes); subsample the row axis.
        return y_val[idx], pred_val[:, idx]
