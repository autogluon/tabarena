"""Cell budgets for in-context models: sub-sample the training rows when rows x columns exceed what the GPU holds.

An in-context model embeds every training cell, so its peak GPU memory grows with ``rows x columns``. The
BeyondArena run of 2026-09-22 (``tmp_scripts/beyondarena_tfms_22092026_failures.md``) put the edge for TabFM and
LimiX-2 at about 21M cells on a 96 GB card, roughly 4.7 kB per cell. A wrapper turns that into a budget derived
from the card (:func:`gpu_cell_budget`), keeps the row count within it (:func:`rows_within_budget`) and, when the
library offers no row cap of its own, sub-samples its stored context (:func:`stratified_row_subsample`).
"""

from __future__ import annotations

import numpy as np


def gpu_cell_budget(*, bytes_per_cell: float, safety: float) -> int | None:
    """Training cells a fit may hold on the current GPU: ``safety x total VRAM / bytes_per_cell``.

    ``None`` without CUDA (a CPU fit is bounded by host memory, which AutoGluon budgets separately).
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    _free, total = torch.cuda.mem_get_info()
    return int(safety * total / bytes_per_cell)


def rows_within_budget(n_rows: int, n_cols: int, cell_budget: int, *, min_rows: int = 1) -> int:
    """The rows to keep so that ``rows x n_cols`` stays within ``cell_budget``; ``n_rows`` when it already does.

    ``min_rows`` is the smallest sample the model works well with (TabFM recommends 5000 per member); it is
    only exceeded by ``n_rows`` itself.
    """
    n_cols = max(int(n_cols), 1)
    if n_rows * n_cols <= cell_budget:
        return int(n_rows)
    return int(min(n_rows, max(min_rows, cell_budget // n_cols)))


def stratified_row_subsample(y: np.ndarray, n_keep: int, *, classification: bool, seed: int) -> np.ndarray:
    """Sorted indices of ``n_keep`` rows: per class in proportion, every class kept, or uniform for regression."""
    y = np.asarray(y)
    n_rows = len(y)
    rng = np.random.default_rng(seed)
    if n_keep >= n_rows:
        return np.arange(n_rows)
    if not classification:
        return np.sort(rng.choice(n_rows, size=n_keep, replace=False))
    _classes, inverse = np.unique(y, return_inverse=True)
    counts = np.bincount(inverse)
    if n_keep < len(counts):
        return np.sort(rng.choice(n_rows, size=n_keep, replace=False))
    quota = np.maximum(1, np.floor(counts * n_keep / n_rows).astype(int))
    while quota.sum() > n_keep:  # the floor plus the per-class minimum can overshoot by a few rows
        candidates = np.flatnonzero(quota > 1)
        quota[candidates[np.argmax(quota[candidates])]] -= 1
    while quota.sum() < n_keep:
        quota[np.argmax(counts - quota)] += 1
    keep = np.concatenate(
        [rng.choice(np.flatnonzero(inverse == k), size=int(q), replace=False) for k, q in enumerate(quota)]
    )
    return np.sort(keep)
