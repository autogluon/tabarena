"""Small helpers shared by the model wrappers.

``root_handlers_preserved`` and ``import_many_class_classifier`` keep a library import or call from changing
the process's logging configuration. The size-cap helpers are an intermediate placeholder: they sub-sample
the training rows of an in-context model above what the GPU holds (about 4.7 kB per training cell for TabFM
on the BeyondArena run) and rank the columns LimiX-2 keeps, until the libraries cap rows and columns or
batch to memory themselves.
"""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def root_handlers_preserved() -> Iterator[None]:
    """Remove the root-logger handlers the block adds.

    A library that logs through the module-level ``logging.info`` / ``logging.warning`` functions
    (tabpfn-extensions at import, TabFM's checkpoint loader) installs a ``StreamHandler`` on a root
    logger that has none, which duplicates every later log line on stderr and fails the global-state
    check of AutoGluon's model tests. Wrap such imports and calls in this context to leave the root
    logger as it was.
    """
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        yield
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)


def import_many_class_classifier() -> type:
    """Return ``tabpfn_extensions.many_class.ManyClassClassifier``, imported under :func:`root_handlers_preserved`."""
    with root_handlers_preserved():
        from tabpfn_extensions.many_class import ManyClassClassifier
    return ManyClassClassifier


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


def univariate_top_columns(X: pd.DataFrame, y: np.ndarray, n_keep: int, *, classification: bool) -> list:
    """The ``n_keep`` columns of ``X`` with the highest univariate F-score against ``y``, in their original order.

    Non-numeric columns are factorized and missing values take the column median (a constant column scores
    0), so the ranking runs on any frame a wrapper stores. All columns when there is nothing to drop.
    """
    if X.shape[1] <= n_keep:
        return list(X.columns)
    from sklearn.feature_selection import f_classif, f_regression

    encoded = np.empty(X.shape, dtype=np.float64)
    for j in range(X.shape[1]):
        column = X.iloc[:, j]
        if isinstance(column.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(column.dtype):
            codes = pd.factorize(column, use_na_sentinel=True)[0].astype(np.float64)
            values = np.where(codes < 0, np.nan, codes)
        else:
            values = column.to_numpy(dtype=np.float64, na_value=np.nan)
        finite = np.isfinite(values)
        encoded[:, j] = np.where(finite, values, np.median(values[finite]) if finite.any() else 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a constant column warns and scores nan, a perfect one scores inf
        scores = (f_classif if classification else f_regression)(encoded, np.asarray(y))[0]
    scores = np.nan_to_num(scores, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)
    keep = np.sort(np.argsort(-scores, kind="stable")[:n_keep])
    return [X.columns[i] for i in keep]
