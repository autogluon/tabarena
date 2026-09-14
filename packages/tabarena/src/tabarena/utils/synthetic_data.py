"""Small synthetic tabular frames for warm-ups, smoke tests and audits.

The frames come from a fixed seed and carry no information about any benchmark task: a few
Gaussian numeric columns, a low-cardinality categorical column stored as a pandas ``category``,
and a label derived from a linear rule over the numeric columns (thresholded at quantiles for
classification so every class is present, continuous with noise for regression).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_CATEGORY_LEVELS = ("a", "b", "c")


def make_synthetic_frames(
    problem_type: str,
    *,
    n_rows: int = 96,
    n_features: int = 6,
    n_categorical: int = 1,
    seed: int = 0,
    n_rows_predict: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build ``(X_train, y_train, X_predict)`` for ``problem_type`` from a fixed seed.

    Args:
        problem_type: ``"binary"`` (two classes, labels 0 and 1), ``"multiclass"`` (three classes,
            labels 0 to 2) or ``"regression"`` (float labels).
        n_rows: Training rows.
        n_features: Total columns; the first ``n_features - n_categorical`` are numeric floats.
        n_categorical: Columns stored as pandas ``category`` dtype with levels ``a``, ``b``, ``c``.
        seed: Seed of the ``numpy.random.default_rng`` used for every draw.
        n_rows_predict: Rows of ``X_predict``; defaults to a quarter of ``n_rows`` (at least 4).

    Returns:
        ``X_train`` and ``X_predict`` as DataFrames with columns ``f0`` to ``f{n-1}`` (categorical
        columns last) and ``y_train`` as a Series named ``target`` aligned with ``X_train``.
    """
    if problem_type not in ("binary", "multiclass", "regression"):
        raise ValueError(f"Unknown problem_type {problem_type!r}; expected binary, multiclass or regression.")
    if not 0 <= n_categorical <= n_features:
        raise ValueError("n_categorical must lie between 0 and n_features.")
    if n_rows_predict is None:
        n_rows_predict = max(4, n_rows // 4)
    rng = np.random.default_rng(seed)
    n_numeric = n_features - n_categorical

    def _frame(n: int) -> tuple[pd.DataFrame, np.ndarray]:
        numeric = rng.standard_normal((n, n_numeric))
        columns = {f"f{i}": numeric[:, i] for i in range(n_numeric)}
        for j in range(n_categorical):
            levels = rng.choice(_CATEGORY_LEVELS, size=n)
            columns[f"f{n_numeric + j}"] = pd.Categorical(levels, categories=list(_CATEGORY_LEVELS))
        score = numeric.sum(axis=1) if n_numeric else np.zeros(n)
        return pd.DataFrame(columns), score

    X_train, score = _frame(n_rows)
    X_predict, _ = _frame(n_rows_predict)

    if problem_type == "regression":
        y = score + 0.1 * rng.standard_normal(n_rows)
        y_train = pd.Series(y.astype(np.float64), name="target")
    else:
        n_classes = 2 if problem_type == "binary" else 3
        cuts = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1]) if n_numeric else []
        labels = np.digitize(score, cuts) if n_numeric else np.arange(n_rows) % n_classes
        y_train = pd.Series(labels.astype(np.int64), name="target")
    return X_train, y_train, X_predict
