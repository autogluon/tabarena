"""Shared fallback-imputation for per-task metric tables.

Used by :class:`~tabarena.evaluation.repo_metrics.RepoMetrics` (keyed on ``framework``) and
:class:`~tabarena.contexts.abstract_arena_context.AbstractArenaContext` (keyed on ``method``,
preserving the per-method descriptive columns).

The grid/fill core is delegated to :meth:`bencheval.evaluator.BenchmarkEvaluator.fillna_data` (via
its ``imputed_col`` flag); this wrapper adds the tabarena-leaderboard concern of preserving
intrinsic per-key columns across an imputation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def fillna_metrics(
    df_to_fill: pd.DataFrame,
    df_fillna: pd.DataFrame,
    *,
    key_col: str = "method",
    dataset_col: str = "dataset",
    split_col: str = "fold",
    preserve_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Fill missing ``(dataset, fold, key)`` rows of ``df_to_fill`` from ``df_fillna``.

    For every ``key_col`` value present in ``df_to_fill``, ensures a row exists for each
    ``(dataset_col, split_col)`` task in ``df_fillna``. Any ``(dataset, fold, key)`` row absent
    from ``df_to_fill`` is filled with that ``(dataset, fold)`` row of ``df_fillna`` (the fallback
    method) and flagged ``imputed=True``; rows already present keep their values and are
    ``imputed=False``.

    Both frames are flat (plain columns, not a ``MultiIndex``). ``df_fillna`` must hold exactly one
    row per ``(dataset_col, split_col)`` and is keyed on that pair; a ``key_col`` column on it, if
    present, is dropped (the fallback's own identity is irrelevant to the imputed rows).

    The grid/fill is delegated to :meth:`bencheval.evaluator.BenchmarkEvaluator.fillna_data`; this
    wrapper adds only the ``preserve_columns`` re-broadcast.

    Parameters
    ----------
    df_to_fill
        Per-``(dataset, fold, key)`` rows to complete. Must contain ``dataset_col``, ``split_col``
        and ``key_col`` columns.
    df_fillna
        Fallback rows, one per ``(dataset_col, split_col)`` task; its non-key columns are copied
        into imputed rows.
    key_col
        Column naming the method / framework / config whose missing tasks are imputed.
    dataset_col, split_col
        The two columns that together identify a task.
    preserve_columns
        Columns that are intrinsic to ``key_col`` (constant per key) and must keep the key's own
        value rather than the fallback's after an imputation — e.g. ``method_type`` /
        ``config_type``. They are re-broadcast from each key's (unique) value after filling.
        Columns absent from ``df_to_fill`` are ignored; a column that is not constant within a key
        raises ``AssertionError``.

    Returns:
    -------
    pd.DataFrame
        A flat frame with the original columns of ``df_to_fill`` plus a boolean ``imputed`` column.
    """
    from bencheval.evaluator import BenchmarkEvaluator

    preserve = [c for c in preserve_columns if c in df_to_fill.columns]
    per_key: dict[str, dict] = {}
    for c in preserve:
        groupby_key = df_to_fill.groupby(key_col)[c]
        nunique = groupby_key.nunique(dropna=False)
        invalid = nunique[nunique != 1]
        if not invalid.empty:
            invalid_counts = (
                df_to_fill[df_to_fill[key_col].isin(invalid.index)].groupby(key_col)[c].value_counts(dropna=False)
            )
            raise AssertionError(
                f"Found a {key_col} with multiple values for column {c} (must be unique):\n{invalid_counts}",
            )
        # nunique == 1 for every key, so .first() is that key's single value
        per_key[c] = groupby_key.first().to_dict()

    # Delegate the (dataset, fold) x key grid + fallback fill + imputed flag to bencheval.
    df_filled = BenchmarkEvaluator(
        method_col=key_col,
        task_col=dataset_col,
        seed_column=split_col,
    ).fillna_data(data=df_to_fill, df_fillna=df_fillna, imputed_col="imputed")

    # Restore each key's own value for intrinsic columns (the fallback's value was copied in above).
    for c in preserve:
        df_filled[c] = df_filled[key_col].map(per_key[c])

    return df_filled
