"""Shared fallback-imputation for per-task metric tables.

A single implementation behind the two near-identical ``fillna_metrics`` methods that used to live
on :class:`~tabarena.evaluation.repo_metrics.RepoMetrics` (keyed on ``framework``) and
:class:`~tabarena.contexts.abstract_arena_context.AbstractArenaContext` (keyed on ``method``,
preserving the per-method descriptive columns). Both now delegate here.

This is the tabarena-leaderboard flavor of imputation; the generic
:meth:`bencheval.evaluator.BenchmarkEvaluator.fillna_data` shares the same core grid/difference/
fill algorithm but does not mark an ``imputed`` flag, does not preserve per-key descriptive
columns, and additionally supports selecting the fallback rows itself (``"worst"`` / a named
method) rather than taking them as an explicit ``df_fillna``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
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

    df_to_fill = df_to_fill.set_index([dataset_col, split_col, key_col], drop=True)
    df_fillna = df_fillna.set_index([dataset_col, split_col], drop=True)
    if key_col in df_fillna.columns:
        df_fillna = df_fillna.drop(columns=[key_col])

    keys = list(df_to_fill.index.unique(level=key_col))
    df_filled = df_fillna.index.to_frame().merge(
        pd.Series(data=keys, name=key_col),
        how="cross",
    )
    df_filled = df_filled.set_index(keys=list(df_filled.columns))

    # (dataset, fold, key) rows in the full grid that are missing from df_to_fill
    nan_vals = df_filled.index.difference(df_to_fill.index)

    fill_cols = list(df_to_fill.columns)
    df_filled[fill_cols] = np.nan
    df_filled[fill_cols] = df_filled[fill_cols].astype(df_to_fill.dtypes)
    df_filled.loc[df_to_fill.index] = df_to_fill

    df_fillna_to_use = df_fillna.loc[nan_vals.droplevel(level=key_col)].copy()
    df_fillna_to_use.index = nan_vals
    df_filled.loc[nan_vals] = df_fillna_to_use

    if "imputed" not in df_filled.columns:
        df_filled["imputed"] = False
    df_filled.loc[nan_vals, "imputed"] = True
    df_filled["imputed"] = df_filled["imputed"].fillna(0).astype(bool)

    df_filled = df_filled.reset_index(drop=False)

    # restore each key's own value for intrinsic columns (the fallback's value was copied in above)
    for c in preserve:
        df_filled[c] = df_filled[key_col].map(per_key[c])

    return df_filled
