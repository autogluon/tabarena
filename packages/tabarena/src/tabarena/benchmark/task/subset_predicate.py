"""Self-describing subset-filter predicate.

A :class:`SubsetPredicate` pairs a boolean mask function with the grid columns it needs, so
applying it to a frame that lacks those columns fails fast with a clear message instead of a
cryptic ``KeyError`` deep in pandas. Arena contexts declare their available filters as a
``dict[str, SubsetPredicate]`` (see ``AbstractArenaContext.SUBSET_PREDICATES``); the evaluator
in :mod:`tabarena.nips2025_utils.compare` validates ``required_columns`` against the task grid /
results frame before evaluating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd


@dataclass(frozen=True)
class SubsetPredicate:
    """A named subset filter plus the columns the mask function reads.

    ``predicate`` maps a frame to a boolean ``Series``; ``required_columns`` are the columns it
    indexes. Instances are callable (``pred(df)``) for backward compatibility with plain-callable
    predicate maps; prefer :meth:`evaluate`, which checks ``required_columns`` first.
    """

    predicate: Callable[[pd.DataFrame], pd.Series]
    required_columns: tuple[str, ...] = ()

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self.predicate(df)

    def evaluate(self, df: pd.DataFrame, *, name: str | None = None) -> pd.Series:
        """Apply the mask, raising a clear error if a required column is missing from ``df``."""
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            label = f"{name!r} " if name else ""
            raise ValueError(
                f"Subset predicate {label}requires column(s) {missing} not present in the grid; "
                f"available columns: {sorted(df.columns)}.",
            )
        return self.predicate(df)


def tasks_in_frame(
    valid_tasks: pd.DataFrame,
    *,
    dataset_col: str = "dataset",
    split_col: str = "split",
) -> SubsetPredicate:
    """A :class:`SubsetPredicate` keeping grid rows whose ``(dataset, split)`` is in ``valid_tasks``.

    Unlike the stateless column-comparison predicates in a context's ``SUBSET_PREDICATES`` (e.g.
    ``"lite"`` == ``split == 0``), this one is *data-dependent*: it closes over an explicit set of
    valid ``(dataset, split)`` tasks (e.g. read from a committed CSV). Use it to express filters a
    single column lambda cannot, such as "the first N splits of each dataset" where N varies per
    dataset. ``dataset_col`` / ``split_col`` name those columns in ``valid_tasks``; the grid it is
    evaluated against must carry ``"dataset"`` and ``"split"``.
    """
    import pandas as pd

    valid: set[tuple[str, int]] = set(
        zip(valid_tasks[dataset_col].astype(str), valid_tasks[split_col].astype(int), strict=False),
    )

    def _mask(df: pd.DataFrame) -> pd.Series:
        keys = zip(df["dataset"].astype(str), df["split"].astype(int), strict=False)
        return pd.Series([key in valid for key in keys], index=df.index)

    return SubsetPredicate(_mask, ("dataset", "split"))
