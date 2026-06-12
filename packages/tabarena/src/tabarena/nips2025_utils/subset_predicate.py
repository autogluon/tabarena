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
