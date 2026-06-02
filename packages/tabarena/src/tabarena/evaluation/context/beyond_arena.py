"""``BeyondArenaContext`` — the data-foundry counterpart to :class:`TabArenaContext`.

One of the pluggable evaluation contexts under :mod:`tabarena.evaluation.context`; subclasses
:class:`~tabarena.nips2025_utils.tabarena_context.TabArenaContext`. BeyondArena differs from
TabArena v0.1 in two ways:

* **Subset predicates** — size buckets keyed on warehouse ``max_train_rows``, plus split-regime
  (``random``/``temporal``/``grouped``), feature-dimensionality, text and high-cardinality subsets.
* **Task metadata** — sourced from the self-contained committed BeyondArena reference CSV (via
  :func:`~tabarena.evaluation.beyond_metadata.load_beyond_task_metadata`), which already carries
  the warehouse fields inline, so no separate ``warehouse_metadata.csv`` merge is needed.

Everything else (method handling, plotting, leaderboard logic) is inherited unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tabarena.nips2025_utils.tabarena_context import TabArenaContext

if TYPE_CHECKING:
    from collections.abc import Callable

#: Warehouse-level columns BeyondArena predicates/plots rely on. Used by the (idempotent)
#: back-compat merge below when a caller still supplies a separate warehouse frame.
_WAREHOUSE_FIELDS = (
    "task_type",
    "num_cols_after_preprocessing",
    "num_text_cols",
    "num_high_cardinality_cats",
    "domain",
    "dataset_year",
    "source",
    "missing_value_fraction",
)


def _merge_warehouse_fields(task_metadata: pd.DataFrame, warehouse_metadata: pd.DataFrame) -> pd.DataFrame:
    """Idempotently add warehouse fields to ``task_metadata`` from ``warehouse_metadata``.

    Only columns that are absent (or entirely null) are merged in (on ``dataset``), so this is a
    no-op when ``task_metadata`` is already self-contained (the new committed CSV). Kept for
    back-compat with callers that pass a separate warehouse frame.
    """
    missing = [
        c
        for c in _WAREHOUSE_FIELDS
        if c in warehouse_metadata.columns and (c not in task_metadata.columns or task_metadata[c].isna().all())
    ]
    if not missing:
        return task_metadata
    task_metadata = task_metadata.drop(columns=[c for c in missing if c in task_metadata.columns])
    return task_metadata.merge(warehouse_metadata[["dataset", *missing]], on="dataset", how="left")


class BeyondArenaContext(TabArenaContext):
    """Evaluation context for the data-foundry BeyondArena benchmark."""

    SUBSET_PREDICATES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "all": lambda df: pd.Series(True, index=df.index),
        # problem_type
        "binary": lambda df: df["problem_type"] == "binary",
        "multiclass": lambda df: df["problem_type"] == "multiclass",
        "classification": lambda df: df["problem_type"].isin(["binary", "multiclass"]),
        "regression": lambda df: df["problem_type"] == "regression",
        # size buckets keyed on training rows
        # upper bound +350 over 1M due to the AMEX grouped dataset's split sizing
        "large": lambda df: df["max_train_rows"].between(100_001, 1_000_350),
        "medium": lambda df: df["max_train_rows"].between(10_001, 100_000),
        "small": lambda df: df["max_train_rows"].between(1_001, 10_000),
        "tiny": lambda df: df["max_train_rows"].between(101, 1_000),
        # split / task type
        "random": lambda df: df["task_type"] == "random",  # backward compatibility IID name, remove in future
        "iid": lambda df: df["task_type"] == "random",
        "temporal": lambda df: df["task_type"] == "temporal",
        "grouped": lambda df: df["task_type"] == "grouped",
        # feature dimensionality / type
        "low-dim": lambda df: df["num_cols_after_preprocessing"] <= 100,
        "high-dim": lambda df: df["num_cols_after_preprocessing"] > 100,
        "text": lambda df: df["num_text_cols"] > 0,
        "high-cardinality": lambda df: df["num_high_cardinality_cats"] > 0,
        # row-level filter (requires a "fold" column; only meaningful when applied to df_results)
        "lite": lambda df: df["fold"] == 0,
    }

    def __init__(
        self,
        methods: str | list = "tabarena",
        task_metadata: str | pd.DataFrame = "BeyondArena",
        *,
        fillna_method: str | None = "RF (default)",
        calibration_method: str | None = "XGB (default)",
        warehouse_metadata: pd.DataFrame | None = None,
        **kwargs,
    ) -> None:
        """Build a BeyondArena context.

        Args:
            methods: Method preset/list, forwarded to :class:`TabArenaContext`.
            task_metadata: A BeyondArena source name (e.g. ``"BeyondArena"``) or CSV path — loaded
                + collapsed via :func:`~tabarena.evaluation.beyond_metadata.load_beyond_task_metadata`
                — a ready DataFrame, or ``"tabarena"`` to defer to the base loader.
            fillna_method: Imputed-method name forwarded to :class:`TabArenaContext`.
            calibration_method: Calibration-method name forwarded to :class:`TabArenaContext`.
            warehouse_metadata: Optional separate warehouse frame; merged in idempotently for any
                warehouse fields the task metadata lacks (no-op for the self-contained CSV).
            **kwargs: Forwarded verbatim to :class:`TabArenaContext`.
        """
        if isinstance(task_metadata, str) and task_metadata != "tabarena":
            from tabarena.evaluation.beyond_metadata import load_beyond_task_metadata

            task_metadata = load_beyond_task_metadata(task_metadata)
        if warehouse_metadata is not None and isinstance(task_metadata, pd.DataFrame):
            task_metadata = _merge_warehouse_fields(task_metadata, warehouse_metadata)

        super().__init__(
            methods=methods,
            task_metadata=task_metadata,
            fillna_method=fillna_method,
            calibration_method=calibration_method,
            **kwargs,
        )

    @property
    def _default_subsets(self):
        return [
            [],
            ["binary"],
            ["multiclass"],
            ["classification"],
            ["regression"],
            ["tiny"],
            ["small"],
            ["medium"],
            ["large"],
        ]
