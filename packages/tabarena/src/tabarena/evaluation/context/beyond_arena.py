"""``BeyondArenaContext`` — the data-foundry counterpart to :class:`TabArenaContext`.

One of the pluggable evaluation contexts under :mod:`tabarena.evaluation.context`; subclasses
:class:`~tabarena.nips2025_utils.tabarena_context.TabArenaContext`. BeyondArena differs from
TabArena v0.1 in two ways:

* **Subset predicates** — size buckets keyed on ``max_train_rows``, plus split-regime
  (``random``/``temporal``/``grouped``), feature-dimensionality, text and high-cardinality subsets.
* **Task metadata** — sourced from the self-contained committed BeyondArena reference CSV (via
  :func:`~tabarena.evaluation.beyond_metadata.load_beyond_task_metadata_collection`), whose tasks
  already carry the warehouse fields inline, so no separate ``warehouse_metadata.csv`` merge is
  needed.

Everything else (method handling, plotting, leaderboard logic) is inherited unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tabarena.nips2025_utils.subset_predicate import SubsetPredicate
from tabarena.nips2025_utils.tabarena_context import TabArenaContext

if TYPE_CHECKING:
    from collections.abc import Callable

    from tabarena.benchmark.task.metadata import TaskMetadataCollection
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

    SUBSET_PREDICATES: dict[str, SubsetPredicate] = {
        "all": SubsetPredicate(lambda df: pd.Series(True, index=df.index)),
        # problem_type
        "binary": SubsetPredicate(lambda df: df["problem_type"] == "binary", ("problem_type",)),
        "multiclass": SubsetPredicate(lambda df: df["problem_type"] == "multiclass", ("problem_type",)),
        "classification": SubsetPredicate(
            lambda df: df["problem_type"].isin(["binary", "multiclass"]), ("problem_type",)
        ),
        "regression": SubsetPredicate(lambda df: df["problem_type"] == "regression", ("problem_type",)),
        # size buckets keyed on training rows
        # upper bound +350 over 1M due to the AMEX grouped dataset's split sizing
        "large": SubsetPredicate(lambda df: df["max_train_rows"].between(100_001, 1_000_350), ("max_train_rows",)),
        "medium": SubsetPredicate(lambda df: df["max_train_rows"].between(10_001, 100_000), ("max_train_rows",)),
        "small": SubsetPredicate(lambda df: df["max_train_rows"].between(1_001, 10_000), ("max_train_rows",)),
        "tiny": SubsetPredicate(lambda df: df["max_train_rows"].between(101, 1_000), ("max_train_rows",)),
        # split / task type
        "random": SubsetPredicate(lambda df: df["task_type"] == "random", ("task_type",)),  # IID; remove in future
        "iid": SubsetPredicate(lambda df: df["task_type"] == "random", ("task_type",)),
        "temporal": SubsetPredicate(lambda df: df["task_type"] == "temporal", ("task_type",)),
        "grouped": SubsetPredicate(lambda df: df["task_type"] == "grouped", ("task_type",)),
        # feature dimensionality / type
        "low-dim": SubsetPredicate(
            lambda df: df["num_cols_after_preprocessing"] <= 100, ("num_cols_after_preprocessing",)
        ),
        "high-dim": SubsetPredicate(
            lambda df: df["num_cols_after_preprocessing"] > 100, ("num_cols_after_preprocessing",)
        ),
        "text": SubsetPredicate(lambda df: df["num_text_cols"] > 0, ("num_text_cols",)),
        "high-cardinality": SubsetPredicate(
            lambda df: df["num_high_cardinality_cats"] > 0, ("num_high_cardinality_cats",)
        ),
        # row-level filter: keeps fold 0 of every repeat (the BeyondArena "lite" convention).
        "lite": SubsetPredicate(lambda df: df["fold"] == 0, ("fold",)),
    }

    def __init__(
        self,
        methods: str | list = "tabarena",
        task_metadata: str | TaskMetadataCollection = "BeyondArena",
        *,
        fillna_method: str | None = "RF (default)",
        calibration_method: str | None = "XGB (default)",
        **kwargs,
    ) -> None:
        """Build a BeyondArena context.

        Args:
            methods: Method preset/list, forwarded to :class:`TabArenaContext`.
            task_metadata: A BeyondArena source name (e.g. ``"BeyondArena"``) or CSV path — loaded
                via :func:`~tabarena.evaluation.beyond_metadata.load_beyond_task_metadata_collection`
                — a ready ``TaskMetadataCollection``, or ``"tabarena"`` to defer to the base loader.
            fillna_method: Imputed-method name forwarded to :class:`TabArenaContext`.
            calibration_method: Calibration-method name forwarded to :class:`TabArenaContext`.
            **kwargs: Forwarded verbatim to :class:`TabArenaContext`.
        """
        if isinstance(task_metadata, str) and task_metadata != "tabarena":
            from tabarena.evaluation.beyond_metadata import load_beyond_task_metadata_collection

            task_metadata = load_beyond_task_metadata_collection(task_metadata)
        if isinstance(task_metadata, pd.DataFrame):
            # The BeyondArena CSV is in the native (per-split) schema, not the legacy
            # one-row-per-dataset frame, so reconstruct a TaskMetadataCollection from it
            # (matching `beyond_arena_eval`); the base context no longer accepts a DataFrame.
            from tabarena.benchmark.task.metadata import InMemoryTaskMetadataSource, TaskMetadataCollection

            task_metadata = TaskMetadataCollection(InMemoryTaskMetadataSource(task_metadata).load())

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
