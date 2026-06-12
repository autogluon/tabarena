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

from tabarena.nips2025_utils.tabarena_context import TabArenaContext

if TYPE_CHECKING:
    from collections.abc import Callable

    from tabarena.benchmark.task.metadata import TaskMetadataCollection


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
