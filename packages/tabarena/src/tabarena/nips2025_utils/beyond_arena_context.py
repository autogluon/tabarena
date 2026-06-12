"""Arena context for the Beyond-IID benchmark suite.

``BeyondArenaContext`` binds :class:`~tabarena.nips2025_utils.tabarena_context.TabArenaContext`'s
paper workflow to the native ``BeyondArena`` task suite (committed reference metadata, no
downloads) and the Beyond method-metadata collection, and replaces the subset predicates with
the Beyond ones (size buckets, split regime, dimensionality / text / cardinality filters).

The predicates evaluate against the base :meth:`TaskMetadataCollection.task_grid`, which
natively carries the warehouse columns they key on (``task_type``, ``num_text_cols``,
``num_cols_after_preprocessing``, ``num_high_cardinality_cats``) — no collection subclass is
needed.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import pandas as pd

from tabarena.nips2025_utils.subset_predicate import SubsetPredicate
from tabarena.nips2025_utils.tabarena_context import TabArenaContext

if TYPE_CHECKING:
    from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection
    from tabarena.models._method_metadata import MethodMetadata


class BeyondArenaContext(TabArenaContext):
    """Beyond-IID arena: ``"beyond"`` task/method presets + Beyond subset filters."""

    SUBSET_PREDICATES: dict[str, SubsetPredicate] = {
        "all": SubsetPredicate(lambda df: pd.Series(True, index=df.index)),
        # problem_type
        "binary": SubsetPredicate(lambda df: df["problem_type"] == "binary", ("problem_type",)),
        "multiclass": SubsetPredicate(lambda df: df["problem_type"] == "multiclass", ("problem_type",)),
        "classification": SubsetPredicate(
            lambda df: df["problem_type"].isin(["binary", "multiclass"]), ("problem_type",)
        ),
        "regression": SubsetPredicate(lambda df: df["problem_type"] == "regression", ("problem_type",)),
        # size buckets keyed on training rows (upper "large" bound +350 over 1M due to the
        # AMEX grouped dataset's uneven split)
        "large": SubsetPredicate(lambda df: df["max_train_rows"].between(100_001, 1_000_350), ("max_train_rows",)),
        "medium": SubsetPredicate(lambda df: df["max_train_rows"].between(10_001, 100_000), ("max_train_rows",)),
        "small": SubsetPredicate(lambda df: df["max_train_rows"].between(1_001, 10_000), ("max_train_rows",)),
        "tiny": SubsetPredicate(lambda df: df["max_train_rows"].between(101, 1_000), ("max_train_rows",)),
        # split / task type
        "iid": SubsetPredicate(lambda df: df["task_type"] == "random", ("task_type",)),
        # backward-compatible alias of "iid"; remove in the future
        "random": SubsetPredicate(lambda df: df["task_type"] == "random", ("task_type",)),
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
        # split-level filter: keeps split 0 == (fold 0, repeat 0); a results frame's "fold"
        # column is the split, so this maps to fold == 0 there.
        "lite": SubsetPredicate(lambda df: df["split"] == 0, ("split",)),
    }

    def __init__(
        self,
        methods: list[MethodMetadata] | str = "beyond",
        task_metadata: str | TaskMetadataCollection = "beyond",
        *,
        extra_methods: list[MethodMetadata] | None = None,
        backend: str = "ray",
        fillna_method: str | None = "RF (default)",
        calibration_method: str | None = "XGB (default)",
    ):
        super().__init__(
            methods=methods,
            task_metadata=task_metadata,
            extra_methods=extra_methods,
            backend=backend,
            fillna_method=fillna_method,
            calibration_method=calibration_method,
        )

    def _resolve_task_metadata_preset(self, name: str) -> TaskMetadataCollection:
        """``"beyond"`` -> the native ``BeyondArena`` suite from its committed reference
        metadata (no downloads); other names defer to :class:`TabArenaContext`.
        """
        if name != "beyond":
            return super()._resolve_task_metadata_preset(name)
        from tabarena.benchmark.task.metadata import TaskMetadataCollection

        return TaskMetadataCollection.from_preset("BeyondArena")

    def _resolve_methods_preset(self, name: str) -> list[MethodMetadata]:
        """``"beyond"`` -> the Beyond method-metadata collection; other names defer to
        :class:`TabArenaContext`.
        """
        if name != "beyond":
            return super()._resolve_methods_preset(name)
        from tabarena.nips2025_utils.artifacts import beyond_method_metadata_collection

        return copy.deepcopy(beyond_method_metadata_collection.method_metadata_lst)

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
