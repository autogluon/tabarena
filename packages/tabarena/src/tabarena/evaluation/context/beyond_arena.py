"""``BeyondArenaContext`` — the data-foundry counterpart to :class:`TabArenaContext`.

One of the pluggable evaluation contexts under :mod:`tabarena.evaluation.context`; subclasses
:class:`~tabarena.nips2025_utils.tabarena_context.TabArenaContext`. BeyondArena differs from
TabArena v0.1 in three ways:

* **Subset predicates** — size buckets keyed on ``max_train_rows``, plus split-regime
  (``iid``/``temporal``/``grouped``), feature-dimensionality, text and high-cardinality subsets.
* **Task metadata** — sourced from the self-contained committed BeyondArena reference CSV (via
  :func:`~tabarena.evaluation.beyond_metadata.load_beyond_task_metadata_collection`), whose tasks
  already carry the warehouse fields inline, so no separate ``warehouse_metadata.csv`` merge is
  needed.
* **Method metadata** — the ``"beyond"`` preset selects the Beyond-IID benchmark's method
  collection (artifact ``beyond_iid_benchmark_2026``; see
  :mod:`tabarena.nips2025_utils.artifacts._beyond_method_metadata`).

Everything else (method handling, plotting, leaderboard logic) is inherited unchanged.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import pandas as pd

from tabarena.nips2025_utils.subset_predicate import SubsetPredicate
from tabarena.nips2025_utils.tabarena_context import TabArenaContext

if TYPE_CHECKING:
    from tabarena.benchmark.task.metadata import TaskMetadataCollection
    from tabarena.models._method_metadata import MethodMetadata


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
        # column is the split, so this maps to fold == 0 there (matching the original
        # results-frame "lite" lambda and the base context's convention).
        "lite": SubsetPredicate(lambda df: df["split"] == 0, ("split",)),
    }

    def __init__(
        self,
        methods: list[MethodMetadata] | str = "beyond",
        task_metadata: str | TaskMetadataCollection | pd.DataFrame = "BeyondArena",
        *,
        extra_methods: list[MethodMetadata] | None = None,
        backend: str = "ray",
        fillna_method: str | None = "RF (default)",
        calibration_method: str | None = "XGB (default)",
    ) -> None:
        """Build a BeyondArena context.

        Args:
            methods: ``"beyond"`` (the Beyond-IID method collection), ``"tabarena"`` (the
                TabArena paper methods), or an explicit ``list[MethodMetadata]``.
            task_metadata: A BeyondArena source name (e.g. ``"BeyondArena"``) or a
                ``*_tasks_metadata.csv`` path — loaded via
                :func:`~tabarena.evaluation.beyond_metadata.load_beyond_task_metadata_collection`
                — a ready ``TaskMetadataCollection``, a native-schema (per-split) DataFrame,
                or ``"tabarena"`` to defer to the base loader.
            extra_methods: Additional ``MethodMetadata`` appended to the resolved methods.
            backend: ``"ray"`` or ``"native"``, forwarded to :class:`TabArenaContext`.
            fillna_method: Imputed-method name forwarded to :class:`TabArenaContext`.
            calibration_method: Calibration-method name forwarded to :class:`TabArenaContext`.
        """
        if isinstance(task_metadata, pd.DataFrame):
            # The BeyondArena CSV is in the native (per-split) schema, not the legacy
            # one-row-per-dataset frame, so reconstruct a TaskMetadataCollection from it
            # (matching `beyond_arena_eval`); the base context does not accept a DataFrame.
            from tabarena.benchmark.task.metadata import InMemoryTaskMetadataSource, TaskMetadataCollection

            task_metadata = TaskMetadataCollection(InMemoryTaskMetadataSource(task_metadata).load())

        super().__init__(
            methods=methods,
            task_metadata=task_metadata,
            extra_methods=extra_methods,
            backend=backend,
            fillna_method=fillna_method,
            calibration_method=calibration_method,
        )

    def _resolve_task_metadata_preset(self, name: str) -> TaskMetadataCollection:
        """``"tabarena"`` defers to the base loader; any other name is a BeyondArena suite
        name (``"beyond"`` aliases ``"BeyondArena"``) or a ``*_tasks_metadata.csv`` path,
        loaded self-contained (committed CSV, no downloads).
        """
        if name == "tabarena":
            return super()._resolve_task_metadata_preset(name)
        from tabarena.evaluation.beyond_metadata import load_beyond_task_metadata_collection

        if name == "beyond":
            name = "BeyondArena"
        return load_beyond_task_metadata_collection(name)

    def _resolve_methods_preset(self, name: str) -> list[MethodMetadata]:
        """``"beyond"`` -> the Beyond-IID method collection; other names defer to
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
