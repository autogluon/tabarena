from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import pandas as pd

from tabarena.benchmark.task.metadata.schema import (
    SplitMetadata,
    TabArenaTaskMetadata,
    derive_task_type,
    tid_from_task_id_str,
    to_legacy_task_metadata,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from tabarena.benchmark.task.metadata.sources import TaskMetadataSource


class TaskMetadataCollection:
    """The single task-metadata concept: which tasks (dataset x split) a benchmark runs.

    A ``list[TabArenaTaskMetadata]`` plus everything its consumers need:

    * **Construction** — :meth:`from_source` (any
      :class:`~tabarena.benchmark.task.metadata.sources.base.TaskMetadataSource`, suite
      literal, DataFrame, CSV path, or task list) and :meth:`from_preset` (the registered
      benchmark suites, e.g. ``"TabArena-v0.1-lite"``).
    * **Filtering** — :meth:`subset_tasks` (declarative filters: problem types, split
      indices, dataset names, dtypes, train-set size) and :meth:`subset` (explicit
      ``(dataset, fold, repeat)`` triplets).
    * **Materialization** — :meth:`materialize` makes the (already-filtered) tasks
      runnable via the retained source (e.g. Data Foundry downloads only this
      collection's tasks); a no-op for already-local metadata.
    * **Views** — :meth:`dataset_names` / :meth:`dataset_fold_repeats` (native, built
      straight from the tasks), :meth:`per_dataset_frame` (one row per dataset, for
      joining onto results), :meth:`task_grid` (subset-predicate frame).
    * **Serialization** — :meth:`to_dataframe` (native schema, one row per split) which
      round-trips through :meth:`from_source`; plus the legacy boundary adapters
      :meth:`dataset_to_tid` / :meth:`to_legacy_df` for consumers still keyed on the
      legacy schema / OpenML ``tid``.

    Mirrors :class:`~tabarena.models._method_metadata_collection.MethodMetadataCollection`
    on the methods axis. The wrapped list may be "unrolled" (one entry per split) or hold
    multi-split tasks; every view iterates ``splits_metadata`` so both forms behave the same.
    Equality compares the task lists (the source is provenance, not identity).
    """

    def __init__(self, tasks: list[TabArenaTaskMetadata], *, source: TaskMetadataSource | None = None):
        self._tasks = list(tasks)
        self._source = source

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_source(
        cls,
        source: TaskMetadataSource | pd.DataFrame | list[TabArenaTaskMetadata] | str | Path,
        *,
        verbose: bool = False,
    ) -> TaskMetadataCollection:
        """Load a collection from any task-metadata source (no dataset downloads).

        ``source`` is resolved via
        :func:`~tabarena.benchmark.task.metadata.sources.resolve_source`: a
        :class:`TaskMetadataSource` instance, a registered suite literal
        (``"TabArena-v0.1"``, ``"BeyondArena"``), a native-schema DataFrame / CSV path,
        or a ``list[TabArenaTaskMetadata]``. The loaded tasks are unrolled to one entry
        per split, and the resolved source is retained so :meth:`materialize` can later
        make the (filtered) tasks runnable.
        """
        from tabarena.benchmark.task.metadata.sources import resolve_source

        resolved = resolve_source(source)
        tasks = resolved.load(verbose=verbose)
        tasks = [single_ttm for ttm in tasks for single_ttm in ttm.unroll_splits()]
        collection = cls(tasks, source=resolved)
        collection._sanity_check_task_ids()
        return collection

    @classmethod
    def from_preset(cls, preset: str, *, verbose: bool = False) -> TaskMetadataCollection:
        """Load a registered benchmark suite by name (metadata only, no downloads).

        Presets: ``"TabArena-v0.1"``, ``"BeyondArena"``, plus their ``"-lite"`` variants
        (first split — ``r0f0`` — of each dataset). Unlike :meth:`from_source`, an
        unknown name raises instead of being treated as a CSV path.
        """
        suites = ("TabArena-v0.1", "BeyondArena")
        base, lite = (preset[: -len("-lite")], True) if preset.endswith("-lite") else (preset, False)
        if base not in suites:
            options = [s + variant for s in suites for variant in ("", "-lite")]
            raise ValueError(f"Unknown preset {preset!r}. Available presets: {options}.")
        collection = cls.from_source(base, verbose=verbose)
        return collection.subset_tasks(split_indices="lite") if lite else collection

    # ------------------------------------------------------------------ list-like
    @property
    def tasks(self) -> list[TabArenaTaskMetadata]:
        return self._tasks

    @property
    def source(self) -> TaskMetadataSource | None:
        """The source this collection was loaded from (``None`` for directly-built ones)."""
        return self._source

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[TabArenaTaskMetadata]:
        return iter(self._tasks)

    def __getitem__(self, idx):
        return self._tasks[idx]

    def __eq__(self, other: object) -> bool:
        """Collections are equal iff their task lists are equal (source is provenance only)."""
        if not isinstance(other, TaskMetadataCollection):
            return NotImplemented
        return self._tasks == other._tasks

    __hash__ = None  # mutable container semantics (like list)

    # ------------------------------------------------------------------ native views
    def dataset_names(self) -> list[str]:
        """Unique dataset names, in first-seen order."""
        seen: dict[str, None] = {}
        for t in self._tasks:
            seen.setdefault(t.tabarena_task_name, None)
        return list(seen)

    def task_metadata_by_dataset(self) -> dict[str, TabArenaTaskMetadata]:
        """Map dataset name -> one ``TabArenaTaskMetadata`` carrying *all* of its splits.

        The inverse of the per-split unrolling done on ingest (:meth:`from_source`):
        the dataset's per-split entries are re-rolled into a single task metadata —
        task-level fields from the first entry (they are duplicated across splits),
        ``splits_metadata`` merged across entries. Keys follow first-seen order.
        """
        first_entry: dict[str, TabArenaTaskMetadata] = {}
        splits_by_dataset: dict[str, dict] = {}
        for t in self._tasks:
            name = t.tabarena_task_name
            first_entry.setdefault(name, t)
            splits_by_dataset.setdefault(name, {}).update(t.splits_metadata)
        return {name: replace(t, splits_metadata=splits_by_dataset[name]) for name, t in first_entry.items()}

    def dataset_fold_repeats(self) -> list[tuple[str, int, int]]:
        """All ``(dataset, fold, repeat)`` triplets, from each task's ``splits_metadata``."""
        return [
            (t.tabarena_task_name, split.fold, split.repeat)
            for t in self._tasks
            for split in t.splits_metadata.values()
        ]

    def subset(self, dataset_fold_repeats: list[tuple[str, int, int]]) -> TaskMetadataCollection:
        """A new collection restricted to the given ``(dataset, fold, repeat)`` triplets.

        Each task keeps only the splits whose ``(fold, repeat)`` was requested for its
        dataset; tasks left with no splits are dropped. Every requested triplet must be a
        real split of this collection (raises otherwise), so the result's
        :meth:`dataset_fold_repeats` is exactly the requested set. Useful for scoping a
        run to a subset of splits without a separate "allowed triplets" channel — the
        collection itself becomes the single source of truth for what to run.
        """
        valid = set(self.dataset_fold_repeats())
        invalid = [t for t in dataset_fold_repeats if t not in valid]
        if invalid:
            invalid_str = "\n\t".join(str(x) for x in invalid)
            raise ValueError(
                f"{len(invalid)} requested (dataset, fold, repeat) triplet(s) are not splits of this "
                f"collection:\n\t{invalid_str}",
            )

        wanted: dict[str, set[tuple[int, int]]] = {}
        for dataset, fold, repeat in dataset_fold_repeats:
            wanted.setdefault(dataset, set()).add((fold, repeat))

        new_tasks: list[TabArenaTaskMetadata] = []
        for t in self._tasks:
            wanted_splits = wanted.get(t.tabarena_task_name)
            if not wanted_splits:
                continue
            kept = {idx: s for idx, s in t.splits_metadata.items() if (s.fold, s.repeat) in wanted_splits}
            if kept:
                new_tasks.append(replace(t, splits_metadata=kept))
        return TaskMetadataCollection(new_tasks, source=self._source)

    def subset_tasks(
        self,
        *,
        problem_types: list[str] | None = None,
        split_indices: list[str] | Literal["lite"] | None = None,
        dataset_names: list[str] | None = None,
        task_ids: list[str | int] | None = None,
        required_dtypes: list[str] | None = None,
        forbidden_dtypes: list[str] | None = None,
        n_train_samples: tuple[int | None, int | None] | None = None,
        verbose: bool = False,
    ) -> TaskMetadataCollection:
        """A new collection restricted by declarative filters (``None`` = no filter).

        The filters (applied in this order, each on the previous result):

        * ``problem_types`` — keep tasks whose ``problem_type`` is listed
          (options: ``"binary"``, ``"multiclass"``, ``"regression"``).
        * ``split_indices`` — keep only the listed splits (``"r{repeat}f{fold}"`` strings,
          e.g. ``["r0f0", "r0f1"]``); ``"lite"`` keeps only the first split (``r0f0``).
          Tasks left with no splits are dropped.
        * ``dataset_names`` — keep tasks whose ``dataset_name`` is listed; raises if a
          requested name is not in this collection.
        * ``task_ids`` — keep tasks whose ``task_id_str`` is listed (ints are accepted
          for OpenML task ids); raises if a requested id is not in this collection.
        * ``required_dtypes`` / ``forbidden_dtypes`` — keep datasets with at least one /
          no column of the given dtypes (options: ``"numeric"``, ``"categorical"``,
          ``"text"``, ``"datetime"``).
        * ``n_train_samples`` — ``(lower, upper)`` bounds on a split's number of training
          samples; lower is exclusive, upper inclusive, ``None`` means unbounded on that
          side. Splits outside the band are dropped (and tasks left with no splits).

        The source ref is preserved, so the usual flow is filter-then-:meth:`materialize`
        (only the surviving tasks are downloaded). ``verbose`` prints how many tasks
        survived each filter step.
        """
        steps: list[tuple[str, TaskMetadataCollection]] = [("Starting", self)]
        result = self
        if problem_types is not None:
            result = result._with_tasks([t for t in result if t.problem_type in problem_types])
            steps.append(("Filter to problem types", result))
        if split_indices is not None:
            result = result._filter_split_indices(split_indices)
            steps.append(("Filter to splits", result))
        if dataset_names is not None:
            result = result._filter_dataset_names(dataset_names)
            steps.append(("Filter to dataset names", result))
        if task_ids is not None:
            result = result._filter_task_ids(task_ids)
            steps.append(("Filter to task ids", result))
        if required_dtypes is not None or forbidden_dtypes is not None:
            result = result._with_tasks(
                [
                    t
                    for t in result
                    if t.has_supported_dtypes(required_dtypes=required_dtypes, forbidden_dtypes=forbidden_dtypes)
                ],
            )
            steps.append(("Filter to dtypes", result))
        if n_train_samples is not None:
            result = result._filter_train_samples(n_train_samples)
            steps.append(("Filter to dataset size", result))

        if verbose:
            lines = [
                f"Found {len(steps[-1][1])} tasks to run.",
                "\tTask Filter History:",
                *(f"\t({i}) {label}: {len(coll)}." for i, (label, coll) in enumerate(steps, start=1)),
            ]
            print("\n".join(lines))
        return result

    def _with_tasks(self, tasks: list[TabArenaTaskMetadata]) -> TaskMetadataCollection:
        """A new collection over ``tasks``, preserving this collection's source ref."""
        return TaskMetadataCollection(tasks, source=self._source)

    def _filter_split_indices(self, split_indices: list[str] | Literal["lite"]) -> TaskMetadataCollection:
        """Keep only the splits whose ``split_index`` is listed; drop tasks left empty."""
        if split_indices == "lite":
            split_indices = [SplitMetadata.get_split_index(repeat_i=0, fold_i=0)]
        split_index_pattern = re.compile(r"^r\d+f\d+$")
        for split_index in split_indices:
            if not split_index_pattern.match(split_index):
                raise ValueError(f"Invalid SplitIndex format: {split_index!r}, expected 'r{{int}}f{{int}}'")

        wanted = set(split_indices)
        new_tasks = []
        for t in self._tasks:
            kept = {idx: s for idx, s in t.splits_metadata.items() if s.split_index in wanted}
            if kept:
                new_tasks.append(replace(t, splits_metadata=kept))
        return self._with_tasks(new_tasks)

    def _filter_dataset_names(self, dataset_names: list[str]) -> TaskMetadataCollection:
        """Keep tasks whose ``dataset_name`` is listed; raise on names not in the collection."""
        requested = set(dataset_names)
        available = {t.dataset_name for t in self._tasks}
        missing = requested - available
        if missing:
            raise ValueError(
                f"Requested dataset names not found in task metadata: {sorted(missing)}. "
                f"Available dataset names: {sorted(available)}",
            )
        return self._with_tasks([t for t in self._tasks if t.dataset_name in requested])

    def _filter_task_ids(self, task_ids: list[str | int]) -> TaskMetadataCollection:
        """Keep tasks whose ``task_id_str`` is listed; raise on ids not in the collection."""
        requested = {str(task_id) for task_id in task_ids}
        available = {str(t.task_id_str) for t in self._tasks}
        missing = requested - available
        if missing:
            raise ValueError(
                f"Requested task ids not found in task metadata: {sorted(missing)}. "
                f"Available task ids: {sorted(available)}",
            )
        return self._with_tasks([t for t in self._tasks if str(t.task_id_str) in requested])

    def _filter_train_samples(self, n_train_samples: tuple[int | None, int | None]) -> TaskMetadataCollection:
        """Keep splits whose train size is in ``(lower, upper]``; drop tasks left empty."""
        lb, ub = n_train_samples
        lb = lb if lb is not None else 0
        ub = ub if ub is not None else float("inf")
        new_tasks = []
        for t in self._tasks:
            kept = {idx: s for idx, s in t.splits_metadata.items() if lb < s.num_instances_train <= ub}
            if kept:
                new_tasks.append(replace(t, splits_metadata=kept))
        return self._with_tasks(new_tasks)

    # ------------------------------------------------------------------ materialization
    def materialize(self) -> TaskMetadataCollection:
        """Make this collection's tasks runnable, in place; returns ``self`` for chaining.

        Delegates to the retained source's
        :meth:`~tabarena.benchmark.task.metadata.sources.base.TaskMetadataSource.materialize`
        hook: for remote-backed sources (e.g. Data Foundry) this downloads + converts only
        this collection's tasks (so filter first — see :meth:`subset_tasks`) and updates
        each ``task_id_str``; for already-local sources (or a collection built directly
        from a task list) it is a no-op.
        """
        if self._source is not None:
            self._source.materialize(self._tasks)
        self._sanity_check_task_ids()
        return self

    def _sanity_check_task_ids(self) -> None:
        for ttm in self._tasks:
            if ttm.task_id_str is None:
                raise ValueError(f"Task metadata for task {ttm.tabarena_task_name} does not have a task_id_str!")

    def per_dataset_frame(self) -> pd.DataFrame:
        """One row per dataset with the dataset-level (split-invariant) metadata.

        Uses native column names (``problem_type``, ``num_features``, ``num_classes``, ...)
        plus a ``dataset`` key column. Stage 2 can extend this with predicate-derived
        columns (e.g. a ``max_train_rows`` aggregate over splits).
        """
        rows: dict[str, dict] = {}
        for t in self._tasks:
            # Static fields are dataset-level, so the first task seen per dataset wins.
            rows.setdefault(t.tabarena_task_name, t.to_dict(exclude_splits_metadata=True))
        frame = pd.DataFrame(list(rows.values()))
        if not frame.empty:
            frame["dataset"] = list(rows.keys())
        return frame

    def task_grid(self) -> pd.DataFrame:
        """One row per ``(dataset, fold, repeat, split)`` carrying the subset-predicate metadata.

        This is the frame the subset predicates evaluate against (see
        :attr:`~tabarena.nips2025_utils.tabarena_context.TabArenaContext.SUBSET_PREDICATES`): a
        task-level view enumerated natively from each task's ``splits_metadata``, joined with the
        dataset-level predicate columns.

        Columns:

        * ``dataset`` / ``fold`` / ``repeat`` — task identity (raw fold/repeat).
        * ``split`` — ``n_folds * repeat + fold`` (``n_folds`` = max fold + 1 per dataset); this is
          what ``"lite"`` keys on (``split == 0``) and what a results frame's ``fold`` column maps
          to when subsetting.
        * predicate columns, using the predicate-facing names: ``max_train_rows`` (mean per-fold
          train size over the dataset's splits — matches :meth:`to_legacy_df`'s
          ``n_samples_train_per_fold``), ``n_features`` (``num_features``), ``n_classes``
          (``num_classes``), ``problem_type``, and the warehouse fields ``task_type``,
          ``num_cols_after_preprocessing``, ``num_text_cols``, ``num_high_cardinality_cats``
          (``None`` for tasks that don't carry them, e.g. TabArena v0.1).
        """
        # Predicate-facing grid column -> TabArenaTaskMetadata attribute. Warehouse fields are
        # None for tasks that don't carry them (e.g. TabArena v0.1); BeyondArena populates them.
        grid_col_to_field = {
            "n_features": "num_features",
            "n_classes": "num_classes",
            "problem_type": "problem_type",
            "task_type": "task_type",
            "num_cols_after_preprocessing": "num_cols_after_preprocessing",
            "num_text_cols": "num_text_cols",
            "num_high_cardinality_cats": "num_high_cardinality_cats",
        }
        cols = ["dataset", "fold", "repeat", "split", "max_train_rows", *grid_col_to_field]
        n_folds_by_dataset: dict[str, int] = {}
        train_sizes: dict[str, list[float]] = {}
        meta: dict[str, dict] = {}
        for t in self._tasks:
            ds = t.tabarena_task_name
            meta.setdefault(ds, {col: getattr(t, field) for col, field in grid_col_to_field.items()})
            for split in t.splits_metadata.values():
                n_folds_by_dataset[ds] = max(n_folds_by_dataset.get(ds, 0), split.fold + 1)
                train_sizes.setdefault(ds, []).append(split.num_instances_train)
        mean_train = {ds: (sum(sizes) / len(sizes)) for ds, sizes in train_sizes.items()}
        rows = [
            {
                "dataset": ds,
                "fold": fold,
                "repeat": repeat,
                "split": n_folds_by_dataset[ds] * repeat + fold,
                "max_train_rows": mean_train[ds],
                **meta[ds],
            }
            for ds, fold, repeat in self.dataset_fold_repeats()
        ]
        return pd.DataFrame(rows, columns=cols)

    # ------------------------------------------------------------------ serialization
    def to_dataframe(self) -> pd.DataFrame:
        """The native-schema DataFrame: one row per split, all task + split fields.

        Round-trips through :meth:`from_source` (a multi-split task reloads as its
        unrolled splits) — this is the on-disk form used to ship a collection (e.g. to
        benchmark compute nodes) and the same schema as the committed reference CSVs.
        """
        if not self._tasks:
            return pd.DataFrame()
        return pd.concat([t.to_dataframe() for t in self._tasks], ignore_index=True)

    # ------------------------------------------------------------------ legacy boundary
    @classmethod
    def from_legacy_df(cls, task_metadata: pd.DataFrame) -> TaskMetadataCollection:
        """Build a collection from a legacy ``task_metadata`` DataFrame (**lossy** shim).

        Inverse of :meth:`to_legacy_df`, for callers that still hand ``TabArenaContext`` a
        legacy frame (the ``load_task_metadata`` format: one row per dataset with
        ``tid``/``dataset``/``n_folds``/``n_repeats``/``n_features``/``n_classes``/
        ``n_samples_train_per_fold``/...). Prefer the native path
        (:meth:`from_source` / :meth:`from_preset`) when you can — the legacy frame
        lacks most rich fields, so the rebuilt tasks are structurally valid but sparse:

        * derived: ``eval_metric`` (from ``problem_type``), ``is_classification``, and the
          ``splits_metadata`` grid (per-split sizes from ``n_samples_train/test_per_fold``,
          kept as floats so they round-trip through :meth:`to_legacy_df` without truncation).
        * normalized: ``num_classes`` is set to the schema's ``-1`` for regression (rather than
          the legacy ``0``), so the rebuilt collection matches the native convention.
        * unavailable -> ``None``: dtype flags, domain/year/source, group/time split fields,
          multiclass min/max, class-consistency.

        For a *native*-schema frame (``TabArenaTaskMetadata.to_dataframe`` format) use
        :meth:`TabArenaTaskMetadata.from_row` instead — that round-trips losslessly.
        """
        metric_map = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}
        required = [
            "problem_type",
            "n_folds",
            "n_repeats",
            "n_features",
            "n_classes",
            "NumberOfInstances",
            "tid",
            "n_samples_train_per_fold",
            "n_samples_test_per_fold",
        ]
        missing = [c for c in required if c not in task_metadata.columns]
        if missing:
            raise ValueError(f"Legacy task_metadata is missing required columns: {missing}")
        if "dataset" not in task_metadata.columns and "name" not in task_metadata.columns:
            raise ValueError("Legacy task_metadata must have a 'dataset' or 'name' column.")

        tasks: list[TabArenaTaskMetadata] = []
        for row in task_metadata.to_dict("records"):
            dataset = row.get("dataset") or row.get("name")
            problem_type = row["problem_type"]
            # Normalize to the schema's regression convention (-1) rather than preserving the
            # legacy 0, so the rebuilt collection matches the native bundle.
            n_classes = -1 if problem_type == "regression" else int(row["n_classes"])
            num_features = int(row["n_features"])
            num_instances = int(row["NumberOfInstances"])
            # Kept as floats: the legacy frame stores a (possibly fractional) per-fold average,
            # so int() would truncate it. to_legacy_df recovers it via a float mean, so the
            # value round-trips exactly.
            n_train = float(row["n_samples_train_per_fold"])
            n_test = float(row["n_samples_test_per_fold"])
            n_folds, n_repeats = int(row["n_folds"]), int(row["n_repeats"])
            target_name = row.get("target_feature")
            if target_name is not None and pd.isna(target_name):
                target_name = None

            splits_metadata = {
                SplitMetadata.get_split_index(repeat_i=repeat_i, fold_i=fold_i): SplitMetadata(
                    repeat=repeat_i,
                    fold=fold_i,
                    num_instances_train=n_train,
                    num_instances_test=n_test,
                    num_instance_groups_train=n_train,
                    num_instance_groups_test=n_test,
                    num_classes_train=n_classes,
                    num_classes_test=n_classes,
                    num_features_train=num_features,
                    num_features_test=num_features,
                )
                for repeat_i in range(n_repeats)
                for fold_i in range(n_folds)
            }

            tasks.append(
                TabArenaTaskMetadata(
                    dataset_name=dataset,
                    tabarena_task_name=dataset,
                    problem_type=problem_type,
                    is_classification=problem_type != "regression",
                    target_name=target_name,
                    eval_metric=metric_map[problem_type],
                    splits_metadata=splits_metadata,
                    split_time_horizon=None,
                    split_time_horizon_unit=None,
                    stratify_on=None,
                    time_on=None,
                    group_on=None,
                    group_time_on=None,
                    group_labels=None,
                    multiclass_min_n_classes_over_splits=None,
                    multiclass_max_n_classes_over_splits=None,
                    class_consistency_over_splits=None,
                    num_instances=num_instances,
                    num_features=num_features,
                    num_classes=n_classes,
                    num_instance_groups=num_instances,
                    task_id_str=str(int(row["tid"])),
                    task_type=derive_task_type(time_on=None, group_on=None),
                ),
            )
        return cls(tasks)

    def to_legacy_df(self) -> pd.DataFrame:
        """The legacy one-row-per-dataset ``task_metadata`` DataFrame.

        The single named adapter for consumers that still require the legacy schema
        (tabrepo repo generation, ``ExperimentBatchRunner``). Delegates to
        :func:`~tabarena.benchmark.task.metadata.schema.to_legacy_task_metadata`.
        """
        return to_legacy_task_metadata(self._tasks)

    def dataset_to_tid(self) -> dict[str, int]:
        """Map dataset name -> integer OpenML ``tid``, parsed natively from each task's
        ``task_id_str`` (first task per dataset wins; no legacy-frame round-trip).
        """
        out: dict[str, int] = {}
        for t in self._tasks:
            out.setdefault(t.tabarena_task_name, tid_from_task_id_str(t.task_id_str))
        return out
