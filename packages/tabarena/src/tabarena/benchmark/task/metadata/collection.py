from __future__ import annotations

import logging
import re
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, Literal, Self

import pandas as pd

from tabarena.benchmark.task.metadata.schema import (
    SplitMetadata,
    TabArenaTaskMetadata,
    derive_task_type,
    tid_from_task_id_str,
    to_legacy_task_metadata,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path

    from tabarena.benchmark.experiment.job import Job
    from tabarena.benchmark.task.metadata.sources import TaskMetadataSource
    from tabarena.benchmark.task.subset_predicate import SubsetPredicate
    from tabarena.benchmark.task.user_task import UserTask

logger = logging.getLogger(__name__)


def _beyond_arena_subset_predicates() -> dict[str, SubsetPredicate]:
    from tabarena.contexts import BeyondArenaContext

    return BeyondArenaContext.SUBSET_PREDICATES


def _tabarena_subset_predicates() -> dict[str, SubsetPredicate]:
    from tabarena.contexts import TabArenaContext

    return TabArenaContext.SUBSET_PREDICATES


def _preset_subset_predicates_provider(
    suite_name: str,
) -> Callable[[], dict[str, SubsetPredicate]] | None:
    """Lazy provider of a suite's default subset predicates (``None`` for unknown suites).

    Returns a zero-arg thunk so the (heavier) arena-context import happens only when the
    subset filter actually runs — never at :meth:`TaskMetadataCollection.from_preset` time.
    Module-level functions (not local closures) so a collection holding one stays picklable
    (e.g. for the process-parallel ``generate_all_figs``).
    """
    if suite_name == "BeyondArena":
        return _beyond_arena_subset_predicates
    if suite_name == "TabArena-v0.1":
        return _tabarena_subset_predicates
    return None


class TaskMetadataCollection:
    """The single task-metadata concept: which tasks (dataset x split) a benchmark runs.

    A ``list[TabArenaTaskMetadata]`` plus everything its consumers need:

    * **Construction** — :meth:`from_source` (any
      :class:`~tabarena.benchmark.task.metadata.sources.base.TaskMetadataSource`, suite
      literal, DataFrame, CSV path, or task list), :meth:`from_preset` (the registered
      benchmark suites, e.g. ``"TabArena-v0.1"``), and :meth:`from_user_tasks` (local
      ``UserTask`` objects, loading each one's metadata for you).
    * **Filtering** — :meth:`subset_tasks` (declarative filters: problem types, split
      indices, dataset names, dtypes, train-set size, and named subset predicates such as
      ``BeyondArenaContext.SUBSET_PREDICATES``) and :meth:`subset` (explicit
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

    def __init__(
        self,
        tasks: list[TabArenaTaskMetadata],
        *,
        source: TaskMetadataSource | None = None,
        default_predicates_provider: Callable[[], dict[str, SubsetPredicate]] | None = None,
    ):
        self._tasks = list(tasks)
        self._source = source
        # Lazy provider of this collection's default subset predicates (set by `from_preset`
        # per suite family). A zero-arg thunk, called only when `subset_tasks(subset=...)`
        # runs without an explicit `predicates=`, so the (heavier) context import stays lazy.
        self._default_predicates_provider = default_predicates_provider

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

        Presets: ``"TabArena-v0.1"``, ``"BeyondArena"``. Unlike :meth:`from_source`, an
        unknown name raises instead of being treated as a CSV path. To run a single split
        per dataset, filter the result with ``subset_tasks(split_indices="lite")`` (or
        ``subset="lite"``).

        The returned collection remembers the suite's default subset predicates, so
        :meth:`subset_tasks` accepts ``subset=`` without an explicit ``predicates=``
        (BeyondArena -> ``BeyondArenaContext.SUBSET_PREDICATES``, TabArena ->
        ``TabArenaContext.SUBSET_PREDICATES``). The predicates are loaded lazily — only when
        the subset filter actually runs.
        """
        suites = ("TabArena-v0.1", "BeyondArena")
        if preset not in suites:
            raise ValueError(f"Unknown preset {preset!r}. Available presets: {list(suites)}.")
        collection = cls.from_source(preset, verbose=verbose)
        collection._default_predicates_provider = _preset_subset_predicates_provider(preset)
        return collection

    @classmethod
    def from_user_tasks(
        cls,
        user_tasks: UserTask | Iterable[UserTask],
        *,
        verbose: bool = False,
    ) -> TaskMetadataCollection:
        """Build a collection from local ``UserTask`` objects, loading each one's metadata.

        Because a standardized ``UserTask`` (saved to the default cache) carries a portable,
        path-free ``task_id_str``, the resulting collection is runnable as-is: the runner
        resolves each task from that ``task_id_str`` against the ambient cache, so no separate
        ``user_tasks=`` override is needed at ``build_and_run_jobs`` time.
        """
        from tabarena.benchmark.task.user_task import UserTask

        tasks = [user_tasks] if isinstance(user_tasks, UserTask) else list(user_tasks)
        return cls.from_source([task.load().metadata for task in tasks], verbose=verbose)

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
        return self._with_tasks(new_tasks)

    def subset_to_jobs(self, jobs: list[Job]) -> TaskMetadataCollection:
        """A new collection restricted to the ``(dataset, fold, repeat)`` splits the ``jobs`` touch.

        De-duplicates the jobs' splits (several experiments may share one) and delegates to
        :meth:`subset`, so the result's :meth:`dataset_fold_repeats` is exactly the jobs' splits.
        This is the collection a runner resolves against — scope to it (then :meth:`materialize`)
        so only the tasks the jobs actually run are downloaded.
        """
        triples = list(dict.fromkeys(job.task.as_triple() for job in jobs))
        return self.subset(triples)

    def metadata_for_jobs(self, jobs: list[Job]) -> list[TabArenaTaskMetadata]:
        """The :class:`TabArenaTaskMetadata` for each job — one per job, aligned to ``jobs`` order.

        Each entry carries *only that job's split* in ``splits_metadata`` (unrolled to its
        ``(fold, repeat)``), so a single typed object exposes both the dataset-level fields
        (``num_features``, ``num_classes``, ``problem_type``, ``eval_metric``, ...) and the
        per-split :class:`SplitMetadata` (``num_instances_train``, ``num_features_train``, ...).
        """
        index: dict[tuple[str, int, int], TabArenaTaskMetadata] = {}
        for task in self._tasks:
            for single in task.unroll_splits():
                split = next(iter(single.splits_metadata.values()))
                index[(single.tabarena_task_name, split.fold, split.repeat)] = single
        keys = [job.task.as_triple() for job in jobs]
        missing = sorted({key for key in keys if key not in index})
        if missing:
            raise ValueError(f"{len(missing)} job split(s) are not splits of this collection: {missing}")
        return [index[key] for key in keys]

    def subset_tasks(
        self,
        task_subset: TaskSubset | dict[str, Any] | None = None,
        *,
        predicates: dict[str, SubsetPredicate] | None = None,
        verbose: bool = False,
        **filters: Any,
    ) -> TaskMetadataCollection:
        """A new collection restricted by the filters of a :class:`TaskSubset` (``None`` = no filter).

        :class:`TaskSubset` is the single source of truth for *which* filters exist and what each
        one means — see its field docstrings for the per-filter semantics. Specify the scope either
        as a ``TaskSubset`` (or a dict that resolves
        to one) passed as ``task_subset``, or as the same filters given as loose keyword arguments
        (e.g. ``subset_tasks(split_indices="lite")``). Loose keywords are validated against the
        ``TaskSubset`` fields (unknown names raise) and override the passed ``task_subset`` per
        field — so the field set lives in exactly one place.

        The filters are applied in a fixed order, each on the previous result: ``problem_types`` →
        ``split_indices`` → ``dataset_names`` → ``task_ids`` → ``required_dtypes`` /
        ``forbidden_dtypes`` → ``n_train_samples`` → ``subset``. Split-level filters drop tasks
        left with no splits.

        ``predicates`` (orthogonal to the spec) is the subset-name → :class:`SubsetPredicate` map
        the ``subset`` expression resolves against: ``None`` uses this collection's preset default
        (set by :meth:`from_preset`) and otherwise falls back to
        ``TabArenaContext.SUBSET_PREDICATES``; ignored when ``subset`` is ``None``. The source ref
        is preserved, so the usual flow is filter-then-:meth:`materialize` (only the surviving tasks
        are downloaded). ``verbose`` prints how many tasks survived each filter step.
        """
        # TaskSubset owns the field set; loose **filters are folded into a spec (validated there)
        # and override the passed task_subset per field, so both call styles share one definition.
        spec = TaskSubset.from_input(task_subset).merged_with(TaskSubset.from_input(filters))
        steps: list[tuple[str, TaskMetadataCollection]] = [("Starting", self)]
        result = self
        if spec.problem_types is not None:
            result = result._with_tasks([t for t in result if t.problem_type in spec.problem_types])
            steps.append(("Filter to problem types", result))
        if spec.split_indices is not None:
            result = result._filter_split_indices(spec.split_indices)
            steps.append(("Filter to splits", result))
        if spec.dataset_names is not None:
            result = result._filter_dataset_names(spec.dataset_names)
            steps.append(("Filter to dataset names", result))
        if spec.task_ids is not None:
            result = result._filter_task_ids(spec.task_ids)
            steps.append(("Filter to task ids", result))
        if spec.required_dtypes is not None or spec.forbidden_dtypes is not None:
            result = result._with_tasks(
                [
                    t
                    for t in result
                    if t.has_supported_dtypes(
                        required_dtypes=spec.required_dtypes,
                        forbidden_dtypes=spec.forbidden_dtypes,
                    )
                ],
            )
            steps.append(("Filter to dtypes", result))
        if spec.n_train_samples is not None:
            result = result._filter_train_samples(spec.n_train_samples)
            steps.append(("Filter to dataset size", result))
        if spec.subset is not None:
            result = result._filter_subset(spec.subset, predicates=predicates)
            steps.append(("Filter to subset predicates", result))

        if verbose:
            lines = [
                f"Found {len(steps[-1][1])} tasks to run.",
                "\tTask Filter History:",
                *(f"\t({i}) {label}: {len(coll)}." for i, (label, coll) in enumerate(steps, start=1)),
            ]
            logger.info("\n".join(lines))
        return result

    def _with_tasks(self, tasks: list[TabArenaTaskMetadata]) -> TaskMetadataCollection:
        """A new collection over ``tasks``, preserving this collection's source ref and
        default subset-predicate provider.
        """
        return TaskMetadataCollection(
            tasks,
            source=self._source,
            default_predicates_provider=self._default_predicates_provider,
        )

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

    def _filter_subset(
        self,
        subset: str | list[str] | list[str | list[str]],
        *,
        predicates: dict[str, SubsetPredicate] | None,
    ) -> TaskMetadataCollection:
        """Keep splits matching the named subset predicate expression(s); drop tasks left empty.

        Evaluates each expression against :meth:`task_grid` using the same evaluator the
        arena contexts use (so ``|`` / ``!`` and context-defined names like ``"tiny"`` or
        ``"iid"`` behave identically). A string or flat string-list is one view, whose
        expressions intersect (AND) into one surviving ``(dataset, fold, repeat)`` set. A
        list containing any inner list is a union of views — each view's surviving set is
        computed independently, then OR-ed together.

        When ``predicates`` is ``None``, the collection's default provider (set by
        :meth:`from_preset`) is consulted lazily; if there is none either, the evaluator
        falls back to ``TabArenaContext.SUBSET_PREDICATES``.
        """
        from tabarena.nips2025_utils.compare import _evaluate_subset_expression

        if predicates is None and self._default_predicates_provider is not None:
            predicates = self._default_predicates_provider()
        grid = self.task_grid()

        def _surviving_for_view(view: str | list[str]) -> set[tuple[str, int, int]]:
            """The splits surviving one view: a string expression, or string-list AND-ed."""
            expressions = [view] if isinstance(view, str) else list(view)
            view_grid = grid
            for expression in expressions:
                mask = _evaluate_subset_expression(expression, view_grid, predicates=predicates)
                view_grid = view_grid[mask.values]
            return {
                (dataset, int(fold), int(repeat))
                for dataset, fold, repeat in zip(
                    view_grid["dataset"], view_grid["fold"], view_grid["repeat"], strict=False
                )
            }

        if isinstance(subset, list) and any(isinstance(view, list) for view in subset):
            surviving: set[tuple[str, int, int]] = set()
            for view in subset:
                surviving |= _surviving_for_view(view)
        else:
            surviving = _surviving_for_view(subset)
        new_tasks = []
        for t in self._tasks:
            kept = {
                idx: s for idx, s in t.splits_metadata.items() if (t.tabarena_task_name, s.fold, s.repeat) in surviving
            }
            if kept:
                new_tasks.append(replace(t, splits_metadata=kept))
        return self._with_tasks(new_tasks)

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
        plus a ``dataset`` key column, ``max_train_rows`` — the per-dataset *maximum*
        training-fold size over splits, which the BeyondArena size predicates key on (note
        :meth:`task_grid`'s ``max_train_rows`` is the per-dataset *mean*, matching the
        legacy schema) — and ``n_splits``, the total split count per dataset.

        ``n_splits`` is summed across every task for a dataset because it is the
        :class:`~tabarena.benchmark.task.metadata.schema.TabArenaTaskMetadata.n_splits`
        *property* (``len(splits_metadata)``), which ``to_dict`` / ``asdict`` drop — only
        declared dataclass fields survive serialization.
        """
        rows: dict[str, dict] = {}
        max_train_rows: dict[str, int] = {}
        n_splits: dict[str, int] = {}
        for t in self._tasks:
            ds = t.tabarena_task_name
            # Static fields are dataset-level, so the first task seen per dataset wins.
            rows.setdefault(ds, t.to_dict(exclude_splits_metadata=True))
            # TODO: key into task metadata in the future?
            n_splits[ds] = n_splits.get(ds, 0) + len(t.splits_metadata)
            for split in t.splits_metadata.values():
                max_train_rows[ds] = max(max_train_rows.get(ds, 0), split.num_instances_train)
        frame = pd.DataFrame(list(rows.values()))
        if not frame.empty:
            frame["dataset"] = list(rows.keys())
            frame["max_train_rows"] = [max_train_rows[ds] for ds in rows]
            frame["n_splits"] = [n_splits[ds] for ds in rows]
        return frame

    def task_grid(self) -> pd.DataFrame:
        """One row per ``(dataset, fold, repeat, split)`` carrying the subset-predicate metadata.

        This is the frame the subset predicates evaluate against (see
        :attr:`~tabarena.contexts.tabarena.context.TabArenaContext.SUBSET_PREDICATES`): a
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
          ``num_cols_after_preprocessing``, ``num_text_cols``, ``num_high_cardinality_cats``,
          ``group_labels`` (``None`` for tasks that don't carry them, e.g. TabArena v0.1).
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
            "group_labels": "group_labels",
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


@dataclass
class TaskSubset:
    """Typed, declarative scope over a :class:`TaskMetadataCollection` (``None`` field = no filter).

    The single source of truth for *which* subset filters exist and what each one means:
    :meth:`TaskMetadataCollection.subset_tasks` (and therefore
    :meth:`~tabarena.contexts.abstract_arena_context.AbstractArenaContext.build_jobs`) is
    expressed in terms of these fields rather than re-declaring them. Pass a ``TaskSubset`` directly,
    or splat it back into either call via :meth:`as_kwargs`::

        spec = TaskSubset(subset="lite", dataset_names=["anneal"])
        collection.subset_tasks(spec)                 # or: subset_tasks(**spec.as_kwargs())
        context.build_jobs(experiments, task_subset=spec)

    A plain dict (e.g. ``{"subset": "lite"}``) resolves to a ``TaskSubset`` via :meth:`from_input`,
    with unknown keys rejected up front — so dict-based call sites keep working but gain validation.
    The predicate registry (``predicates``) and ``verbose`` are intentionally *not* fields: they are
    orthogonal knobs of ``subset_tasks`` (a caller / arena context supplies its own ``predicates``),
    not part of *which* tasks to keep.
    """

    subset: str | list[str] | list[str | list[str]] | None = None
    """Named subset-predicate expression(s) evaluated against :meth:`TaskMetadataCollection.task_grid`
    (the same predicates an arena context applies in ``compare``). A single string is one expression;
    a flat list of strings is AND-ed together. Within an expression, ``|`` is a union (OR) and a
    leading ``!`` negates an atom — e.g. ``"tiny"``, ``["classification", "!tiny"]``,
    ``"binary|multiclass"``. A *list of lists* is a union (OR) across views — each element is one view
    (a string, or a string-list AND-ed together) and its surviving splits are unioned, so
    ``[["lite", "classification"], ["regression"]]`` keeps the lite classification splits plus every
    regression split. Splits not matching are dropped (and tasks left with no splits), so split-level
    predicates like ``"lite"`` work too. Names resolve against ``subset_tasks``'s ``predicates``."""
    problem_types: list[str] | None = None
    """Keep tasks whose ``problem_type`` is listed (``"binary"`` / ``"multiclass"`` / ``"regression"``)."""
    split_indices: list[str] | Literal["lite"] | None = None
    """Keep only the listed splits (``"r{repeat}f{fold}"`` strings, e.g. ``["r0f0", "r0f1"]``);
    ``"lite"`` keeps only the first split (``r0f0``). Tasks left with no splits are dropped."""
    dataset_names: list[str] | None = None
    """Keep tasks whose ``dataset_name`` is listed; raises if a requested name is not present."""
    task_ids: list[str | int] | None = None
    """Keep tasks whose ``task_id_str`` is listed (ints accepted for OpenML task ids); raises if a
    requested id is not present."""
    required_dtypes: list[str] | None = None
    """Keep datasets with at least one column of the given dtypes (``"numeric"`` / ``"categorical"``
    / ``"text"`` / ``"datetime"``)."""
    forbidden_dtypes: list[str] | None = None
    """Keep datasets with no column of the given dtypes (same options as ``required_dtypes``)."""
    n_train_samples: tuple[int | None, int | None] | None = None
    """``(lower, upper]`` bounds on a split's number of training samples (lower exclusive, upper
    inclusive, ``None`` = unbounded that side). Splits outside the band are dropped (and tasks left
    with no splits)."""

    def as_kwargs(self) -> dict[str, Any]:
        """The non-``None`` fields as a ``subset_tasks`` / ``build_jobs`` keyword dict.

        ``None`` fields (no filter) are dropped, so ``TaskSubset().as_kwargs() == {}`` runs the
        full collection. The result is meant to be splatted: ``subset_tasks(**spec.as_kwargs())``.
        """
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}

    def merged_with(self, other: TaskSubset) -> TaskSubset:
        """A new ``TaskSubset`` with ``other``'s set (non-``None``) fields layered on top of this one.

        Per-field override (``other`` wins where it sets a field; this one's value is kept where
        ``other`` leaves it ``None``) — the typed counterpart of ``{**self_kwargs, **other_kwargs}``.
        Used to combine a base/plan-level scope with a per-job scope.
        """
        return replace(self, **other.as_kwargs())

    @classmethod
    def from_input(cls, value: TaskSubset | dict[str, Any] | None) -> Self:
        """Normalize a ``TaskSubset`` / dict / ``None`` into a ``TaskSubset``.

        ``None`` -> an empty spec (no filter); a ``TaskSubset`` is returned as-is; a dict is
        validated (unknown keys raise, listing the valid field names) and constructed.
        """
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            valid = {f.name for f in fields(cls)}
            unknown = set(value) - valid
            if unknown:
                raise ValueError(
                    f"Unknown TaskSubset field(s) {sorted(unknown)}. Valid fields: {sorted(valid)}.",
                )
            return cls(**value)
        raise TypeError(
            f"Cannot interpret {value!r} as a TaskSubset. Expected a TaskSubset, a dict of "
            f"subset_tasks filters, or None.",
        )


class _PresetTaskMetadataCollection(TaskMetadataCollection):
    """Base for ready-to-use, suite-specific collections built from a registered preset.

    Subclasses set ``_PRESET``; instantiating one is the importable shorthand for
    :meth:`TaskMetadataCollection.from_preset` — metadata only (no downloads), with the
    suite's subset predicates attached (so :meth:`subset_tasks` accepts ``subset=`` without
    ``predicates=``). To run a single split per dataset, filter with
    ``subset_tasks(split_indices="lite")``.
    """

    _PRESET: str

    def __init__(self, *, verbose: bool = False) -> None:
        loaded = TaskMetadataCollection.from_preset(self._PRESET, verbose=verbose)
        super().__init__(
            loaded._tasks,
            source=loaded._source,
            default_predicates_provider=loaded._default_predicates_provider,
        )


class TabArenaTaskMetadataCollection(_PresetTaskMetadataCollection):
    """The TabArena-v0.1 suite as a ready-to-use collection.

    ``TabArenaTaskMetadataCollection()`` == ``TaskMetadataCollection.from_preset("TabArena-v0.1")``.
    Its default subset predicates are ``TabArenaContext.SUBSET_PREDICATES`` (loaded lazily on
    first ``subset=`` use).
    """

    _PRESET = "TabArena-v0.1"


class BeyondArenaTaskMetadataCollection(_PresetTaskMetadataCollection):
    """The BeyondArena suite as a ready-to-use collection.

    ``BeyondArenaTaskMetadataCollection()`` == ``TaskMetadataCollection.from_preset("BeyondArena")``.
    Its default subset predicates are ``BeyondArenaContext.SUBSET_PREDICATES`` (loaded lazily on
    first ``subset=`` use).
    """

    _PRESET = "BeyondArena"
