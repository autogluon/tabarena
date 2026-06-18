"""``AbstractArenaContext`` — the arena-agnostic core shared by every arena context.

An "arena context" bundles a benchmark's task metadata and method metadata and exposes the
operations that depend only on them: building a scoped ``ExperimentBatchRunner``, loading the
methods' artifacts (results, repos, hyperparameters), running HPO / portfolio simulation,
computing a leaderboard via ``compare``, subsetting results, plotting, and rendering a website
leaderboard. None of that is specific to TabArena — it works for any benchmark whose
tasks/methods are described by a :class:`TaskMetadataCollection` /
:class:`MethodMetadataCollection`, and the class is directly instantiable with an explicit
``methods`` list and ``task_metadata`` collection (e.g. for a self-contained custom benchmark).

A concrete arena subclass may override two hooks to support *named presets*:

* :meth:`_resolve_task_metadata_preset` — turn a named preset (e.g. ``"tabarena"``) into a
  :class:`TaskMetadataCollection`.
* :meth:`_resolve_methods_preset` — turn a named methods preset into a ``list[MethodMetadata]``.

(the base accepts no preset names), and may override the class-level :attr:`SUBSET_PREDICATES`
and :attr:`_default_subsets` to declare arena-specific subset filters.
:class:`~tabarena.nips2025_utils.tabarena_context.TabArenaContext` is the reference arena
(TabArena v0.1 presets + the paper-specific workflow); ``BeyondArenaContext`` subclasses it.
"""

from __future__ import annotations

import copy
import functools
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import pandas as pd

from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection, TaskSubset
from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._method_metadata_collection import MethodMetadataCollection
from tabarena.nips2025_utils.per_dataset_tables import get_per_dataset_tables
from tabarena.nips2025_utils.subset_predicate import SubsetPredicate
from tabarena.paper.paper_runner_tabarena import PaperRunTabArena
from tabarena.paper.tabarena_evaluator import TabArenaEvaluator
from tabarena.repository import EvaluationRepository, EvaluationRepositoryCollection
from tabarena.website.website_format import format_leaderboard

if TYPE_CHECKING:
    from tabarena.benchmark.experiment import Experiment, Job
    from tabarena.benchmark.result import BaselineResult
    from tabarena.benchmark.task.metadata.schema import TabArenaTaskMetadata
    from tabarena.repository.abstract_repository import AbstractRepository


class AbstractArenaContext:
    """Arena-agnostic base: task/method metadata + artifact loading, simulation, comparison,
    runner, and leaderboard plumbing. Directly instantiable; subclasses add named presets.
    """

    #: Subset-filter predicates available to `compare` / `build_jobs`, keyed by
    #: name. Each :class:`SubsetPredicate` declares the grid columns it needs (validated before
    #: it runs). Subclasses override to add arena-specific filters (size buckets, split regimes,
    #: ...). Read via :attr:`subset_predicates` so subclass overrides take effect.
    SUBSET_PREDICATES: dict[str, SubsetPredicate] = {
        "all": SubsetPredicate(lambda df: pd.Series(True, index=df.index)),
        "binary": SubsetPredicate(lambda df: df["problem_type"] == "binary", ("problem_type",)),
        "multiclass": SubsetPredicate(lambda df: df["problem_type"] == "multiclass", ("problem_type",)),
        "classification": SubsetPredicate(
            lambda df: df["problem_type"].isin(["binary", "multiclass"]), ("problem_type",)
        ),
        "regression": SubsetPredicate(lambda df: df["problem_type"] == "regression", ("problem_type",)),
        # split-level filter: keeps split 0 == (fold 0, repeat 0). Evaluated on the task grid's
        # "split" column (see TaskMetadataCollection.task_grid); a results frame's "fold" is the
        # split, so this maps to fold == 0 there.
        "lite": SubsetPredicate(lambda df: df["split"] == 0, ("split",)),
    }

    def __init__(
        self,
        methods: str | list[MethodMetadata],
        task_metadata: str | TaskMetadataCollection,
        *,
        extra_methods: list[MethodMetadata] | None = None,
        backend: Literal["ray", "native"] = "ray",
        fillna_method: str | None = None,
        calibration_method: str | None = None,
        only_valid_tasks: bool = False,
    ):
        # A TaskMetadataCollection is the single source of truth; the legacy `task_metadata`
        # DataFrame view is derived from it on demand (see the `task_metadata` cached_property).
        self.task_metadata_collection = self._resolve_task_metadata_collection(task_metadata)
        self.fillna_method = fillna_method
        self.calibration_method = calibration_method
        assert backend in ["ray", "native"]
        self.backend = backend
        self.engine = "ray" if self.backend == "ray" else "sequential"

        # A string selects a named preset (resolved by the concrete arena); otherwise an
        # explicit list of MethodMetadata is used as-is.
        if isinstance(methods, str):
            method_metadata_lst = self._resolve_methods_preset(methods)
        else:
            method_metadata_lst = list(methods)
        self.method_metadata_collection: MethodMetadataCollection = MethodMetadataCollection(method_metadata_lst)

        # Names of the "new" methods registered on top of the baselines via `extra_methods=`
        # (and :meth:`register`). These define the valid-task scope for `only_valid_tasks`
        # (whether their results live in memory or are loaded from disk); `methods=` baselines
        # are the comparison set, not the scope.
        self._new_method_names: set[str] = set()

        # `only_valid_tasks=True` pre-filters the context's task_metadata down to the tasks the
        # registered "new" methods actually ran, so it becomes the single source of truth for
        # what is in scope — `compare` (which scopes results to `task_metadata` by default), the
        # runner, plotting, and per-dataset tables all inherit the restriction without the caller
        # repeating `only_valid_tasks=True` at each call site.
        self.only_valid_tasks = False
        if extra_methods:
            self._register_methods(list(extra_methods), scope_to_valid_tasks=only_valid_tasks)
        elif only_valid_tasks:
            # No new methods to define the scope — raise the standard, helpful error.
            self._scope_to_valid_tasks()

    @classmethod
    def from_new_methods(cls, new_methods: list[MethodMetadata], **kwargs) -> Self:
        """Build a context with ``new_methods`` registered and scoped to the tasks they ran.

        ``new_methods`` are the methods to register, typically the (in-memory) ones returned by
        :meth:`~tabarena.nips2025_utils.end_to_end.EndToEnd.from_raw_to_methods`, though any
        ``MethodMetadata`` works (in-memory or disk-backed). The baselines / task-metadata preset
        and any other settings (``backend``, ``fillna_method``, ...) come from ``**kwargs`` (and
        the concrete arena's constructor defaults).
        """
        return cls(extra_methods=new_methods, only_valid_tasks=True, **kwargs)

    def register(
        self,
        results: list[BaselineResult | dict],
        *,
        new_result_prefix: str | None = None,
        scope_to_valid_tasks: bool = True,
    ) -> list[MethodMetadata]:
        """Register externally-produced raw results into this context as new methods."""
        # Deferred: tabarena.nips2025_utils.end_to_end imports TabArenaContext at module level,
        # which would be circular at import time.
        from tabarena.nips2025_utils.end_to_end import EndToEnd

        new_methods = EndToEnd.from_raw_to_methods(
            results_lst=results,
            task_metadata=self.task_metadata_collection,
            new_result_prefix=new_result_prefix,
        )
        self._register_methods(new_methods, scope_to_valid_tasks=scope_to_valid_tasks)
        return new_methods

    def build_jobs(
        self,
        experiments: list[Experiment],
        *,
        task_subset: TaskSubset | dict | None = None,
        pre_materialize: bool = False,
        **subset_kwargs,
    ) -> list[Job]:
        """Enumerate ``experiments`` x this context's task splits into a flat ``list[Job]``.

        This context's :attr:`task_metadata_collection` is first scoped by
        :meth:`TaskMetadataCollection.subset_tasks`. Express the scope with a typed
        :class:`~tabarena.benchmark.task.metadata.collection.TaskSubset` via ``task_subset=``
        (the source of truth for the available filters), or pass the same filters as loose
        keyword arguments (``subset``, ``dataset_names``, ``split_indices``, ``problem_types``,
        ``task_ids``, ``n_train_samples``, ``required_dtypes``, ...) — loose keywords override
        ``task_subset`` per field. ``subset`` predicate names resolve against this context's
        :attr:`subset_predicates` unless an explicit ``predicates=`` is passed. The scoped
        collection is then expanded by the shared
        :func:`~tabarena.benchmark.experiment.build_jobs` grid enumerator (each experiment x
        each ``(dataset, fold, repeat)`` split, with constraint-violating pairs dropped).

        Materialization (downloading + converting the scoped tasks for remote-backed suites
        like BeyondArena / Data Foundry; a no-op for already-local sources) is lazy by default
        — :meth:`run_jobs` does it right before running. Pass ``pre_materialize=True`` to do it
        here instead, front-loading the network I/O; the download is disk-cached, so a later
        :meth:`run_jobs` still re-materializes but hits the cache.

        The result is ready for :meth:`run_jobs` (or :meth:`run_job` for a single unit).
        """
        from tabarena.benchmark.experiment import build_jobs as build_jobs_grid

        predicates = subset_kwargs.pop("predicates", self.subset_predicates)
        collection = self.task_metadata_collection.subset_tasks(task_subset, predicates=predicates, **subset_kwargs)
        if pre_materialize:
            collection.materialize()
        return build_jobs_grid(experiments, collection)

    def metadata_for_jobs(self, jobs: list[Job]) -> list[TabArenaTaskMetadata]:
        """The :class:`TabArenaTaskMetadata` for each job — one per job, in ``jobs`` order.

        Delegates to :meth:`TaskMetadataCollection.metadata_for_jobs`: each entry carries only
        that job's split in ``splits_metadata``, so one typed object exposes both the dataset-level
        fields (``num_features``, ``problem_type``, ...) and the per-split :class:`SplitMetadata`.
        Lets a caller derive a per-job quantity (e.g. a backend load-balancing cost
        ``split.num_instances_train * meta.num_features``); pair entries with jobs via
        ``zip(jobs, metas)``.
        """
        return self.task_metadata_collection.metadata_for_jobs(jobs)

    def run_jobs(
        self,
        jobs: list[Job],
        *,
        expname: str | Path | None,
        register: bool = True,
        new_result_prefix: str | None = None,
        **runner_kwargs,
    ) -> list[dict[str, Any]]:
        """Run a ``list[Job]`` and register the results.

        Scopes this context's task metadata to the jobs' ``(dataset, fold, repeat)`` splits
        and materializes it (downloading + converting only those tasks for remote-backed
        sources), builds an :class:`ExperimentBatchRunner` over it, and dispatches the jobs
        via :meth:`ExperimentBatchRunner.run_jobs` in a single task-grouped pass (a task
        shared by several experiments is loaded once). When ``register`` (the default), the
        raw results are registered back into this context via :meth:`register` (pre-filtering
        ``task_metadata`` to the tasks just run, so a subsequent :meth:`compare` is scoped to
        them with nothing extra).

        ``expname`` is the runner's results-cache directory (a real path), or ``None`` to cache to
        a throwaway temp dir (cleaned up after) when you only want the returned results and don't
        need a persistent / resumable cache. There is no default; pass a path or ``None``. Extra
        ``**runner_kwargs`` (e.g. ``debug_mode``, ``cache_mode``) reach :class:`ExperimentBatchRunner`.
        Returns the raw per-split result dicts (also registered when ``register`` is True).
        """
        from tabarena.benchmark.experiment import ExperimentBatchRunner

        if not jobs:
            return []
        # Scope-then-materialize: only the tasks the jobs actually touch are downloaded, and
        # the collection itself is the single source of truth for what the runner resolves.
        collection = self.task_metadata_collection.subset_to_jobs(jobs).materialize()
        # `expname=None` -> a throwaway cache: the returned result dicts are in memory, so the
        # cache dir is only needed while the runner runs and is discarded right after.
        tmp_expname = tempfile.TemporaryDirectory() if expname is None else None
        try:
            runner = ExperimentBatchRunner(
                expname=str(tmp_expname.name if tmp_expname is not None else expname),
                task_metadata=collection,
                **runner_kwargs,
            )
            results = runner.run_jobs(jobs)
        finally:
            if tmp_expname is not None:
                tmp_expname.cleanup()
        if register:
            self.register(results, new_result_prefix=new_result_prefix)
        return results

    def run_job(self, job: Job, **kwargs) -> list[dict[str, Any]]:
        """Run a single :class:`Job` — convenience for ``run_jobs([job], ...)``."""
        return self.run_jobs([job], **kwargs)

    def build_and_run_jobs(
        self,
        experiments: list[Experiment],
        *,
        expname: str | Path | None,
        task_subset: TaskSubset | dict | None = None,
        subset: str | list[str] | None = None,
        register: bool = True,
        new_result_prefix: str | None = None,
        pre_materialize: bool = False,
        build_kwargs: dict | None = None,
        **runner_kwargs,
    ) -> list[dict[str, Any]]:
        """Build the jobs for ``experiments`` and run them — :meth:`build_jobs` then :meth:`run_jobs`.

        The one-call path that avoids the two-step. Scoping goes to :meth:`build_jobs`: pass a typed
        :class:`~tabarena.benchmark.task.metadata.collection.TaskSubset` via ``task_subset`` (the
        recommended way — the single source of truth for the available filters), and/or the
        conveniences ``subset`` (a predicate expression; names resolve against this context's
        :attr:`subset_predicates`) and ``build_kwargs`` (any other
        :meth:`TaskMetadataCollection.subset_tasks` filter, e.g.
        ``{"dataset_names": [...], "split_indices": "lite"}``). The conveniences are merged onto
        ``task_subset`` and win per field. Running goes to :meth:`run_jobs` — ``expname`` /
        ``register`` / ``new_result_prefix`` and ``**runner_kwargs`` (``debug_mode``,
        ``cache_mode``, ``user_tasks``, ...). To inspect or slice the jobs before running, call
        :meth:`build_jobs` and :meth:`run_jobs` separately.
        """
        jobs = self.build_jobs(
            experiments,
            task_subset=task_subset,
            pre_materialize=pre_materialize,
            **{"subset": subset, **(build_kwargs or {})},
        )
        return self.run_jobs(
            jobs,
            expname=expname,
            register=register,
            new_result_prefix=new_result_prefix,
            **runner_kwargs,
        )

    # ------------------------------------------------------------------ arena-specific hooks
    def _resolve_task_metadata_preset(self, name: str) -> TaskMetadataCollection:
        """Resolve a named task-metadata preset to a collection. The base arena defines no
        presets — pass an explicit ``TaskMetadataCollection`` or override in a subclass.
        """
        raise ValueError(
            f"Unknown task_metadata preset {name!r}: {type(self).__name__} defines no presets. "
            f"Pass an explicit TaskMetadataCollection instead.",
        )

    def _resolve_methods_preset(self, name: str) -> list[MethodMetadata]:
        """Resolve a named methods preset to a ``list[MethodMetadata]``. The base arena defines
        no presets — pass an explicit list or override in a subclass.
        """
        raise ValueError(
            f"Unknown methods preset {name!r}: {type(self).__name__} defines no presets. "
            f"Pass an explicit list[MethodMetadata] instead.",
        )

    def load_results(
        self,
        methods: list[str] | None = None,
        download_results: str | bool = "auto",
        methods_drop: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load the cached results of this arena's methods (downloading on cache miss).

        These are the baseline/reference results ``compare`` compares new results against.
        A context constructed with no methods contributes none (empty DataFrame), so a
        self-contained arena's leaderboard is computed purely from ``new_results``.
        """
        if methods is None:
            methods = self.methods
        if methods_drop is not None:
            for method in methods_drop:
                if method not in methods:
                    raise AssertionError(
                        f"Specified '{method}' in `methods_drop`, but '{method}' is not present in methods: {methods}",
                    )
            methods = [method for method in methods if method not in methods_drop]
        if not methods:
            return pd.DataFrame()

        df_results_lst = []
        for method in methods:
            method_metadata = self.method_metadata(method=method)
            if isinstance(download_results, bool) and download_results:
                method_downloader = method_metadata.method_downloader()
                method_downloader.download_results()

            try:
                df_results = method_metadata.load_results()
            except FileNotFoundError as err:
                if isinstance(download_results, str) and download_results == "auto":
                    print(
                        f"Missing local results files for method! "
                        f"Attempting to download from s3 and retry... "
                        f'(download_results={download_results}, method="{method_metadata.method}")',
                    )
                    method_downloader = method_metadata.method_downloader()
                    method_downloader.download_results()
                    df_results = method_metadata.load_results()
                else:
                    print(
                        f"Missing local results files for method {method_metadata.method}! "
                        f"Try setting `download_results=True` to get the required files.",
                    )
                    raise err
            df_results_lst.append(df_results)

        return pd.concat(df_results_lst, ignore_index=True)

    # ------------------------------------------------------------------ metadata views
    def _resolve_task_metadata_collection(
        self,
        task_metadata: str | TaskMetadataCollection,
    ) -> TaskMetadataCollection:
        """Normalize the constructor input to a native ``TaskMetadataCollection``.

        Accepts an explicit ``TaskMetadataCollection`` or a named preset (delegated to
        :meth:`_resolve_task_metadata_preset`). A legacy DataFrame / ``list[TabArenaTaskMetadata]``
        is not accepted — wrap it before constructing the context
        (``TaskMetadataCollection.from_legacy_df(df)`` / ``TaskMetadataCollection(tasks)``) so the
        (lossy) legacy conversion stays an explicit, opt-in step at the call site.
        """
        if isinstance(task_metadata, TaskMetadataCollection):
            return task_metadata
        if isinstance(task_metadata, str):
            return self._resolve_task_metadata_preset(task_metadata)
        raise TypeError(
            f"task_metadata must be a preset name or a TaskMetadataCollection, got "
            f"{type(task_metadata).__name__}. Wrap a legacy DataFrame with "
            f"TaskMetadataCollection.from_legacy_df(df) or a list with TaskMetadataCollection(tasks).",
        )

    @functools.cached_property
    def task_metadata(self) -> pd.DataFrame:
        """Legacy one-row-per-dataset ``task_metadata`` DataFrame (back-compat bridge only).

        The sole remaining legacy-DataFrame surface on the context: nothing internal consumes
        it (the runner, ``compare``, and subset predicates all work off
        :attr:`task_metadata_collection` and its native :meth:`~TaskMetadataCollection.task_grid`).
        Kept as a convenience for external callers still on the legacy schema; derived from the
        collection via ``to_legacy_df()`` and cached (effectively immutable post-init).
        """
        return self.task_metadata_collection.to_legacy_df()

    @property
    def subset_predicates(self) -> dict[str, SubsetPredicate]:
        """Predicates available for subset filtering. Reads from
        ``type(self).SUBSET_PREDICATES`` so subclass overrides take effect.
        """
        return type(self).SUBSET_PREDICATES

    @property
    def _default_subsets(self):
        return [
            [],
            ["binary"],
            ["multiclass"],
            ["classification"],
            ["regression"],
        ]

    @property
    def methods(self) -> list[str]:
        return [m.method for m in self.method_metadata_collection.method_metadata_lst]

    def method_metadata(
        self,
        method: str,
        artifact_name: str | None = None,
        s3_bucket: str | None = None,
        s3_prefix: str | None = None,
    ) -> MethodMetadata:
        return self.method_metadata_collection.get_method_metadata(
            method=method,
            artifact_name=artifact_name,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
        )

    def get_method_rename_map(self) -> dict[str, str]:
        method_rename_map = dict()
        method_metadatas = self.method_metadata_collection.method_metadata_lst
        for m in method_metadatas:
            if m.method_type == "config":
                display_name = m.display_name
                if display_name is not None:
                    if m.config_type in method_rename_map:
                        print(
                            f"WARNING: Multiple display_name values detected for the same config_type={m.config_type!r}"
                            f"\n\tdisplay_name 1: {method_rename_map[m.config_type]!r}"
                            f"\n\tdisplay_name 2: {display_name!r}",
                        )
                    method_rename_map[m.config_type] = display_name
        return method_rename_map

    def _method_rename_map_to_display_names(self) -> dict[str, str]:
        """Build a mapping ``"<config_type> (<subtype>)" -> "<display_name>
        (<subtype>)"`` covering every config method in this collection plus
        the bare ``method -> display_name`` mapping for baseline/portfolio
        methods. Used to switch the rendered ``method`` column from
        ``config_type``/``ag_key``-based codes to friendlier display names.
        """
        rename_map: dict[str, str] = {}
        suffixes = [" (default)", " (tuned)", " (tuned + ensemble)"]
        for m in self.method_metadata_collection.method_metadata_lst:
            if not m.display_name:
                continue
            if m.method_type == "config" and m.config_type and m.config_type != m.display_name:
                for suffix in suffixes:
                    rename_map[f"{m.config_type}{suffix}"] = f"{m.display_name}{suffix}"
            elif m.method_type in ("baseline", "portfolio") and m.method != m.display_name:
                rename_map[m.method] = m.display_name
        return rename_map

    def leaderboard_to_website_format(
        self,
        leaderboard: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        method_metadata_info = self.method_metadata_collection.info()
        method_metadata_info = method_metadata_info.rename(
            columns={
                "method": "ta_name",
                "artifact_name": "ta_suite",
            },
        )
        return format_leaderboard(
            df_leaderboard=leaderboard,
            method_metadata_info=method_metadata_info,
            **kwargs,
        )

    # ------------------------------------------------------------------ comparison / runner
    def _registered_new_results(self) -> pd.DataFrame | None:
        """Concatenated results of the registered "new" methods, or None.

        The "new" methods are those registered on top of the baselines via ``extra_methods=``,
        regardless of whether their results live in memory or are loaded from disk. ``methods=``
        baselines are the comparison set, not the scope. ``compare`` treats these as the "new"
        results, so ``only_valid_tasks=True`` can restrict the leaderboard to the tasks they ran
        without the caller having to pass ``new_results`` (or list out method names) again.
        """
        frames = [
            m.load_results()
            for m in self.method_metadata_collection.method_metadata_lst
            if m.method in self._new_method_names
        ]
        return pd.concat(frames, ignore_index=True) if frames else None

    def _registered_valid_task_triplets(self) -> list[tuple[str, int, int]]:
        """``(dataset, fold, repeat)`` triplets the registered new methods ran.

        The valid-task scope used by ``__init__(only_valid_tasks=True)`` to pre-filter
        :attr:`task_metadata_collection`. A results frame's ``fold`` column is the *split*
        index (``n_folds * repeat + fold``), which the task grid carries as ``split``; this
        maps each ``(dataset, split)`` the methods ran onto the collection's ``(fold, repeat)``.
        Pairs absent from the grid are dropped (a method that ran a task outside this context's
        universe does not widen it).
        """
        df_filter = self._registered_new_results()
        if df_filter is None:
            raise ValueError(
                "only_valid_tasks=True needs new methods registered via extra_methods= "
                "(e.g. extra_methods=EndToEnd.from_raw_to_methods(...)) to define the valid tasks.",
            )
        grid = self.task_metadata_collection.task_grid()
        split_to_fold_repeat = {
            (dataset, int(split)): (int(fold), int(repeat))
            for dataset, split, fold, repeat in zip(
                grid["dataset"], grid["split"], grid["fold"], grid["repeat"], strict=False
            )
        }
        triplets: list[tuple[str, int, int]] = []
        seen: set[tuple[str, int]] = set()
        for dataset, split in zip(df_filter["dataset"], df_filter["fold"], strict=False):
            key = (dataset, int(split))
            if key in seen or key not in split_to_fold_repeat:
                continue
            seen.add(key)
            fold, repeat = split_to_fold_repeat[key]
            triplets.append((dataset, fold, repeat))
        if not triplets:
            raise ValueError(
                "only_valid_tasks=True: the registered new methods share no (dataset, split) "
                "with this context's task_metadata, so there is nothing to scope to.",
            )
        return triplets

    def _task_metadata_results_filter(self) -> pd.DataFrame:
        """A ``(dataset, fold)`` frame of this context's task grid for ``filter_to_valid_tasks``.

        A results frame's ``fold`` column is the *split* index, carried by the grid as
        ``split`` — so the grid's ``split`` is renamed to ``fold`` here to line the two up.
        """
        grid = self.task_metadata_collection.task_grid()
        return grid[["dataset", "split"]].rename(columns={"split": "fold"})

    def _register_methods(self, new_methods: list[MethodMetadata], *, scope_to_valid_tasks: bool) -> None:
        """Append ``new_methods`` as "new" methods (shared by ``__init__`` and :meth:`register`).

        Validates name uniqueness, extends :attr:`method_metadata_collection` and
        :attr:`_new_method_names`, and — when ``scope_to_valid_tasks`` — pre-filters
        ``task_metadata`` to the tasks the registered new methods ran (see
        :meth:`_scope_to_valid_tasks`).
        """
        existing = {m.method for m in self.method_metadata_collection.method_metadata_lst}
        for m in new_methods:
            assert m.method not in existing, f"{m.method} already in methods..."
            existing.add(m.method)
        self.method_metadata_collection = MethodMetadataCollection(
            [*self.method_metadata_collection.method_metadata_lst, *new_methods],
        )
        self._new_method_names.update(m.method for m in new_methods)
        if scope_to_valid_tasks:
            self.only_valid_tasks = True
            self._scope_to_valid_tasks()

    def _scope_to_valid_tasks(self) -> None:
        """Pre-filter :attr:`task_metadata_collection` to the registered new methods' tasks.

        Subsets the collection to :meth:`_registered_valid_task_triplets` (raising if no new
        methods define a scope) and invalidates the derived :attr:`task_metadata` cache.
        """
        self.task_metadata_collection = self.task_metadata_collection.subset(
            self._registered_valid_task_triplets(),
        )
        # The legacy `task_metadata` DataFrame is a cached_property derived from the collection;
        # drop it so it is recomputed from the now-filtered collection on next access.
        self.__dict__.pop("task_metadata", None)

    def _resolve_only_valid_tasks(
        self,
        only_valid_tasks: bool | str | list[str] | MethodMetadata | list[MethodMetadata],
        new_results: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame | None, str | list[str] | None]:
        """Resolve ``only_valid_tasks`` (see :meth:`compare`) to a ``(df_filter, names)`` pair.

        Exactly one of the two is non-``None`` (or both ``None`` for "no restriction"):

        * ``df_filter`` — a results frame whose ``(dataset, fold)`` pairs are the valid tasks
          (applied here via ``filter_to_valid_tasks``). Used for ``True`` and ``MethodMetadata``.
        * ``names`` — a method-column name / list handed to the lower-level compare, which
          restricts to the tasks where each named method has results.
        """
        if isinstance(only_valid_tasks, (tuple, np.ndarray)):
            only_valid_tasks = list(only_valid_tasks)
        if isinstance(only_valid_tasks, MethodMetadata):
            only_valid_tasks = [only_valid_tasks]

        if isinstance(only_valid_tasks, list) and any(isinstance(m, MethodMetadata) for m in only_valid_tasks):
            if not all(isinstance(m, MethodMetadata) for m in only_valid_tasks):
                raise TypeError(
                    "only_valid_tasks list must be all method-column names or all MethodMetadata, not mixed.",
                )
            return pd.concat([m.load_results() for m in only_valid_tasks], ignore_index=True), None
        if isinstance(only_valid_tasks, (str, list)):
            return None, only_valid_tasks
        if only_valid_tasks is True:
            df_filter = new_results if new_results is not None else self._registered_new_results()
            if df_filter is None:
                raise ValueError(
                    "only_valid_tasks=True needs new_results or new methods registered via "
                    "extra_methods= (e.g. extra_methods=EndToEnd.from_raw_to_methods(...)) "
                    "to define the valid tasks.",
                )
            return df_filter, None
        return None, None  # False / None -> no restriction

    def compare(
        self,
        output_dir: str | Path | None,
        new_results: pd.DataFrame | None = None,
        ta_results: pd.DataFrame | None = None,
        only_valid_tasks: bool | str | list[str] | MethodMetadata | list[MethodMetadata] = False,
        filter_to_task_metadata: bool = True,
        subset: str | list[str] | None = None,
        tasks: list[tuple[str, int]] | None = None,
        datasets: list[str] | None = None,
        folds: list[int] | None = None,
        score_on_val: bool = False,
        average_seeds: bool = False,
        fillna: str | pd.DataFrame | None = "auto",
        calibration_method: str | None = "auto",
        remove_imputed: bool = False,
        leaderboard_kwargs: dict | None = None,
        figure_file_type: str = "pdf",
        compute_fold_similarity: bool = False,
        fold_similarity_kwargs: dict | None = None,
        return_results: bool = False,
        new_methods_only: bool = False,
        return_single: bool = False,
        **kwargs,
    ) -> pd.DataFrame | pd.Series | tuple[pd.DataFrame | pd.Series, pd.DataFrame]:
        """Compute the leaderboard comparing ``new_results`` against this arena's baselines.

        ``output_dir`` is where the leaderboard figures / CSVs are written (a real path), or
        ``None`` when you only want the returned DataFrame — figures then go to a throwaway temp
        dir that is cleaned up before returning. There is no default; pass a path or ``None``.

        ``ta_results`` defaults to :meth:`load_results` (which includes any registered
        methods); ``new_results`` (if given) are concatenated to them.
        ``fillna`` / ``calibration_method`` resolve ``"auto"`` to the context's settings.

        ``filter_to_task_metadata`` (default ``True``) scopes the results to this context's
        :attr:`task_metadata_collection`: rows whose ``(dataset, fold)`` is not a task of the
        collection are dropped before scoring. For a full-suite context this is a no-op (the
        results are already within the suite); for a context constructed with
        ``only_valid_tasks=True`` (whose ``task_metadata`` was pre-filtered to the registered
        new methods' tasks) it is what restricts the leaderboard to those tasks — so the
        pre-filtered context needs nothing more here. Pass ``False`` to evaluate ``ta_results``
        / ``new_results`` exactly as given (the historical behaviour).

        ``only_valid_tasks`` restricts the leaderboard to a subset of tasks:

        * ``False`` (default) — no restriction.
        * ``True`` — restrict to the tasks the "new" results ran: ``new_results`` if given,
          else the methods registered via ``extra_methods=`` (so a context built with
          ``extra_methods=EndToEnd.from_raw_to_methods(...)`` needs nothing more here).
        * a ``MethodMetadata`` (or list) — restrict to the tasks those registered methods ran.
        * a method-column name (or list) — passed through to the lower-level compare, which
          restricts to the tasks where each named method has results.

        ``compute_fold_similarity`` ranks datasets by how consistently their folds/seeds agree
        (and estimates folds-needed-for-stability), writing ``fold_similarity.csv`` to
        ``output_dir``. ``fold_similarity_kwargs`` is forwarded to
        :meth:`bencheval.evaluator.BenchmarkEvaluator.rank_datasets_by_fold_similarity` (e.g.
        ``{"similarity": "pearson", "target_reliability": 0.95}``); ignored unless
        ``compute_fold_similarity`` is True.

        Two flags shape the return:

        * ``return_results`` (default ``False``) — also return the per-(method, dataset, fold)
          results frame the leaderboard was scored from (columns ``method`` / ``dataset`` /
          ``fold`` / ``metric_error`` / ...). The return becomes ``(leaderboard, results)``
          instead of ``leaderboard``.
        * ``new_methods_only`` (default ``False``) — restrict *both* the leaderboard and the
          returned results to the registered "new" methods (matched by :attr:`_new_method_names`).
          The leaderboard is still computed against all baselines (so elo / rank / win-rate are
          meaningful), then filtered to the new method's row(s). Raises if no new methods are
          registered.
        * ``return_single`` (default ``False``) — like ``new_methods_only`` but asserts there is
          *exactly one* matching new-method row and returns it as a single leaderboard **row**
          (``pd.Series``) instead of a one-row frame — the "I evaluated one model" case. Raises if
          zero or more than one new method is present. Composes with ``return_results`` (the
          results are still the matched per-split frame).
        """
        # Deferred import: tabarena.nips2025_utils.compare imports TabArenaContext at module
        # level, which would be circular at import time.
        from tabarena.nips2025_utils.compare import compare, filter_to_valid_tasks

        if fillna == "auto":
            fillna = self.fillna_method
        if calibration_method == "auto":
            calibration_method = self.calibration_method

        if ta_results is None:
            ta_results = self.load_results(
                download_results="auto",
            )

        if new_results is not None:
            new_results = new_results.copy(deep=True)
            if "method_subtype" not in new_results.columns:
                new_results["method_subtype"] = np.nan

        df_results = pd.concat([ta_results, new_results], ignore_index=True) if new_results is not None else ta_results

        # Scope to the context's task_metadata (the single source of truth): the lower-level
        # `compare` builds the leaderboard from `df_results` alone and never drops rows by
        # task_metadata, so without this a pre-filtered task_metadata would not restrict the
        # leaderboard. No-op for a full-suite context (results already ⊆ the grid). Skipped for
        # an empty frame (no methods / new_results), which carries no dataset/fold columns.
        if filter_to_task_metadata and not df_results.empty:
            df_results = filter_to_valid_tasks(
                df_to_filter=df_results,
                df_filter=self._task_metadata_results_filter(),
            )

        kwargs = kwargs.copy()
        df_filter, passthrough_names = self._resolve_only_valid_tasks(only_valid_tasks, new_results)
        if passthrough_names is not None:
            kwargs["only_valid_tasks"] = passthrough_names
        if df_filter is not None:
            df_results = filter_to_valid_tasks(df_to_filter=df_results, df_filter=df_filter)

        # Subset to the requested tasks here (with this context's subset predicates) — the
        # lower-level `compare` evaluates the results frame as-is.
        if isinstance(subset, str):
            subset = [subset]
        df_results = self.subset_results(
            df_results=df_results,
            subset=subset,
            tasks=tasks,
            datasets=datasets,
            folds=folds,
        )

        # TODO: only methods that exist in runs
        #  Pair with (method, artifact_name)
        method_rename_map = self.get_method_rename_map()

        # `output_dir=None` means "I only want the leaderboard": write the figures / CSVs to a
        # throwaway temp dir (cleaned up after) instead of persisting them. The dir is only needed
        # while the lower-level `compare` runs; the post-processing below never touches it.
        tmp_output_dir = tempfile.TemporaryDirectory() if output_dir is None else None
        try:
            leaderboard = compare(
                df_results=df_results,
                output_dir=Path(tmp_output_dir.name if tmp_output_dir is not None else output_dir),
                task_metadata=self.task_metadata_collection,
                fillna=fillna,
                calibration_framework=calibration_method,
                score_on_val=score_on_val,
                average_seeds=average_seeds,
                remove_imputed=remove_imputed,
                leaderboard_kwargs=leaderboard_kwargs,
                method_rename_map=method_rename_map,
                figure_file_type=figure_file_type,
                compute_fold_similarity=compute_fold_similarity,
                fold_similarity_kwargs=fold_similarity_kwargs,
                **kwargs,
            )
        finally:
            if tmp_output_dir is not None:
                tmp_output_dir.cleanup()
        if new_methods_only or return_single:
            if not self._new_method_names:
                raise ValueError(
                    "new_methods_only/return_single=True but no new methods are registered; register "
                    "results first (e.g. via register() / extra_methods=).",
                )
            # The leaderboard was computed against all baselines; keep only the new method's row(s).
            # `df_results` is pre-rename (the rename map applies only inside scoring), so its
            # `method` column still holds the registered names tracked in `_new_method_names`.
            leaderboard = self._select_new_methods(leaderboard)
            df_results = self._select_new_methods(df_results)
        if return_single:
            if len(leaderboard) != 1:
                raise ValueError(
                    f"return_single=True but {len(leaderboard)} new-method row(s) matched the leaderboard "
                    f"(registered new methods: {sorted(self._new_method_names)}); register exactly one new method.",
                )
            leaderboard = leaderboard.iloc[0]
        if return_results:
            return leaderboard, df_results.reset_index(drop=True)
        return leaderboard

    def _select_new_methods(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rows of ``df`` whose method (a ``method`` column or the index) is a registered new method."""
        methods = df["method"] if "method" in df.columns else df.index.to_series()
        return df[methods.isin(self._new_method_names).to_numpy()]

    def compare_per_dataset(
        self,
        output_dir: str | Path,
        new_results: pd.DataFrame | None = None,
        ta_results: pd.DataFrame | None = None,
        **kwargs,
    ) -> dict[str, pd.DataFrame]:
        output_dir = Path(output_dir)
        if ta_results is None:
            ta_results = self.load_results(
                download_results="auto",
            )
        datasets = sorted(ta_results["dataset"].unique())
        if new_results is not None:
            new_datasets = sorted(new_results["dataset"].unique())
            datasets = sorted(datasets + [d for d in new_datasets if d not in datasets])

        outs = {}
        plot_tuning_kwargs = kwargs.pop("plot_tuning_kwargs", {})
        for dataset in datasets:
            plot_tuning_kwargs_dataset = copy.deepcopy(plot_tuning_kwargs)
            plot_tuning_kwargs_dataset["title"] = f"Dataset: {dataset}"
            outs[dataset] = self.compare(
                output_dir=output_dir / "per_dataset" / dataset,
                ta_results=ta_results,
                new_results=new_results,
                datasets=[dataset],
                plot_tuning_kwargs=plot_tuning_kwargs_dataset,
                **kwargs,
            )
        return outs

    def subset_results(
        self,
        df_results: pd.DataFrame,
        *,
        subset: list[str] | None = None,
        tasks: list[tuple[str, int]] | None = None,
        datasets: list[str] | None = None,
        folds: list[int] | None = None,
    ) -> pd.DataFrame:
        from tabarena.nips2025_utils.compare import subset_tasks

        if subset is not None or datasets is not None or folds is not None or tasks is not None:
            df_results = subset_tasks(
                df_results=df_results,
                subset=subset,
                tasks=tasks,
                datasets=datasets,
                folds=folds,
                task_metadata_og=self.task_metadata_collection,
                predicates=self.subset_predicates,
            )
        return df_results

    # ------------------------------------------------------------------ artifacts / simulation / plotting
    # FIXME: Finish this, it is WIP
    def generate_all_figs(
        self,
        output_dir,
        subsets: list[list[str] | tuple[str, list[str]]] | str = "auto",
        new_results=None,
        compare_kwargs=None,
        tuning_trajectory_kwargs=None,
        plot_compare: bool = True,
        plot_runtime_per_method: bool = False,
        plot_tuning_trajectories: bool = False,
        save_website_leaderboard: bool = False,
        website_leaderboard_kwargs: dict | None = None,
        website_leaderboard_filename: str = "leaderboard_website.csv",
    ) -> None:
        if compare_kwargs is None:
            compare_kwargs = {}
        if tuning_trajectory_kwargs is None:
            tuning_trajectory_kwargs = {}
        if website_leaderboard_kwargs is None:
            website_leaderboard_kwargs = {}
        if subsets == "auto":
            subsets = self._default_subsets
        for subset in subsets:
            output_suffix = None
            if subset is None:
                subset = []
            if isinstance(subset, tuple):
                assert len(subset) == 2
                output_suffix, subset = subset
            if isinstance(subset, str):
                subset = [subset]
            if isinstance(subset, list):
                if output_suffix is None:
                    output_suffix = "all" if not subset else "&".join(subset)
            else:
                raise ValueError(f"Unknown subset: {subset!r}")
            output_dir_subset = output_dir / output_suffix

            # FIXME: new_results
            if plot_compare:
                lb_df = self.compare(
                    output_dir=output_dir_subset,
                    subset=subset,
                    new_results=new_results,
                    subset_label=output_suffix,
                    **compare_kwargs,
                )
                if save_website_leaderboard and lb_df is not None:
                    lb_website = self.leaderboard_to_website_format(
                        leaderboard=lb_df,
                        **website_leaderboard_kwargs,
                    )
                    output_dir_subset.mkdir(parents=True, exist_ok=True)
                    lb_website.to_csv(
                        output_dir_subset / website_leaderboard_filename,
                        index=False,
                    )
            if plot_tuning_trajectories:
                self.plot_tuning_trajectories(
                    save_path=output_dir_subset / "tuning_trajectories",
                    subset=subset,
                    extra_results=new_results,
                    **tuning_trajectory_kwargs,
                )
            if plot_runtime_per_method:
                self.plot_runtime_per_method(
                    save_path=output_dir_subset / "ablation" / "all-runtimes",
                    # new_results=new_results,
                    subset=subset,
                )

    def load_raw(self, method: str, as_holdout: bool = False) -> list[BaselineResult]:
        metadata: MethodMetadata = self.method_metadata(method=method)
        return metadata.load_raw(engine=self.engine, as_holdout=as_holdout)

    def load_repo(
        self, methods: list[str | MethodMetadata] | None = None, config_fallback: str | None = None
    ) -> EvaluationRepositoryCollection:
        if methods is None:
            methods = self.methods
        repos = []
        for method in methods:
            metadata = method if isinstance(method, MethodMetadata) else self.method_metadata(method=method)
            cur_repo = metadata.load_processed()
            repos.append(cur_repo)
        return EvaluationRepositoryCollection(repos=repos, config_fallback=config_fallback)

    def generate_repo(self, method: str) -> Path:
        metadata = self.method_metadata(method=method)
        metadata.generate_repo(
            results_lst=None,
            task_metadata=self.task_metadata_collection,
            cache=True,
            engine=self.engine,
        )
        return metadata.path_processed

    # FIXME: This is a hacky approach, refactor
    def generate_hpo_trajectories(
        self,
        methods: list[str | MethodMetadata],
        n_configs: list[int | None] | str = "auto",
        seeds: int | list[int] = 20,
        n_iterations: int = 40,
        default_method: str | None = None,
        always_include_default: bool = True,
        fixed_configs: list[str] | None = None,
        fit_order: Literal["original", "random"] = "random",
        time_limit: float | None = None,
        backend: Literal["ray", "native"] = "ray",
        repo: EvaluationRepository | None = None,
        folds: list[int] | None = None,
        ta_name: str | None = None,
        ta_suite: str | None = None,
        display_name: str | None = None,
    ) -> pd.DataFrame:
        methods: list[MethodMetadata] = [
            m if isinstance(m, MethodMetadata) else self.method_metadata(m) for m in methods
        ]
        if repo is None:
            repo = self.load_repo(methods=methods)
            if folds is not None:
                repo = repo.subset(folds=folds)
        if not default_method:
            default_method = methods[0]
        else:
            for method in methods:
                if method.method == default_method:
                    default_method = method
                    break
        hpo_trajectory = default_method.generate_hpo_trajectories(
            n_configs=n_configs,
            repo=repo,
            seeds=seeds,
            n_iterations=n_iterations,
            always_include_default=always_include_default,
            fixed_configs=fixed_configs,
            fit_order=fit_order,
            time_limit=time_limit,
            backend=backend,
            config_type=repo.config_types(),
            cache=False,
        )

        hpo_trajectory["ta_name"] = ta_name
        hpo_trajectory["ta_suite"] = ta_suite
        hpo_trajectory["display_name"] = display_name
        return hpo_trajectory

    # TODO: Refine this
    def generate_portfolio_trajectories(
        self,
        configs: list[str],
        config_fallback: str | None = None,
        n_configs: list[int | None] | str = "auto",
        seeds: int | list[int] = 1,
        n_iterations: int = 40,
        fit_order: Literal["original", "random"] = "original",
        time_limit: float | None = None,
        methods: str | None = None,
        repo: EvaluationRepository | None = None,
        folds: list[int] | None = None,
        name: str | None = None,
        ta_name: str | None = None,
        ta_suite: str | None = None,
        display_name: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Given a list of configs, compute the tuning trajectories
        for the first N configs for each N in n_configs.
        """
        if n_configs == "auto":
            n_configs = [
                1,
                2,
                5,
                10,
                25,
                50,
                100,
                150,
                None,  # all configs
            ]
        if isinstance(seeds, int):
            seeds = list(range(seeds))

        if repo is None:
            if methods is not None:
                methods: list[MethodMetadata] = [
                    m if isinstance(m, MethodMetadata) else self.method_metadata(m) for m in methods
                ]
            repo = self.load_repo(methods=methods, config_fallback=config_fallback)

        # TODO: also include config_fallback
        configs_w_fallback = copy.deepcopy(configs)
        if config_fallback is not None:
            if config_fallback not in configs_w_fallback:
                configs_w_fallback.append(config_fallback)
            repo.set_config_fallback(config_fallback=config_fallback)
        repo = repo.subset(configs=configs_w_fallback, folds=folds)

        df_results_hpo_lst = []

        n_config_total = len(configs)

        n_configs = [n_config if n_config is not None else n_config_total for n_config in n_configs]
        n_configs = [n_config for n_config in n_configs if n_config <= n_config_total]
        n_configs = sorted(set(n_configs))

        for n_config in n_configs:
            print(f"Running n_config={n_config}")
            for seed in seeds:
                df_results_hpo = self.simulate_portfolio_from_configs(
                    n_iterations=n_iterations,
                    configs=configs[:n_config],
                    repo=repo,
                    folds=folds,
                    seed=seed,
                    fit_order=fit_order,
                    time_limit=time_limit,
                    **kwargs,
                )

                if name is not None:
                    df_results_hpo["method"] = f"HPO-N{n_config}-{name}"
                df_results_hpo["n_configs"] = n_config
                df_results_hpo["n_iterations"] = n_iterations
                df_results_hpo_lst.append(df_results_hpo)
        hpo_trajectory = pd.concat(df_results_hpo_lst, ignore_index=True)

        hpo_trajectory["ta_name"] = ta_name
        hpo_trajectory["ta_suite"] = ta_suite
        hpo_trajectory["display_name"] = display_name

        return hpo_trajectory

    def combine_hpo(
        self,
        methods: list[str],
        new_config_type: str,
        ta_name: str,
        ta_suite: str,
        method_default: str | None = None,
        repo: EvaluationRepository | None = None,
        n_configs: int | None = None,
        time_limit: float | None = None,
        fit_order: Literal["original", "random"] = "original",
        default_always_first: bool = True,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Perform HPO across multiple methods.

        Returns default, tuned, and tuned + ensembled results.
        """
        if method_default is None:
            method_default = methods[0]
        if repo is None:
            repo = self.load_repo(methods=methods)

        config_type_default = self.method_metadata(method_default).config_type
        simulator = PaperRunTabArena(repo=repo, backend=self.backend)
        config_default = simulator._config_default(config_type=config_type_default, use_first_if_missing=True)
        if config_default is not None:
            default = simulator.run_config_default(model_type=config_type_default)
            default = default.rename(columns={"framework": "method"})
            default["ta_name"] = ta_name
            default["ta_suite"] = ta_suite
            default["config_type"] = new_config_type
            default["method"] = f"{new_config_type} (default)"
        else:
            default = None

        fixed_configs = [config_default] if default_always_first and config_default else None

        tuned = self.run_hpo(
            method=methods,
            repo=repo,
            n_iterations=1,
            n_configs=n_configs,
            time_limit=time_limit,
            fit_order=fit_order,
            seed=seed,
            fixed_configs=fixed_configs,
        )

        tuned_ens = self.run_hpo(
            method=methods,
            repo=repo,
            n_iterations=40,
            n_configs=n_configs,
            time_limit=time_limit,
            fit_order=fit_order,
            seed=seed,
            fixed_configs=fixed_configs,
        )

        tuned["ta_name"] = ta_name
        tuned["ta_suite"] = ta_suite
        tuned["config_type"] = new_config_type
        tuned["method"] = f"{new_config_type} (tuned)"
        tuned_ens["ta_name"] = ta_name
        tuned_ens["ta_suite"] = ta_suite
        tuned_ens["config_type"] = new_config_type
        tuned_ens["method"] = f"{new_config_type} (tuned + ensemble)"

        return pd.concat(
            [
                default,
                tuned,
                tuned_ens,
            ],
            ignore_index=True,
        )

    def run_hpo(
        self,
        method: str | list[str],
        repo: EvaluationRepository = None,
        n_iterations: int = 40,
        n_configs: int | None = None,
        time_limit: float | None = None,
        fit_order: Literal["original", "random"] = "original",
        seed: int = 0,
        **kwargs,
    ) -> pd.DataFrame:
        if not isinstance(method, list):
            method = [method]
        valid_methods = self.methods
        if repo is None:
            repo = self.load_repo(methods=method)
        method_new = []
        for m in method:
            if m in valid_methods:
                method_metadata = self.method_metadata(method=m)
                config_type = method_metadata.config_type
            else:
                config_type = m
            method_new.append(config_type)
        method = method_new
        if len(method) == 1:
            method = method[0]
        simulator = PaperRunTabArena(repo=repo, backend=self.backend)
        df_results_family_hpo = simulator.run_ensemble_config_type(
            config_type=method,
            n_iterations=n_iterations,
            n_configs=n_configs,
            time_limit=time_limit,
            fit_order=fit_order,
            seed=seed,
            **kwargs,
        )
        df_results_family_hpo = df_results_family_hpo.rename(
            columns={
                "framework": "method",
            }
        )
        name = "HPO"
        if n_configs is not None:
            name += f"-N{n_configs}"
        name += f"-{method}"
        df_results_family_hpo["method"] = name
        return df_results_family_hpo

    # FIXME: WIP
    def _run_compare_pca(
        self,
        configs: list[str] | None = None,
        repo: EvaluationRepository | None = None,
        config_fallback: str | None = None,
    ):
        if repo is None:
            repo = self.load_repo(config_fallback=config_fallback)
        if configs is None:
            configs = self._get_config_defaults()

        simulator = PaperRunTabArena(repo=repo, backend=self.backend)
        simulator.evaluator.compute_avg_config_prediction_delta(configs=configs)

    # FIXME: WIP
    def _get_config_defaults(self):
        config_defaults = []
        for m in self.method_metadata_collection.method_metadata_lst:
            if m.method_type != "config":
                continue
            config_defaults.append(m.get_config_default())
        return config_defaults

    def simulate_portfolio_from_configs(
        self,
        configs: list[str],
        config_fallback: str | None = None,
        repo: EvaluationRepositoryCollection = None,
        **kwargs,
    ):
        if repo is None:
            repo = self.load_repo(config_fallback=config_fallback)
        simulator = PaperRunTabArena(repo=repo, backend=self.backend)

        results = simulator.evaluate_ensembles(
            configs=configs,
            **kwargs,
        )

        return results.rename(columns={"framework": "method"})

    def simulate_portfolio_from_configs_per(
        self,
        df_info: pd.DataFrame,
        config_fallback: str | None = None,
        repo: EvaluationRepositoryCollection = None,
        **kwargs,
    ):
        if repo is None:
            repo = self.load_repo(config_fallback=config_fallback)
        simulator = PaperRunTabArena(repo=repo, backend=self.backend)

        results = simulator.evaluate_ensembles_per(
            df_info=df_info,
            **kwargs,
        )

        return results.rename(columns={"framework": "method"})

    def simulate_portfolio_search(
        self,
        methods: list[str],
        config_fallback: str,
        result_baselines: pd.DataFrame,
        repo: EvaluationRepositoryCollection = None,
        config_types: list[str] | None = None,
        selected_types: list[str] | None = None,
        n_portfolio: int = 25,
        n_ensemble: int = 40,
        time_limit: float | None = 14400,
        average_seeds: bool = False,
    ):
        if repo is None:
            repo = self.load_repo(methods=methods, config_fallback=config_fallback)
        if config_types is None:
            config_types = repo.config_types()
        simulator = PaperRunTabArena(repo=repo, backend=self.backend)

        simulator.run_portfolio_search(
            model_types=config_types,
            selected_types=selected_types,
            result_baselines=result_baselines,
            n_portfolio=n_portfolio,
            n_ensemble=n_ensemble,
            time_limit=time_limit,
            average_seeds=average_seeds,
        )

    def run_portfolio(
        self,
        repo: AbstractRepository,
        configs: list[str],
        n_portfolio: int,
        n_ensemble: int | None = None,
        time_limit: int | None = 14400,
    ) -> pd.DataFrame:
        simulator = PaperRunTabArena(repo=repo, backend=self.backend)
        return simulator.run_zs(
            configs=configs,
            n_portfolios=n_portfolio,
            n_ensemble=n_ensemble,
            n_ensemble_in_name=True,
            time_limit=time_limit,
        )

    def simulate_portfolio(
        self, methods: list[str], config_fallback: str, repo: EvaluationRepositoryCollection = None, **kwargs
    ):
        if repo is None:
            repo = self.load_repo(methods=methods, config_fallback=config_fallback)
        simulator = PaperRunTabArena(repo=repo, backend=self.backend)

        df_results_n_portfolio = []
        n_portfolios = [200] if "n_portfolios" not in kwargs else kwargs.pop("n_portfolios")
        for n_portfolio in n_portfolios:
            df_results_n_portfolio.append(
                simulator.run_zs(n_portfolios=n_portfolio, n_ensemble=None, n_ensemble_in_name=False, **kwargs)
            )
        return pd.concat(df_results_n_portfolio, ignore_index=True)

    def run_portfolio_from_config_types(
        self,
        repo: AbstractRepository,
        config_types: list[str],
        n_portfolio: int,
        n_ensemble: int | None = None,
        time_limit: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        simulator = PaperRunTabArena(repo=repo, backend=self.backend)
        return simulator.run_zs_from_types(
            config_types=config_types,
            n_portfolios=n_portfolio,
            n_ensemble=n_ensemble,
            n_ensemble_in_name=True,
            time_limit=time_limit,
            **kwargs,
        )

    def load_hpo_results(self, method: str) -> pd.DataFrame:
        metadata = self.method_metadata(method=method)
        return metadata.load_hpo_results()

    def load_config_results(self, method: str) -> pd.DataFrame:
        metadata = self.method_metadata(method=method)
        return metadata.load_model_results()

    def load_portfolio_results(self, method: str) -> pd.DataFrame:
        metadata = self.method_metadata(method=method)
        return metadata.load_portfolio_results()

    def load_configs_hyperparameters(
        self,
        methods: list[str] | None = None,
        download: bool | str = False,
    ) -> dict[str, dict]:
        if methods is None:
            methods = self.methods
            methods = [m for m in methods if self.method_metadata(m).method_type == "config"]
        configs_hyperparameters_lst = []
        for method in methods:
            metadata = self.method_metadata(method=method)
            configs_hyperparameters = metadata.load_configs_hyperparameters(download=download)
            configs_hyperparameters_lst.append(configs_hyperparameters)

        def merge_dicts_no_duplicates(dicts: list[dict]) -> dict:
            merged = {}
            for d in dicts:
                for key in d:
                    if key in merged:
                        raise KeyError(
                            f"Duplicate key found in configs_hyperparameters: {key}\n"
                            f"This should never happen and may mean that a given config name "
                            f"belongs to multiple different hyperparameters!",
                        )
                merged.update(d)
            return merged

        return merge_dicts_no_duplicates(configs_hyperparameters_lst)

    def plot_tuning_trajectories(
        self,
        save_path: str | Path,
        subset: list[str] | None = None,
        **kwargs,
    ):
        from tabarena.plot.tuning_trajectories.plot_pareto_over_tuning_time import plot_tuning_trajectories

        plot_tuning_trajectories(
            tabarena_context=self,
            fig_save_dir=save_path,
            subset_map=subset,
            **kwargs,
        )

    def plot_tuning_trajectories_per_dataset(
        self,
        save_path: str | Path,
        file_ext: str = ".pdf",
        to_grid: bool = False,
        **kwargs,
    ):
        if to_grid:
            assert file_ext == ".png", f"to_grid=True only works with file_ext={'.png'!r}"
        from tabarena.plot.tuning_trajectories.plot_pareto_over_tuning_time import plot_tuning_trajectories_per_dataset

        plot_tuning_trajectories_per_dataset(
            tabarena_context=self,
            fig_save_dir=save_path,
            file_ext=file_ext,
            **kwargs,
        )

        if to_grid:
            self._make_png_grid(
                save_path=save_path,
            )

    def _make_png_grid(
        self,
        save_path: str | Path,
        suffix: str | Path = "tuning_trajectories/pareto_n_configs_err_tot_train.png",
        output_suffix: str | Path = "per_dataset_train_vs_error.png",
        n_cols: int = 5,
        datasets: list[str] | None = None,
    ):
        from tabarena.plot.png_to_grid import make_png_grid

        if not datasets:
            datasets = sorted(self.task_metadata_collection.dataset_names())

        n_datasets = len(datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols

        prefix = save_path
        output_path = save_path.parent / output_suffix

        png_files = [prefix / dataset / suffix for dataset in datasets]
        make_png_grid(
            image_paths=png_files,
            output_path=output_path,
            n_rows=n_rows,
            n_cols=n_cols,
            padding=12,
            bg_color=(255, 255, 255, 255),
            resize_mode="fit",
            scale=0.33,
        )

    def plot_runtime_per_method(
        self,
        save_path: str | Path,
        df_results_configs: pd.DataFrame = None,
        subset: list[str] | None = None,
        **kwargs,
    ):
        if df_results_configs is None:
            df_results_configs = self.load_config_results_multi()
        else:
            df_results_configs = df_results_configs.copy()
        if "imputed" in df_results_configs.columns:
            # Remove imputed results
            df_results_configs["imputed"] = df_results_configs["imputed"].fillna(0)
            df_results_configs = df_results_configs[df_results_configs["imputed"] == 0]

        if subset:
            df_results_configs = self.subset_results(df_results=df_results_configs, subset=subset)

        # Group/legend by per-method display_name (falling back to the method name)
        # rather than config_type, so methods sharing a config_type (e.g. CPU/GPU
        # variants like RealMLP and RealMLP_GPU) get distinct lines.
        method_to_display_name = {
            m.method: m.display_name
            for m in self.method_metadata_collection.method_metadata_lst
            if m.method_type == "config"
        }
        df_results_configs["config_type"] = (
            df_results_configs["ta_name"].map(method_to_display_name).fillna(df_results_configs["ta_name"])
        )

        deep_dive_kwargs = dict(kwargs.pop("deep_dive_kwargs", None) or {})

        evaluator = TabArenaEvaluator(output_dir=save_path, task_metadata=self.task_metadata_collection)
        evaluator.generate_runtime_plot(
            df_results=df_results_configs,
            deep_dive_kwargs=deep_dive_kwargs,
            **kwargs,
        )

    def generate_per_dataset_tables(
        self,
        save_path: str | Path,
        df_results: pd.DataFrame = None,
        fillna_method: str | None = "auto",  # FIXME: Don't hardcode
        per_dataset_dir: str | Path | None = None,
        method_order: list[str] | None = None,
        use_display_names: bool = False,
    ):
        if fillna_method == "auto":
            fillna_method = self.fillna_method
        if df_results is None:
            df_results = self.load_results(download_results="auto")

        if use_display_names:
            rename_map = self._method_rename_map_to_display_names()
            if rename_map:
                df_results = df_results.copy()
                df_results["method"] = df_results["method"].replace(rename_map)
                if fillna_method in rename_map:
                    fillna_method = rename_map[fillna_method]

        if fillna_method is not None:
            df_results = self.fillna_metrics(
                df_to_fill=df_results,
                df_fillna=df_results[df_results["method"] == fillna_method],
            )

        get_per_dataset_tables(
            df_results=df_results,
            save_path=Path(save_path),
            task_metadata=self.task_metadata_collection,
            per_dataset_dir=Path(per_dataset_dir) if per_dataset_dir is not None else None,
            method_order=method_order,
        )

    def load_config_results_multi(
        self,
        method_metadata_lst: list[MethodMetadata] | None = None,
    ) -> pd.DataFrame:
        if method_metadata_lst is None:
            method_metadata_lst = self.method_metadata_collection.method_metadata_lst
        df_results_configs_lst = []
        for method_metadata in method_metadata_lst:
            if method_metadata.method_type == "config":
                df_results_configs_lst.append(method_metadata.load_model_results())
        return pd.concat(df_results_configs_lst, ignore_index=True)

    def find_missing(self, method: str):
        metadata = self.method_metadata(method=method)
        repo = EvaluationRepository.from_dir(path=metadata.path_processed)

        tasks = repo.tasks()
        n_tasks = len(tasks)
        print(f"Method: {method} | n_tasks={n_tasks}")

        metrics = repo.metrics()
        metrics = metrics.reset_index(drop=False)

        configs = repo.configs()

        n_configs = len(configs)

        runs_missing_lst = []

        fail_dict = {}
        for i, config in enumerate(configs):
            metrics_config = metrics[metrics["framework"] == config]
            n_tasks_config = len(metrics_config)

            tasks_config = list(metrics_config[["dataset", "fold"]].values)
            tasks_config = {tuple(t) for t in tasks_config}

            n_tasks_missing = n_tasks - n_tasks_config
            tasks_missing = [t for t in tasks if t not in tasks_config] if n_tasks_missing != 0 else []

            for dataset, fold in tasks_missing:
                runs_missing_lst.append(
                    (dataset, fold, config),
                )

            print(f"{n_tasks_missing}\t{config}\t{i + 1}/{n_configs}")
            fail_dict[config] = n_tasks_missing

        # fail_series = pd.Series(fail_dict).sort_values()

        df_missing = pd.DataFrame(data=runs_missing_lst, columns=["dataset", "fold", "framework"])
        df_missing = df_missing.rename(columns={"framework": "method"})
        print(df_missing)

        # save_pd.save(path="missing_runs.csv", df=df_missing)

        return df_missing

    @classmethod
    def fillna_metrics(cls, df_to_fill: pd.DataFrame, df_fillna: pd.DataFrame) -> pd.DataFrame:
        """Fills missing (dataset, fold, framework) rows in df_to_fill with the (dataset, fold) row in df_fillna.

        Parameters
        ----------
        df_to_fill
        df_fillna

        Returns:
        -------

        """
        method_col = "method"
        split_col = "fold"
        dataset_col = "dataset"

        columns_to_keep = [
            "method_type",
            "method_subtype",
            "config_type",
            "ta_name",
            "ta_suite",
        ]
        columns_to_keep = [c for c in columns_to_keep if c in df_to_fill]
        per_column: dict[str, dict] = {}
        for c in columns_to_keep:
            groupby_method = df_to_fill.groupby(method_col)[c]
            nunique = groupby_method.nunique(dropna=False)
            invalid = nunique[nunique != 1]
            df_to_fill_invalid = df_to_fill[df_to_fill[method_col].isin(invalid.index)]
            groupby_method_invalid = df_to_fill_invalid.groupby(method_col)[c]
            if not invalid.empty:
                raise AssertionError(
                    f"Found a method with multiple values for column {c} (must be unique):\n"
                    f"{groupby_method_invalid.value_counts(dropna=False)}",
                )

            # Using .first() is safe because nunique == 1 for every method
            per_column[c] = groupby_method.first().to_dict()

        df_to_fill = df_to_fill.set_index([dataset_col, split_col, method_col], drop=True)
        df_fillna = df_fillna.set_index([dataset_col, split_col], drop=True).drop(columns=[method_col])

        unique_frameworks = list(df_to_fill.index.unique(level=method_col))

        df_filled = df_fillna.index.to_frame().merge(
            pd.Series(data=unique_frameworks, name=method_col),
            how="cross",
        )
        df_filled = df_filled.set_index(keys=list(df_filled.columns))

        # missing results
        nan_vals = df_filled.index.difference(df_to_fill.index)

        # fill valid values
        fill_cols = list(df_to_fill.columns)
        df_filled[fill_cols] = np.nan
        df_filled[fill_cols] = df_filled[fill_cols].astype(df_to_fill.dtypes)
        df_filled.loc[df_to_fill.index] = df_to_fill

        df_fillna_to_use = df_fillna.loc[nan_vals.droplevel(level=method_col)].copy()
        df_fillna_to_use.index = nan_vals
        df_filled.loc[nan_vals] = df_fillna_to_use

        if "imputed" not in df_filled.columns:
            df_filled["imputed"] = False
        df_filled.loc[nan_vals, "imputed"] = True
        df_filled["imputed"] = df_filled["imputed"].fillna(0).astype(bool)

        df_filled = df_filled.reset_index(drop=False)

        # Overwrite values column-by-column while preserving order
        for c in columns_to_keep:
            mapping = per_column[c]
            df_filled[c] = df_filled[method_col].map(mapping)

        return df_filled
