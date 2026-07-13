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
:class:`~tabarena.contexts.tabarena.context.TabArenaContext` is the reference arena
(TabArena v0.1 presets + the paper-specific workflow); ``BeyondArenaContext`` subclasses it.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import tempfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import pandas as pd

from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection, TaskSubset
from tabarena.benchmark.task.subset_predicate import SubsetPredicate
from tabarena.evaluation.leaderboard_reporter import LeaderboardReporter
from tabarena.models._in_memory_method_metadata import InMemoryMethodMetadata
from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._method_metadata_collection import MethodMetadataCollection
from tabarena.models._method_simulator import MethodSimulator
from tabarena.nips2025_utils.per_dataset_tables import get_per_dataset_tables
from tabarena.repository import EvaluationRepository, EvaluationRepositoryCollection
from tabarena.simulation.repo_simulator import RepoSimulator
from tabarena.website.website_format import format_leaderboard

if TYPE_CHECKING:
    from tabarena.benchmark.experiment import Experiment, Job
    from tabarena.benchmark.result import BaselineResult
    from tabarena.benchmark.task.metadata.schema import TabArenaTaskMetadata
    from tabarena.caching import CacheConfig
    from tabarena.repository.abstract_repository import AbstractRepository

# Sentinel default for `run_jobs`/`build_and_run_jobs` `expname`, distinguishing "caller omitted
# expname" (fall back to `cache_config.results`, else error) from an explicit `expname=None`
# (always a throwaway temp dir). Keeps the original "expname is required; None means throwaway" API.
_EXPNAME_UNSET: Any = object()


class AbstractArenaContext:
    """Arena-agnostic base: task/method metadata + artifact loading, simulation, comparison,
    runner, and leaderboard plumbing. Directly instantiable; subclasses add named presets.
    """

    #: Human-readable benchmark name, vended to plot titles (e.g. ``"Arena-large Pareto
    #: Frontier"``). Subclasses override (``TabArenaContext`` -> ``"TabArena"``,
    #: ``BeyondArenaContext`` -> ``"BeyondArena"``).
    benchmark_name: str = "Arena"

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

    #: Named shortcuts for standard subslices, each a list of :attr:`SUBSET_PREDICATES` names
    #: AND-ed together (the same form ``compare`` / ``build_jobs`` accept as a ``subset``). Lets
    #: callers refer to a reusable slice (e.g. ``"high_dim"`` -> ``["core", "high-dim"]``) by name
    #: instead of respelling the predicate list. Base context defines none; subclasses override.
    #: Read via :attr:`subset_shortcuts` so subclass overrides take effect.
    SUBSET_SHORTCUTS: dict[str, list[str]] = {}

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
        cache_config: CacheConfig | None = None,
    ):
        # Configure the caches first: `cache_config.apply()` points this (driver) process at
        # the declared OpenML / HuggingFace / TabArena locations, so resolving + materializing
        # task metadata below — and any later `build_jobs(pre_materialize=...)` / `compare()` —
        # reads and writes the right directories. `cache_config.apply_on_run` re-applies the same
        # config inside `run_jobs`, so a worker that runs this context inherits it too. Stored
        # before resolving the collection so preset hooks can honor it.
        #
        # `cache_config.scope_openml=True` keeps the ambient `openml.config` untouched: we apply
        # only the TabArena/HuggingFace caches here (the TabArena cache is needed by `compare`, and
        # is private to us), and point OpenML at `cache_config.openml` *only* for the duration of
        # each data operation (see `_cache_scope`), restoring the prior OpenML location after.
        self.cache_config = cache_config
        if cache_config is not None:
            cache_config.apply(openml=not cache_config.scope_openml)

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
        :meth:`~tabarena.end_to_end.EndToEnd.from_raw_to_methods`, though any
        ``MethodMetadata`` works (in-memory or disk-backed). The baselines / task-metadata preset
        and any other settings (``backend``, ``fillna_method``, ...) come from ``**kwargs`` (and
        the concrete arena's constructor defaults).
        """
        return cls(extra_methods=new_methods, only_valid_tasks=True, **kwargs)

    def register(
        self,
        results: list[BaselineResult | dict] | list[MethodMetadata],
        *,
        new_result_prefix: str | None = None,
        scope_to_valid_tasks: bool = True,
    ) -> list[MethodMetadata]:
        """Register externally-produced results into this context as new methods.

        ``results`` is either:

        * raw results (``list[BaselineResult | dict]``, e.g. what
          :meth:`~tabarena.benchmark.experiment.ExperimentBatchRunner.run_jobs` returns) —
          converted to methods via :meth:`EndToEnd.from_raw_to_methods`, honoring
          ``new_result_prefix``; or
        * already-built methods (``list[MethodMetadata]``, e.g. from
          ``EndToEnd.from_path_raw(...).to_method_metadata_lst(...)`` or
          :meth:`EndToEnd.from_raw_to_methods`) — registered as-is. The prefix is baked in when
          the methods are built, so ``new_result_prefix`` must be ``None`` here (raises
          otherwise).

        Either way the methods are appended via :meth:`_register_methods` (which tracks them as
        "new" methods and, when ``scope_to_valid_tasks``, pre-filters ``task_metadata`` to their
        tasks). Returns the registered ``MethodMetadata`` list.
        """
        is_method = [isinstance(r, MethodMetadata) for r in results]
        if results and any(is_method):
            if not all(is_method):
                raise TypeError(
                    "register() received a mix of MethodMetadata and raw results; pass either a "
                    "list of raw results or a list of MethodMetadata, not both.",
                )
            if new_result_prefix is not None:
                raise ValueError(
                    "new_result_prefix only applies to raw results; it is baked into "
                    "MethodMetadata when they are built. Pass new_result_prefix to "
                    "to_method_metadata_lst / from_raw_to_methods instead.",
                )
            new_methods = list(results)
        else:
            # Deferred: keeps context imports cheap (the pipeline pulls in simulation machinery).
            from tabarena.end_to_end import EndToEnd

            new_methods = EndToEnd.from_raw_to_methods(
                results_lst=results,
                task_metadata=self.task_metadata_collection,
                new_result_prefix=new_result_prefix,
            )
        self._register_methods(new_methods, scope_to_valid_tasks=scope_to_valid_tasks)
        return new_methods

    @contextlib.contextmanager
    def _cache_scope(self) -> Iterator[None]:
        """Ensure the configured caches are active for the wrapped cache-touching operation.

        No-op when there is no ``cache_config`` or its ``apply_on_run`` is off. With
        ``cache_config.scope_openml=True`` the OpenML root is set only for the duration of the
        block and the prior location is restored afterwards (so an ambient ``openml.config`` is
        preserved); otherwise the config is applied to the process (re-applying covers a worker
        that reconstructed this context).
        """
        cfg = self.cache_config
        if cfg is None or not cfg.apply_on_run:
            yield
        elif cfg.scope_openml:
            with cfg.scoped_openml():
                yield
        else:
            cfg.apply()
            yield

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
            # Materializing downloads into the OpenML/HF caches; make sure this process is
            # pointed at the configured locations first (covers a context built/used on a worker).
            with self._cache_scope():
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
        expname: str | Path | None = _EXPNAME_UNSET,
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
        need a persistent / resumable cache. It is required: pass a path or ``None`` — *unless*
        this context's ``cache_config.results`` is set, which is used as the default only when
        ``expname`` is omitted entirely. An explicit ``expname`` (including ``None``) always wins,
        so ``expname=None`` is still a throwaway temp dir even when ``cache_config.results`` is set.
        Extra ``**runner_kwargs`` (e.g. ``debug_mode``, ``cache_mode``) reach
        :class:`ExperimentBatchRunner`. Returns the raw per-split result dicts (also registered
        when ``register`` is True).
        """
        from tabarena.benchmark.experiment import ExperimentBatchRunner

        if not jobs:
            return []
        if expname is _EXPNAME_UNSET:
            # Omitted entirely: fall back to the cache_config default, else preserve the original
            # "expname is required" contract with a clear error (rather than a missing-arg TypeError).
            if self.cache_config is not None and self.cache_config.results is not None:
                expname = self.cache_config.results
            else:
                raise TypeError(
                    "run_jobs() is missing required keyword 'expname': pass a path (a persistent, "
                    "resumable results cache) or None (a throwaway temp dir). Alternatively, set "
                    "`cache_config.results` to provide a default expname.",
                )
        # `_cache_scope` points this process at the configured caches for the materialize + fit
        # (a distributed worker may have reconstructed this context). `materialize()` downloads
        # into the OpenML cache, so the scope must wrap it. With `cache_config.scope_openml=True`
        # the OpenML root is restored to its prior value once the run finishes.
        with self._cache_scope():
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
        expname: str | Path | None = _EXPNAME_UNSET,
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

    def load_model_results(
        self,
        methods: list[str] | None = None,
        *,
        configs: list[str] | None = None,
        download_results: str | bool = "auto",
    ) -> pd.DataFrame:
        """Load the raw per-config ``model_results`` of this arena's methods (downloading on miss).

        Unlike :meth:`load_results` — which returns each method's leaderboard-style rows (for a
        ``config`` method, the aggregated default / tuned / tuned+ensemble entries) — this returns
        the per-config rows from each method's ``model_results.parquet``: one row per
        ``(dataset, fold, config)`` carrying that individual config's own ``time_train_s`` /
        ``time_infer_s``. Useful for inspecting the configs a portfolio selected (whose ids live in
        the returned frame's ``method`` column, e.g. ``"CatBoost_r8_BAG_L1"``).

        Args:
            methods: methods to load (defaults to all of this context's methods).
            configs: if given, keep only rows whose config id (the ``method`` column) is in this
                set.
            download_results: ``"auto"`` (download on cache miss), ``True`` (download first), or
                ``False`` (never download — raise on a cache miss).
        """
        if methods is None:
            methods = self.methods
        if not methods:
            return pd.DataFrame()

        df_results_lst = []
        for method in methods:
            method_metadata = self.method_metadata(method=method)
            if isinstance(download_results, bool) and download_results:
                method_metadata.method_downloader().download_results()

            try:
                df_results = method_metadata.load_model_results()
            except FileNotFoundError as err:
                if isinstance(download_results, str) and download_results == "auto":
                    print(
                        f"Missing local model_results for method! Attempting to download from s3 "
                        f'and retry... (method="{method_metadata.method}")',
                    )
                    method_metadata.method_downloader().download_results()
                    df_results = method_metadata.load_model_results()
                else:
                    print(
                        f"Missing local model_results for method {method_metadata.method}! "
                        f"Try setting `download_results=True` to get the required files.",
                    )
                    raise err
            df_results_lst.append(df_results)

        df_model_results = pd.concat(df_results_lst, ignore_index=True)
        if configs is not None:
            df_model_results = df_model_results[df_model_results["method"].isin(set(configs))]
            df_model_results = df_model_results.reset_index(drop=True)
        return df_model_results

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
    def subset_shortcuts(self) -> dict[str, list[str]]:
        """Named shortcuts for standard subslices (shortcut name -> list of subset-predicate
        names AND-ed together). Reads from ``type(self).SUBSET_SHORTCUTS`` so subclass overrides
        take effect.
        """
        return type(self).SUBSET_SHORTCUTS

    @classmethod
    def subset_shortcut_name(cls, subset: str | list[str] | None) -> str | None:
        """Reverse of :attr:`subset_shortcuts`: the shortcut name whose predicate list
        matches ``subset`` (order-insensitive), or ``None`` if none does.

        ``subset`` takes the AND-list forms a single :class:`TaskSubset` view accepts — a lone
        predicate name (``"core"``), a list (``["core", "high-dim"]``), or ``None``/``[]`` — and is
        compared as a set, so ``["high-dim", "core"]`` still resolves to ``"high_dim"``.
        """
        if subset is None:
            subset = []
        elif isinstance(subset, str):
            subset = [subset]
        key = tuple(sorted(subset))
        for name, shortcut in cls.SUBSET_SHORTCUTS.items():
            if tuple(sorted(shortcut)) == key:
                return name
        return None

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
        suite: str | None = None,
    ) -> MethodMetadata:
        return self.method_metadata_collection.get_method_metadata(
            method=method,
            suite=suite,
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
                "suite": "ta_suite",
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
        plot_only: list[str] | None = None,
        method_color_overrides: dict[str, str] | None = None,
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

        ``plot_only`` (default ``None`` -> plot everything) restricts the *figures* to a subset of
        methods without changing any numbers: the leaderboard, Elo, win-rates and the saved
        ``tabarena_leaderboard.csv`` are always computed over the full method set, and only the
        plots (Elo bar plot, win-rate matrix, Pareto frontier, LaTeX table) are filtered down. Pass
        method *display names* — the long config display name for config methods (e.g.
        ``"LightGBM"``, ``"TabM"``) and the method name for baselines (e.g. ``"TabPFN-3"``). It
        composes with a ``hidden_methods`` denylist (anything hidden stays hidden).

        ``method_color_overrides`` (default ``None``) pins a fixed color per method family in the
        Pareto plots — a ``{display_name: color}`` map (e.g. ``{"TabPFN-3": "midnightblue"}``).
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
        # Plotting-only method allowlist (forwarded to the lower-level compare -> LeaderboardReporter
        # .eval, which translates it into the hidden_methods complement). Does not affect scoring.
        if plot_only is not None:
            kwargs["plot_only"] = plot_only
        # Per-method Pareto colors (also plotting-only); forwarded to eval the same way.
        if method_color_overrides is not None:
            kwargs["method_color_overrides"] = method_color_overrides
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
        #  Pair with (method, suite)
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
                tabarena_context=self,
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
                benchmark_name=self.benchmark_name,
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
        save_composite_leaderboard: bool = False,
        composite_leaderboard_kwargs: dict | None = None,
        trajectory_extra_results: pd.DataFrame | None = None,
        max_workers: int = 1,
    ) -> None:
        """Generate the compare / tuning-trajectory / runtime figures for each subset.

        ``new_results`` are the new methods compared in the leaderboard (``plot_compare``) and, by
        default, also the ``extra_results`` overlaid on the tuning-trajectory plots
        (``plot_tuning_trajectories``). Pass ``trajectory_extra_results`` to give the trajectory
        pass a *different* frame than the compare pass — e.g. show full per-config trajectories
        (multiple ``n_configs`` rows per method) on the trajectory plot while the leaderboard keeps
        the single-point method results — in one call instead of two.

        ``save_composite_leaderboard=True`` additionally aggregates the per-subset leaderboards
        into a single composite table (rows = (method, metric), one column per subset) written to
        ``<output_dir>/composite_leaderboard.csv`` plus color-graded PNG renderings — see
        :func:`tabarena.plot.composite_leaderboard.generate_composite_leaderboard`, which
        ``composite_leaderboard_kwargs`` is forwarded to (e.g. ``sort_by``, ``top_n``,
        ``excluded_method_prefixes``, ``title``). Requires ``plot_compare=True``; the composite is
        built from the compact website format independently of ``save_website_leaderboard``.

        ``max_workers`` parallelizes the per-subset passes across processes (capped at the subset
        count; 1 = sequential). Each subset's compare/plots are independent and write to their own
        directory, so they compose freely; processes (not threads) because the passes are CPU-bound
        (Elo bootstrap, plotting) and matplotlib is not thread-safe. Requires the context (and any
        results frames passed in) to be picklable.
        """
        if compare_kwargs is None:
            compare_kwargs = {}
        if tuning_trajectory_kwargs is None:
            tuning_trajectory_kwargs = {}
        if website_leaderboard_kwargs is None:
            website_leaderboard_kwargs = {}
        if save_composite_leaderboard and not plot_compare:
            raise ValueError(
                "save_composite_leaderboard=True requires plot_compare=True: the composite is "
                "aggregated from the per-subset leaderboards computed by the compare pass.",
            )
        if subsets == "auto":
            subsets = self._default_subsets
        output_dir = Path(output_dir)

        subset_fig_kwargs = dict(
            output_dir=output_dir,
            new_results=new_results,
            compare_kwargs=compare_kwargs,
            tuning_trajectory_kwargs=tuning_trajectory_kwargs,
            plot_compare=plot_compare,
            plot_runtime_per_method=plot_runtime_per_method,
            plot_tuning_trajectories=plot_tuning_trajectories,
            save_website_leaderboard=save_website_leaderboard,
            website_leaderboard_kwargs=website_leaderboard_kwargs,
            website_leaderboard_filename=website_leaderboard_filename,
            collect_composite=save_composite_leaderboard,
            trajectory_extra_results=trajectory_extra_results,
        )
        max_workers = max(1, min(max_workers, len(subsets)))
        if max_workers == 1:
            subset_results = [self._generate_subset_figs(subset, **subset_fig_kwargs) for subset in subsets]
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                subset_results = list(
                    executor.map(functools.partial(self._generate_subset_figs, **subset_fig_kwargs), subsets)
                )

        # Per-subset compact website-format leaderboards collected for the composite,
        # keyed by the subset's output suffix (= its column name in the composite).
        composite_inputs: dict[str, pd.DataFrame] = {
            output_suffix: lb_compact for output_suffix, lb_compact in subset_results if lb_compact is not None
        }
        if save_composite_leaderboard and composite_inputs:
            from tabarena.plot.composite_leaderboard import generate_composite_leaderboard

            generate_composite_leaderboard(
                leaderboards=composite_inputs,
                output_dir=output_dir,
                **(composite_leaderboard_kwargs or {}),
            )

    def _generate_subset_figs(
        self,
        subset: list[str] | tuple[str, list[str]] | str | None,
        *,
        output_dir: Path,
        new_results,
        compare_kwargs: dict,
        tuning_trajectory_kwargs: dict,
        plot_compare: bool,
        plot_runtime_per_method: bool,
        plot_tuning_trajectories: bool,
        save_website_leaderboard: bool,
        website_leaderboard_kwargs: dict,
        website_leaderboard_filename: str,
        collect_composite: bool,
        trajectory_extra_results: pd.DataFrame | None,
    ) -> tuple[str, pd.DataFrame | None]:
        """One subset's pass of :meth:`generate_all_figs` (compare + figures under
        ``output_dir/<suffix>/``). Returns ``(output_suffix, compact_leaderboard_or_None)`` —
        the subset's composite-leaderboard input when ``collect_composite`` and the compare
        pass produced a leaderboard. Self-contained so subsets can run in worker processes.
        """
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

        lb_compact = None
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
            if collect_composite and lb_df is not None:
                # Always the compact format — the composite reads its metric
                # columns (Elo / Impro%) from it, regardless of what format
                # `website_leaderboard_kwargs` chose for the saved CSVs.
                lb_compact = self.leaderboard_to_website_format(
                    leaderboard=lb_df,
                    compact=True,
                )
        if plot_tuning_trajectories:
            self.plot_tuning_trajectories(
                save_path=output_dir_subset / "tuning_trajectories",
                subset=subset,
                extra_results=trajectory_extra_results if trajectory_extra_results is not None else new_results,
                **tuning_trajectory_kwargs,
            )
        if plot_runtime_per_method:
            self.plot_runtime_per_method(
                save_path=output_dir_subset / "ablation" / "all-runtimes",
                # new_results=new_results,
                subset=subset,
            )
        return output_suffix, lb_compact

    def load_repo(
        self,
        methods: list[str | MethodMetadata] | None = None,
        config_fallback: str | None = None,
        max_workers: int | None = 16,
    ) -> EvaluationRepositoryCollection:
        """Load each method's processed artifacts and combine them into a collection.

        Methods load in a thread pool of ``max_workers`` (capped at the method count; pass
        ``None`` or ``1`` to load sequentially). Loading is dominated by per-file filesystem
        round trips — predictions are memmapped, not read — so overlapping the I/O across
        methods gives a near-linear speedup on network filesystems (e.g. NFS).
        """
        if methods is None:
            methods = self.methods
        metadatas = [
            method if isinstance(method, MethodMetadata) else self.method_metadata(method=method) for method in methods
        ]
        max_workers = max(1, min(max_workers if max_workers is not None else 1, len(metadatas)))
        if max_workers == 1:
            repos = [metadata.load_processed() for metadata in metadatas]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                repos = list(executor.map(lambda metadata: metadata.load_processed(), metadatas))
        return EvaluationRepositoryCollection(repos=repos, config_fallback=config_fallback)

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
        hpo_trajectory = MethodSimulator(default_method).generate_hpo_trajectories(
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

    def generate_portfolio_trajectories_per(
        self,
        portfolios: dict[str, list[str]],
        *,
        name: str,
        config_fallback: str | None = None,
        n_configs: list[int | None] | str = "auto",
        seeds: int | list[int] = 1,
        n_iterations: int = 40,
        fit_order: Literal["original", "random"] = "original",
        time_limit: float | None = None,
        repo: EvaluationRepository | None = None,
        folds: list[int] | None = None,
        ta_name: str | None = None,
        ta_suite: str | None = None,
        display_name: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Per-dataset analogue of :meth:`generate_portfolio_trajectories`.

        Where :meth:`generate_portfolio_trajectories` takes a single global ordered ``configs``
        list, this takes a per-dataset mapping ``portfolios`` (dataset -> ordered config list, e.g.
        the leave-one-out portfolio each dataset selected). For each ``N`` in ``n_configs`` it
        evaluates every dataset's *first ``N``* configs as an ensemble (via
        :meth:`simulate_portfolio_from_configs_per`) and tags the result with ``n_configs=N``.

        All ``N`` share the same ``method`` (== ``name``), so the rows form a single tuning
        trajectory under the ``extra_results`` convention consumed by ``plot_tuning_trajectories``
        (each point ensembles a prefix of the order — first config, first two, ...). Because the
        ensemble's ``time_train_s`` is the cumulative cost of the considered configs, the trajectory
        directly reflects the config *ordering* — which is what makes two orderings of the same set
        (identical final point, different anytime curve) comparable.
        """
        if n_configs == "auto":
            n_configs = [1, 2, 5, 10, 25, 50, 100, 150, None]
        if isinstance(seeds, int):
            seeds = list(range(seeds))

        datasets = list(portfolios)
        all_configs = sorted({config for configs in portfolios.values() for config in configs})

        if repo is None:
            repo = self.load_repo(config_fallback=config_fallback)

        configs_w_fallback = list(all_configs)
        if config_fallback is not None:
            if config_fallback not in configs_w_fallback:
                configs_w_fallback.append(config_fallback)
            repo.set_config_fallback(config_fallback=config_fallback)
        repo = repo.subset(configs=configs_w_fallback, datasets=datasets, folds=folds)

        max_n_configs = max(len(configs) for configs in portfolios.values())
        n_configs = [n_config if n_config is not None else max_n_configs for n_config in n_configs]
        n_configs = sorted({min(n_config, max_n_configs) for n_config in n_configs})

        # The (dataset, fold) tasks to evaluate, taken from the (subset) repo.
        dataset_folds = [(dataset, fold) for dataset in datasets for fold in repo.dataset_to_folds(dataset)]

        df_trajectory_lst = []
        for n_config in n_configs:
            print(f"Running n_config={n_config}")
            for seed in seeds:
                df_info = pd.DataFrame(
                    [
                        {"dataset": dataset, "fold": fold, "configs": portfolios[dataset][:n_config]}
                        for dataset, fold in dataset_folds
                    ]
                )
                df_results = self.simulate_portfolio_from_configs_per(
                    df_info=df_info,
                    repo=repo,
                    n_iterations=n_iterations,
                    fit_order=fit_order,
                    seed=seed,
                    time_limit=time_limit,
                    **kwargs,
                )
                df_results["n_configs"] = n_config
                df_results["n_iterations"] = n_iterations
                df_results["seed"] = seed
                df_trajectory_lst.append(df_results)

        trajectory = pd.concat(df_trajectory_lst, ignore_index=True)
        trajectory["method"] = name
        trajectory["ta_name"] = ta_name
        trajectory["ta_suite"] = ta_suite
        trajectory["display_name"] = display_name
        return trajectory

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
    ) -> InMemoryMethodMetadata:
        """Perform HPO across multiple methods, pooling all of their configs into one family.

        ``methods`` may be registered method names (resolved to their ``config_type``) and/or
        raw ``config_type`` strings; ``run_hpo`` selects the configs of every named family from
        ``repo`` (defaulting to ``self.load_repo(methods=methods)``) and tunes/ensembles over the
        union. ``method_default`` (the first method if unset) supplies the always-first default
        config.

        Returns the pooled family as an :class:`InMemoryMethodMetadata` (method/config_type
        ``new_config_type``, suite ``ta_suite``) whose in-memory results hold the default / tuned
        / tuned+ensemble rows. Register it via ``extra_methods=`` / :meth:`register` so it flows
        through :meth:`compare` and the leaderboard like any method (e.g.
        ``ContextCls(extra_methods=[combined], only_valid_tasks=True).compare(...)``); call
        ``.load_results()`` for the raw frame.
        """
        if method_default is None:
            method_default = methods[0]
        if repo is None:
            repo = self.load_repo(methods=methods)

        config_type_default = self.method_metadata(method_default).config_type
        simulator = RepoSimulator(repo=repo, backend=self.backend)
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

        results = pd.concat(
            [
                default,
                tuned,
                tuned_ens,
            ],
            ignore_index=True,
        )
        # `method_metadata` is a per-row dict of run knobs (n_iterations/n_configs/...) that the
        # leaderboard never reads and parquet can't serialize; drop it so the family's results
        # frame is a clean, persistable results artifact.
        results = results.drop(columns=["method_metadata"], errors="ignore")
        # No repo is attached: the pooled family's configs belong to the *constituent* families
        # (the config_types in `methods`), not to `new_config_type`, so there is no coherent
        # processed repo for it — it is a results-only method. Attaching the merged repo would
        # mislead repo-backed MethodMetadata logic (e.g. MethodSimulator.get_config_default keyed
        # on this family's config_type, which the merged repo does not contain).
        return InMemoryMethodMetadata.from_results_df(
            results,
            method=new_config_type,
            suite=ta_suite,
            config_type=new_config_type,
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
        simulator = RepoSimulator(repo=repo, backend=self.backend)
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
            configs = [
                MethodSimulator(m).get_config_default()
                for m in self.method_metadata_collection.method_metadata_lst
                if m.method_type == "config"
            ]

        simulator = RepoSimulator(repo=repo, backend=self.backend)
        simulator.repo_metrics.compute_avg_config_prediction_delta(configs=configs)

    def simulate_portfolio_from_configs(
        self,
        configs: list[str],
        config_fallback: str | None = None,
        repo: EvaluationRepositoryCollection = None,
        **kwargs,
    ):
        if repo is None:
            repo = self.load_repo(config_fallback=config_fallback)
        simulator = RepoSimulator(repo=repo, backend=self.backend)

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
        simulator = RepoSimulator(repo=repo, backend=self.backend)

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
        simulator = RepoSimulator(repo=repo, backend=self.backend)

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
        simulator = RepoSimulator(repo=repo, backend=self.backend)
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
        simulator = RepoSimulator(repo=repo, backend=self.backend)

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
        simulator = RepoSimulator(repo=repo, backend=self.backend)
        return simulator.run_zs_from_types(
            config_types=config_types,
            n_portfolios=n_portfolio,
            n_ensemble=n_ensemble,
            n_ensemble_in_name=True,
            time_limit=time_limit,
            **kwargs,
        )

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

        evaluator = LeaderboardReporter(output_dir=save_path, task_metadata=self.task_metadata_collection)
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

    @classmethod
    def fillna_metrics(cls, df_to_fill: pd.DataFrame, df_fillna: pd.DataFrame) -> pd.DataFrame:
        """Fills missing (dataset, fold, method) rows in df_to_fill with the (dataset, fold) row in df_fillna.

        Thin wrapper over :func:`tabarena.evaluation._fillna.fillna_metrics` (keyed on ``method``),
        preserving the per-method descriptive columns (``method_type`` / ``method_subtype`` /
        ``config_type`` / ``ta_name`` / ``ta_suite``) so an imputed row keeps its own identity
        rather than the fallback method's.

        Parameters
        ----------
        df_to_fill
        df_fillna

        Returns:
        -------

        """
        from tabarena.evaluation._fillna import fillna_metrics

        return fillna_metrics(
            df_to_fill=df_to_fill,
            df_fillna=df_fillna,
            key_col="method",
            preserve_columns=["method_type", "method_subtype", "config_type", "ta_name", "ta_suite"],
        )
