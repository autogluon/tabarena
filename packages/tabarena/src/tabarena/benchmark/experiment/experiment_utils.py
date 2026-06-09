from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from tabarena.utils.cache import AbstractCacheFunction, CacheFunctionDummy, CacheFunctionPickle

if TYPE_CHECKING:
    from tabarena.benchmark.experiment.experiment_constructor import Experiment
    from tabarena.benchmark.experiment.job import Job
    from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection
    from tabarena.benchmark.task.user_task import UserTask


# TODO: Inspect artifact folder to load all results without needing to specify them explicitly
#  generate_repo_from_dir(expname)
class ExperimentBatchRunner:
    def __init__(
        self,
        expname: str,
        task_metadata: TaskMetadataCollection,
        dataset_fold_repeats: list[tuple[str, int, int]] | None = None,
        cache_cls: type[AbstractCacheFunction] | None = CacheFunctionPickle,
        cache_cls_kwargs: dict | None = None,
        cache_mode: Literal["default", "ignore", "only", "only_strict"] = "default",
        debug_mode: bool = True,
        raise_on_failure: bool = True,
        user_tasks: list[UserTask] | None = None,
    ):
        """Parameters
        ----------
        expname
        task_metadata: TaskMetadataCollection
            The native task metadata. A legacy one-row-per-dataset DataFrame is no longer
            accepted — wrap it first with ``TaskMetadataCollection.from_legacy_df(df)`` (or
            build one with ``TaskMetadataCollection(tasks)``) so the lossy legacy conversion
            stays an explicit, opt-in step at the call site.
        dataset_fold_repeats: list[tuple[str, int, int]] | None, default None
            The allowed (dataset, fold, repeat) triplets that `run_all` executes. If None,
            `run_all` uses the collection's actual splits (a non-rectangular set is
            respected). Each triplet is validated against the collection's splits.
        cache_cls
        cache_cls_kwargs
        cache_mode: {"default", "ignore", "only", "only_strict"}, default "default"
            How to handle the experiment cache:
                - "default": skip an experiment if its cache exists, otherwise run it.
                - "ignore": always run the experiment, overwriting any existing cache.
                - "only": only load results from cache; never run an experiment, and
                    silently skip experiments whose cache is missing.
                - "only_strict": like "only", but raise if any requested experiment is
                    missing from the cache.
        debug_mode: bool, default True
            If True, will operate in a manner best suited for local model development.
            This mode will be friendly to local debuggers and will avoid subprocesses/threads
            and complex try/except logic.

            IF False, will operate in a manner best suited for large-scale benchmarking.
            This mode will try to record information when method's fail
            and might not work well with local debuggers.
        raise_on_failure: bool, default True
            If True, exceptions raised during an experiment propagate and stop the run.
            If False, failures are recorded and the remaining experiments continue.
        user_tasks: list[UserTask] | None, default None
            Local (custom) tasks to register so the run methods can execute them. Each
            ``UserTask`` is keyed by its ``tabarena_task_name`` (which must also appear as a
            ``dataset`` row in ``task_metadata``, with ``tid == UserTask.task_id``). When a
            run resolves such a dataset it hands the live ``UserTask`` to
            ``run_experiments_new`` (loaded from local disk via ``save_local_openml_task``),
            instead of the integer tid that ``run_experiments_new`` would try to download
            from OpenML. Datasets without a registered ``UserTask`` keep resolving to their
            integer OpenML tid.
        """
        cache_cls = CacheFunctionDummy if cache_cls is None else cache_cls
        cache_cls_kwargs = {"include_self_in_call": True} if cache_cls_kwargs is None else cache_cls_kwargs

        # The TaskMetadataCollection is the single source of truth: the tid map / grid /
        # validation are derived from it natively. A legacy DataFrame is no longer accepted.
        from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection

        if not isinstance(task_metadata, TaskMetadataCollection):
            raise TypeError(
                f"`task_metadata` must be a TaskMetadataCollection, got {type(task_metadata).__name__}. "
                "Wrap a legacy DataFrame with TaskMetadataCollection.from_legacy_df(df) (or build one "
                "with TaskMetadataCollection(tasks)).",
            )
        self.task_metadata_collection = task_metadata
        # Legacy DataFrame view, derived from the collection, for the downstream
        # `run_experiments_new` `tasks_metadata` passthrough.
        self.task_metadata = task_metadata.to_legacy_df()

        self.expname = expname
        self.cache_cls = cache_cls
        self.cache_cls_kwargs = cache_cls_kwargs
        self.cache_mode = cache_mode
        self._dataset_to_tid_dict = self._build_dataset_to_tid()
        self._dataset_to_user_task = self._build_dataset_to_user_task(user_tasks)
        self.debug_mode = debug_mode
        self.raise_on_failure = raise_on_failure
        if dataset_fold_repeats is not None:
            self._validate_dataset_fold_repeats(dataset_fold_repeats)
        self._dataset_fold_repeats = dataset_fold_repeats

    @property
    def datasets(self) -> list[str]:
        return list(self._dataset_to_tid_dict.keys())

    def _build_dataset_to_tid(self) -> dict[str, int]:
        """Map dataset name -> integer tid, parsed natively from the collection's ``task_id_str``."""
        return {dataset: int(tid) for dataset, tid in self.task_metadata_collection.dataset_to_tid().items()}

    def _build_dataset_to_user_task(self, user_tasks: list[UserTask] | None) -> dict[str, UserTask]:
        """Map dataset name -> registered local ``UserTask`` (see the ``user_tasks`` arg).

        Each task is keyed by its ``tabarena_task_name`` (the ``dataset`` column key) and must
        already be present in the tid map, so ``task_metadata`` and the registered tasks stay
        consistent.
        """
        if not user_tasks:
            return {}
        mapping: dict[str, UserTask] = {}
        for task in user_tasks:
            dataset = task.tabarena_task_name
            if dataset not in self._dataset_to_tid_dict:
                raise ValueError(
                    f"Registered user task {dataset!r} is not present in `task_metadata`; add a "
                    f"row for it (its `tid` must equal `UserTask.task_id`).",
                )
            mapping[dataset] = task
        return mapping

    def _resolve_task(self, dataset: str) -> int | UserTask:
        """Resolve a dataset name to what ``run_experiments_new`` should run for it.

        A registered local task resolves to its live ``UserTask`` (loaded from local disk);
        any other dataset resolves to its integer OpenML tid (downloaded on demand).
        """
        user_task = self._dataset_to_user_task.get(dataset)
        if user_task is not None:
            return user_task
        return int(self._dataset_to_tid_dict[dataset])

    def _full_dataset_fold_repeats(self) -> list[tuple[str, int, int]]:
        """The full ``(dataset, fold, repeat)`` grid `run_all` executes when no explicit
        triplets were given: the collection's actual splits (a non-rectangular set is respected).
        """
        return self.task_metadata_collection.dataset_fold_repeats()

    def run(
        self,
        methods: list[Experiment],
        datasets: list[str],
        folds: list[int],
        repeats: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Parameters
        ----------

        Methods:
        datasets
        folds
        repeats

        Returns:
        -------
        results_lst: list[dict[str, Any]]
            A list of experiment run metadata dictionaries.
            Can pass into `exp_bach_runner.repo_from_results(results_lst=results_lst)` to generate an EvaluationRepository.

        """
        self._validate_datasets(datasets=datasets)
        self._validate_folds(folds=folds)
        self._validate_repeats(repeats=repeats)

        tasks = [self._resolve_task(dataset) for dataset in datasets]

        # Translate folds/repeats into explicit (fold, repeat) pairs run for every task.
        # When `repeats` is unspecified, default to repeat 0. The cache path always
        # includes the repeat (`{repeat}_{fold}`).
        if repeats is None:
            fold_repeat_pairs = [(fold, 0) for fold in folds]
        else:
            fold_repeat_pairs = [(fold, repeat) for repeat in repeats for fold in folds]

        return self._run_individual(
            methods=methods,
            tasks=tasks,
            repetitions_mode_args=fold_repeat_pairs,
        )

    def run_dataset_fold_repeats(
        self,
        methods: list[Experiment],
        dataset_fold_repeats: list[tuple[str, int, int]],
    ) -> list[dict[str, Any]]:
        """Run an explicit list of (dataset, fold, repeat) tasks.

        Unlike `run`, which runs the cartesian product of `datasets` x `folds` x
        `repeats`, this runs exactly the (dataset, fold, repeat) triples provided.

        Parameters
        ----------

        Methods:
        dataset_fold_repeats: list[tuple[str, int, int]]
            The (dataset, fold, repeat) triples to run. Must not contain duplicates.

        Returns:
        -------
        results_lst: list[dict[str, Any]]
            A list of experiment run metadata dictionaries.
            Can pass into `exp_bach_runner.repo_from_results(results_lst=results_lst)` to generate an EvaluationRepository.

        """
        if len(dataset_fold_repeats) != len(set(dataset_fold_repeats)):
            raise AssertionError("Duplicate (dataset, fold, repeat) triples present! Ensure all triples are unique.")

        # Group the (fold, repeat) pairs by dataset, preserving first-seen order, so each
        # task gets its own list of pairs (the per-task "individual" arg format).
        pairs_per_dataset: dict[str, list[tuple[int, int]]] = {}
        for dataset, fold, repeat in dataset_fold_repeats:
            pairs_per_dataset.setdefault(dataset, []).append((fold, repeat))

        datasets = list(pairs_per_dataset.keys())
        self._validate_datasets(datasets=datasets)

        tasks = [self._resolve_task(dataset) for dataset in datasets]
        repetitions_mode_args = [pairs_per_dataset[dataset] for dataset in datasets]

        return self._run_individual(
            methods=methods,
            tasks=tasks,
            repetitions_mode_args=repetitions_mode_args,
        )

    def run_all(
        self,
        methods: list[Experiment],
    ) -> list[dict[str, Any]]:
        """Run the configured (dataset, fold, repeat) triplets.

        If `dataset_fold_repeats` was passed at init, runs exactly those triplets.
        Otherwise, for each dataset runs all folds in `range(n_folds)` x repeats in
        `range(n_repeats)`, taken from the `n_folds` and `n_repeats` columns of
        `task_metadata`. Delegates to `run_dataset_fold_repeats`.

        Parameters
        ----------

        Methods:

        Returns:
        -------
        results_lst: list[dict[str, Any]]
            A list of experiment run metadata dictionaries.
            Can pass into `exp_bach_runner.repo_from_results(results_lst=results_lst)` to generate an EvaluationRepository.

        """
        dataset_fold_repeats = self._dataset_fold_repeats
        if dataset_fold_repeats is None:
            dataset_fold_repeats = self._full_dataset_fold_repeats()

        return self.run_dataset_fold_repeats(
            methods=methods,
            dataset_fold_repeats=dataset_fold_repeats,
        )

    def run_jobs(
        self,
        jobs: list[Job],
    ) -> list[dict[str, Any]]:
        """Run an explicit, possibly non-rectangular list of ``(experiment, task)`` jobs.

        Unlike `run` / `run_dataset_fold_repeats`, which cross a single `methods` list with
        every task, each `Job` names *both* its experiment and its `(dataset, fold, repeat)`
        split. Different experiments may therefore run on different tasks — e.g. re-running
        only the specific (method, task, fold) units that failed, or running a heavy model
        on a subset of datasets.

        The jobs are resolved to tid-keyed work units and dispatched in a single pass that
        groups by task, so a task shared by several experiments is materialized (OpenML
        load) once *overall* (not once per experiment) and only one dataset is resident in
        memory at a time. Caching and validation match the other entry points; the returned
        `results_lst` preserves the order of `jobs` (failed or skipped jobs simply drop out).

        Parameters
        ----------
        jobs: list[Job]
            The (experiment, task) units to run. Must be unique on
            `(experiment.name, dataset, fold, repeat)`.

        Returns:
        -------
        results_lst: list[dict[str, Any]]
            A list of experiment run metadata dictionaries (see `run`), in the same order as
            `jobs` (minus any that failed or were skipped).
            Can pass into `exp_bach_runner.repo_from_results(results_lst=results_lst)` to generate an EvaluationRepository.

        """
        if not jobs:
            return []

        # Validate the two ways a job list can be malformed before running anything: a
        # duplicated unit of work, and two *different* experiments sharing a name. The
        # results cache is keyed by `experiment.name`, so a name collision between distinct
        # configs would silently cross-contaminate cached results.
        experiments_by_name: dict[str, Experiment] = {}
        seen: set[tuple[str, str, int, int]] = set()
        datasets: list[str] = []
        for job in jobs:
            experiment = job.experiment
            name = experiment.name
            dataset, fold, repeat = job.task.as_triple()

            existing = experiments_by_name.get(name)
            if existing is None:
                experiments_by_name[name] = experiment
            elif existing is not experiment and existing.to_yaml_str() != experiment.to_yaml_str():
                raise AssertionError(
                    f"Two different experiments share the name {name!r}; experiment names must be "
                    f"unique because the results cache is keyed by name.",
                )

            key = (name, dataset, fold, repeat)
            if key in seen:
                raise AssertionError(
                    f"Duplicate job: experiment={name!r}, (dataset, fold, repeat)={(dataset, fold, repeat)}.",
                )
            seen.add(key)
            if dataset not in datasets:
                datasets.append(dataset)

        self._validate_datasets(datasets=datasets)

        # Resolve dataset name -> tid and build one flat, tid-keyed spec list in `jobs`
        # order. `_run_job_specs` executes grouped by task (one load per shared task) but
        # returns results in this input order, so `results_lst` aligns with `jobs`.
        # Lazy import to avoid a circular import (experiment_runner_api imports this module).
        from tabarena.benchmark.experiment.experiment_runner_api import _JobSpec, _run_job_specs

        job_specs = [
            _JobSpec(
                model_experiment=job.experiment,
                task=self._resolve_task(job.task.dataset),
                fold=job.task.fold,
                repeat=job.task.repeat,
            )
            for job in jobs
        ]
        return _run_job_specs(
            job_specs,
            base_cache_path=self.expname,
            cache_mode=self.cache_mode,
            cache_cls=self.cache_cls,
            cache_cls_kwargs=self.cache_cls_kwargs,
            raise_on_failure=self.raise_on_failure,
            debug_mode=self.debug_mode,
        )

    def _run_individual(
        self,
        methods: list[Experiment],
        tasks: list[int | UserTask],
        repetitions_mode_args: list,
    ) -> list[dict[str, Any]]:
        """Invoke `run_experiments_new` in 'individual' repetitions mode.

        `repetitions_mode_args` is either a single list of (fold, repeat) pairs applied
        to every task, or one (fold, repeat) list per task (aligned with `tasks`). Each task
        is an integer OpenML tid or a local ``UserTask`` (see `_resolve_task`). See
        `run_experiments_new` for the 'individual' arg format.
        """
        # Lazy import to avoid a circular import (experiment_runner_api imports this module).
        from tabarena.benchmark.experiment.experiment_runner_api import run_experiments_new

        return run_experiments_new(
            output_dir=self.expname,
            model_experiments=methods,
            tasks=tasks,
            tasks_metadata=self.task_metadata,
            repetitions_mode="individual",
            repetitions_mode_args=repetitions_mode_args,
            cache_mode=self.cache_mode,
            # Forward the configured cache backend. The default `cache_cls_kwargs`
            # carries `include_self_in_call=True`, preserving the legacy
            # `model_failures` artifact on failure.
            cache_cls=self.cache_cls,
            cache_cls_kwargs=self.cache_cls_kwargs,
            raise_on_failure=self.raise_on_failure,
            debug_mode=self.debug_mode,
        )

    @classmethod
    def _subtask_name(cls, fold: int, repeat: int | None = None) -> str:
        return f"{fold}" if repeat is None else f"{repeat}_{fold}"

    def _validate_datasets(self, datasets: list[str]):
        unknown_datasets = []
        for dataset in datasets:
            if dataset not in self._dataset_to_tid_dict:
                unknown_datasets.append(dataset)
        if unknown_datasets:
            raise ValueError(
                f"Dataset must be present in task_metadata!"
                f"\n\tInvalid Datasets: {unknown_datasets}"
                f"\n\t  Valid Datasets: {self.datasets}",
            )
        if len(datasets) != len(set(datasets)):
            raise AssertionError("Duplicate datasets present! Ensure all datasets are unique.")

    def _validate_folds(self, folds: list[int]):
        if len(folds) != len(set(folds)):
            raise AssertionError("Duplicate folds present! Ensure all folds are unique.")

    def _validate_repeats(self, repeats: list[int] | None):
        if repeats is None:
            return
        if len(repeats) != len(set(repeats)):
            raise AssertionError("Duplicate repeats present! Ensure all repeats are unique.")

    def _validate_dataset_fold_repeats(self, dataset_fold_repeats: list[tuple[str, int, int]]):
        """Verify each (dataset, fold, repeat) is a real split of the collection.

        Validated against the collection's actual splits, so a non-rectangular set of splits
        is respected.
        """
        valid = set(self.task_metadata_collection.dataset_fold_repeats())
        invalid = [
            (
                dataset,
                fold,
                repeat,
                "unknown dataset" if dataset not in self._dataset_to_tid_dict else "no such (fold, repeat) split",
            )
            for dataset, fold, repeat in dataset_fold_repeats
            if (dataset, fold, repeat) not in valid
        ]
        if invalid:
            invalid_str = "\n\t".join(str(x) for x in invalid)
            raise AssertionError(
                f"`dataset_fold_repeats` contains {len(invalid)} entry(ies) not valid for `task_metadata`:\n\t{invalid_str}",
            )


def check_cache_hit(
    *,
    result_dir: str,
    method_name: str,
    task_id: int,
    fold: int,
    repeat: int | None,
    cache_cls: type[AbstractCacheFunction] | None,
    cache_cls_kwargs: dict | None = None,
    delete_cache: bool = False,
) -> bool:
    """Returns true if cache exists for the given experiment."""
    base_cache_path = result_dir

    subtask_cache_name = ExperimentBatchRunner._subtask_name(fold=fold, repeat=repeat)

    cache_prefix = f"data/{method_name}/{task_id}/{subtask_cache_name}"
    cache_name = "results"

    cache_path = f"{base_cache_path}/{cache_prefix}"

    cacher = cache_cls(cache_name=cache_name, cache_path=cache_path, **cache_cls_kwargs)

    if delete_cache:
        Path(cacher.cache_file).unlink(missing_ok=True)

    return cacher.exists
