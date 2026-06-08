from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from tabarena.benchmark.experiment import Experiment, ExperimentBatchRunner
from tabarena.benchmark.task.openml import OpenMLTaskWrapper
from tabarena.benchmark.task.user_task import UserTask
from tabarena.utils.cache import AbstractCacheFunction, CacheFunctionPickle

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import pandas as pd


def _clean_repetitions_mode_args_for_matrix(
    repetitions_mode_args: tuple,
) -> tuple[list[int], list[int]]:
    # Ensure input is a tuple of two elements
    assert isinstance(repetitions_mode_args, tuple), "Input must be tuple!"
    assert len(repetitions_mode_args) == 2, (
        "If `repetitions_mode_args` for 'matrix' is a tuple, it must contain two elements: (folds, repeats)"
    )

    a, b = repetitions_mode_args

    # If both are ints -> convert to ranges
    if isinstance(a, int) and isinstance(b, int):
        return (list(range(a)), list(range(b)))

    # If one is int and other not, error
    if isinstance(a, int) or isinstance(b, int):
        raise AssertionError(
            "If `repetitions_mode_args` for 'matrix' is a tuple with integers, both elements must be integers.",
        )

    # Now both must be lists of ints
    assert isinstance(a, list) and isinstance(b, list), (  # noqa: PT018
        "If `repetitions_mode_args` for 'matrix' is a tuple with lists, both elements must be a list."
    )
    assert (len(a) > 0) and all(isinstance(x, int) for x in a), (  # noqa: PT018
        "If `repetitions_mode_args` for 'matrix' is a tuple with lists, the first list must contain at least one integer for folds."
    )
    assert (len(b) > 0) and all(isinstance(x, int) for x in b), (  # noqa: PT018
        "If `repetitions_mode_args` for 'matrix' is a tuple with lists, the second list must contain at least one integer for repeats."
    )

    return (a, b)


def _parse_repetitions_mode_and_args(
    *,
    repetitions_mode: Literal["TabArena-Lite", "TabArena", "matrix", "individual"],
    repetitions_mode_args: tuple | list | None,
    tasks: list[int | UserTask],
    tasks_metadata: pd.DataFrame | None = None,
) -> list[list[tuple[int, int]]]:
    """Parse the `repetitions_mode` and `repetitions_mode_args` parameters to determine
    which folds and repeats to run per dataset.

    Returns a standardized format: a list of elements, where each element corresponds
    to the repetitions to run for the task; where each element is a list of tuples,
    each tuple represents a fold-repeat pair; and where each tuple contains two
    integers, the first one is the fold index second one is the repeat index.
    """
    if repetitions_mode == "TabArena":
        if tasks_metadata is None:
            from tabarena.nips2025_utils.fetch_metadata import (
                load_curated_task_metadata,
            )

            tasks_metadata = load_curated_task_metadata()
        else:
            # Verify user metadata
            req_columns = [
                "tabarena_num_repeats",
                "num_folds",
                "task_id",
            ]
            for col in req_columns:
                assert col in tasks_metadata.columns, (
                    f"`tasks_metadata` must contain the column '{col}' when `repetitions_mode` is 'TabArena'"
                )
        fold_repeat_pairs_per_task = []
        metadata_task_ids = tasks_metadata["task_id"].astype(str)
        metadata_task_id_set = set(metadata_task_ids.tolist())
        for task in tasks:
            t_id = task.task_id_str if isinstance(task, UserTask) else str(task)
            assert t_id in metadata_task_id_set, f"Task ID '{t_id}' from `tasks` not found in `tasks_metadata`"

            task_meta = tasks_metadata[metadata_task_ids == t_id].iloc[0]
            n_folds = int(task_meta["num_folds"])
            n_repeats = int(task_meta["tabarena_num_repeats"])
            fold_repeat_pairs = [(f, r) for r in range(n_repeats) for f in range(n_folds)]
            fold_repeat_pairs_per_task.append(fold_repeat_pairs)
        return fold_repeat_pairs_per_task

    if repetitions_mode == "TabArena-Lite":
        # Run only the first fold of the first repeat for each task
        return [[(0, 0)] for _ in range(len(tasks))]

    if repetitions_mode == "matrix":
        assert repetitions_mode_args is not None, (
            "If `repetitions_mode` is 'matrix', `repetitions_mode_args` must be provided"
        )
        if isinstance(repetitions_mode_args, list):
            assert len(repetitions_mode_args) == len(tasks), (
                "If `repetitions_mode_args` for 'matrix' is a list, it must have the same length as `tasks`"
            )
            assert all(isinstance(rep, tuple) for rep in repetitions_mode_args), (
                "If `repetitions_mode_args` for 'matrix' is a list, all elements must be tuples"
            )
            repetitions_mode_args = [_clean_repetitions_mode_args_for_matrix(rep) for rep in repetitions_mode_args]
        else:
            assert isinstance(repetitions_mode_args, tuple), (
                "If `repetitions_mode_args` for 'matrix' is not a list, it must be a tuple"
            )
            repetitions_mode_args = [_clean_repetitions_mode_args_for_matrix(repetitions_mode_args)] * len(tasks)
        return [[(f, r) for f in e[0] for r in e[1]] for e in repetitions_mode_args]

    if repetitions_mode == "individual":
        assert repetitions_mode_args is not None, (
            "If `repetitions_mode` is 'individual', `repetitions_mode_args` must be provided"
        )
        assert isinstance(repetitions_mode_args, list), (
            "If `repetitions_mode` is 'individual', `repetitions_mode_args` must be a list"
        )
        assert len(repetitions_mode_args) > 0, "`repetitions_mode_args` for 'individual' must not be empty"

        if isinstance(repetitions_mode_args[0], tuple):
            assert all(
                isinstance(rep, tuple) and (len(rep) == 2) and all(isinstance(i, int) for i in rep)
                for rep in repetitions_mode_args
            ), (
                "If `repetitions_mode_args` for 'individual' is a list of tuples, all elements must be tuples of integers of (fold_index, repeat_index) pairs"
            )
            repetitions_mode_args = [repetitions_mode_args] * len(tasks)

        # At this point, repetitions_mode_args must be list of lists
        assert len(repetitions_mode_args) == len(tasks), (
            "If `repetitions_mode_args` for 'individual' is a list, it must have the same length as `tasks`"
        )
        assert isinstance(repetitions_mode_args[0], list), (
            "Elements of `repetitions_mode_args` for 'individual' must be a list"
        )
        assert all(isinstance(rep, list) for rep in repetitions_mode_args), (
            "If `repetitions_mode_args` for 'individual' is a list, all elements must be lists"
        )
        for e in repetitions_mode_args:
            assert len(e) > 0, (
                "In `repetitions_mode_args`, each task's repetition list must contain at least one (fold, repeat) tuple."
            )
            for rep in e:
                assert isinstance(rep, tuple) and len(rep) == 2, (  # noqa: PT018
                    "In `repetitions_mode_args`, each repetition entry must be a tuple of (fold_index, repeat_index)."
                )
                assert all(isinstance(i, int) for i in rep), (
                    "In `repetitions_mode_args`, each element of a repetition tuple must be an integer."
                )

        return repetitions_mode_args

    raise ValueError(f"Unknown `repetitions_mode` str: {repetitions_mode}")


def _task_cache_key(task: int | UserTask) -> int | str:
    """Canonical, filesystem-safe identifier for a task, used to key its cache.

    Returns the OpenML task id for an integer task, or the ``UserTask.slug`` for a
    local task. This is the per-task component of the results cache path (see
    ``_build_cache_prefix``) and the same key under which the task's text-embedding
    cache is stored (mirrors
    ``tabarena.benchmark.preprocessing.text_cache.text_cache_key``), so a task's
    results and text caches stay consistently keyed off one identifier.
    """
    return task if isinstance(task, int) else task.slug


def _build_cache_prefix(
    *,
    method_name: str,
    cache_task_key: int | str,
    fold: int,
    repeat: int,
) -> str:
    """Build the cache directory prefix (relative to the base cache path).

    The subtask component is always ``{repeat}_{fold}``.
    """
    subtask_cache_name = ExperimentBatchRunner._subtask_name(fold=fold, repeat=repeat)
    return f"data/{method_name}/{cache_task_key}/{subtask_cache_name}"


def _build_results_cacher(
    *,
    cache_cls: type[AbstractCacheFunction],
    cache_cls_kwargs: dict | None,
    base_cache_path: str,
    method_name: str,
    cache_task_key: int | str,
    fold: int,
    repeat: int,
) -> AbstractCacheFunction:
    """Construct the cacher for a single (method, task, fold, repeat) ``results`` artifact.

    Centralizes the results cache layout so the run loop never assembles cache paths by
    hand: the artifact is named ``results`` and lives at
    ``{base_cache_path}/data/{method_name}/{cache_task_key}/{repeat}_{fold}``. Note the
    task identity enters only as the pre-derived ``cache_task_key`` (see
    ``_task_cache_key``); the cache class itself stays task-agnostic and reusable.

    ``include_self_in_call`` passes the cacher into the runner so a failed fit (in
    benchmark mode) can drop a ``model_failures`` artifact next to the results. It
    defaults to False (no such artifact) and does not affect the ``results`` cache
    file's path, format, or hit logic. An explicit
    ``cache_cls_kwargs["include_self_in_call"]`` takes precedence (e.g.
    ``ExperimentBatchRunner`` sets it True to keep the legacy artifact).
    """
    cache_prefix = _build_cache_prefix(
        method_name=method_name,
        cache_task_key=cache_task_key,
        fold=fold,
        repeat=repeat,
    )
    cache_path = f"{base_cache_path}/{cache_prefix}"
    return cache_cls(
        cache_name="results",
        cache_path=cache_path,
        **{"include_self_in_call": False, **(cache_cls_kwargs or {})},
    )


@dataclass
class _RunStats:
    """Running tallies for a `run_experiments_new` sweep, plus its progress line.

    Centralizes the counters that were previously a handful of loose ints so the run
    loop only does ``stats.<field> += 1`` and asks for a formatted progress line.
    """

    total: int
    started: int = 0
    success: int = 0
    fail: int = 0
    cache_exists: int = 0
    missing: int = 0

    def progress_line(self, *, cache_task_key: int | str, repeat: int, fold: int, method_name: str) -> str:
        return (
            f"\t{self.started}/{self.total} ran | "
            f"{self.success} success | "
            f"{self.fail} fail | "
            f"{self.cache_exists} cache_exists | "
            f"{self.missing} missing | "
            f"Fitting {cache_task_key} on repeat {repeat}, fold {fold} for method {method_name}"
        )


class _LazyTask:
    """Materialize a task (OpenML download / local load) at most once, on demand.

    `run_experiments_new` only needs the heavy `OpenMLTaskWrapper` when it actually
    fits a model; fully-cached (method, fold, repeat) jobs load straight from disk. This
    wrapper defers and memoizes that load, and exposes whatever has been loaded so far
    via `current` so a default-mode cache hit reuses a prior load without forcing one
    (mirroring the original loop's per-dataset `task is None` reuse).
    """

    def __init__(self, task_id_or_object: int | UserTask) -> None:
        self._spec = task_id_or_object
        self._loaded = False
        self._task: OpenMLTaskWrapper | None = None
        self._eval_metric_name: str | None = None
        self._task_name: str | None = None

    @property
    def current(self) -> tuple[OpenMLTaskWrapper | None, str | None, str | None]:
        """The `(task, eval_metric_name, task_name)` loaded so far (`None`s if never)."""
        return self._task, self._eval_metric_name, self._task_name

    def materialize(self) -> tuple[OpenMLTaskWrapper, str, str]:
        """Load the task on first call (memoized), returning `(task, eval_metric, name)`."""
        if not self._loaded:
            spec = self._spec
            if isinstance(spec, int):
                task = OpenMLTaskWrapper.from_task_id(task_id=spec)
                task_name = task.task.get_dataset().name
            else:
                task_name = spec.tabarena_task_name
                task = OpenMLTaskWrapper(
                    task=spec.load_local_openml_task(),
                    use_task_eval_metric=True,
                    lazy_load_data=True,
                )
            self._task = task
            self._eval_metric_name = task.eval_metric
            self._task_name = task_name
            self._loaded = True
            print(f"Using eval metric: {self._eval_metric_name}")
        return self._task, self._eval_metric_name, self._task_name


@dataclass
class _Job:
    """One (method, task, fold, repeat) unit of work, with its results cacher resolved.

    The shared scaffold (`_iter_jobs`) builds these; the load and run sweeps consume them.
    `cache_existed` is sampled once at build time so neither sweep re-stats the cache.
    """

    model_experiment: Experiment
    lazy_task: _LazyTask
    cache_task_key: int | str
    fold: int
    repeat: int
    cacher: AbstractCacheFunction
    cache_existed: bool


def _iter_jobs(
    *,
    tasks: list[int | UserTask],
    fold_repeat_pairs_per_task: list[list[tuple[int, int]]],
    model_experiments: list[Experiment],
    stats: _RunStats,
    base_cache_path: str,
    cache_cls: type[AbstractCacheFunction],
    cache_cls_kwargs: dict | None,
) -> Iterator[_Job]:
    """Yield every (task, fold, repeat, method) job — the scaffold shared by both sweeps.

    Owns everything common to loading and running: per-dataset task laziness, the nested
    task/split/method ordering, the progress prints + `stats.started` bump, and building
    each job's results cacher. The load-vs-run split happens in the consumer, not here.
    """
    for dataset_index, task_id_or_object in enumerate(tasks):
        lazy_task = _LazyTask(task_id_or_object)
        cache_task_key = _task_cache_key(task_id_or_object)
        print(f"Starting Dataset {dataset_index + 1}/{len(tasks)}...")

        fold_repeat_pairs = fold_repeat_pairs_per_task[dataset_index]
        for split_index, (fold, repeat) in enumerate(fold_repeat_pairs, start=1):
            print(f"Starting Split {split_index}/{len(fold_repeat_pairs)} (Fold {fold}, Repeat {repeat})...")

            for me_index, model_experiment in enumerate(model_experiments, start=1):
                stats.started += 1
                print(
                    f"Starting Model {me_index}/{len(model_experiments)}...\n"
                    + stats.progress_line(
                        cache_task_key=cache_task_key,
                        repeat=repeat,
                        fold=fold,
                        method_name=model_experiment.name,
                    ),
                )
                cacher = _build_results_cacher(
                    cache_cls=cache_cls,
                    cache_cls_kwargs=cache_cls_kwargs,
                    base_cache_path=base_cache_path,
                    method_name=model_experiment.name,
                    cache_task_key=cache_task_key,
                    fold=fold,
                    repeat=repeat,
                )
                yield _Job(
                    model_experiment=model_experiment,
                    lazy_task=lazy_task,
                    cache_task_key=cache_task_key,
                    fold=fold,
                    repeat=repeat,
                    cacher=cacher,
                    cache_existed=cacher.exists,
                )


def _load_sweep(
    jobs: Iterable[_Job],
    *,
    stats: _RunStats,
    strict: bool,
) -> list[dict]:
    """Load-only workflow: read each job's cached `results`, never fitting a model.

    Only the loading path lives here. A missing cache file is counted (and, when
    `strict`, collected and raised after the full sweep); a present one is loaded and
    counted as a success. The non-finite-metric guard is a fit-time concern (see
    `_run_sweep`) — cached results were already validated when written, so this path
    just reads them back.
    """
    results: list[dict] = []
    missing: list[tuple] = []
    for job in jobs:
        if not job.cache_existed:
            stats.missing += 1
            if strict:
                missing.append((job.model_experiment.name, job.cache_task_key, job.fold, job.repeat))
            continue

        stats.cache_exists += 1
        stats.success += 1
        results.append(job.cacher.load_cache())

    if strict and missing:
        missing_str = "\n\t".join(str(m) for m in missing)
        raise AssertionError(
            f"cache_mode='only_strict': missing cached results for "
            f"{len(missing)}/{stats.total} experiment(s).\n"
            f"Missing (method, task, fold, repeat):\n\t{missing_str}",
        )
    return results


def _run_sweep(
    jobs: Iterable[_Job],
    *,
    stats: _RunStats,
    ignore_cache: bool,
    debug_mode: bool,
    raise_on_failure: bool,
) -> list[dict]:
    """Run workflow: fit each job, reusing a cached result when one is already present.

    Only the running path lives here, and it is thin: each job's full fit lifecycle —
    task configuration, the failure guard, and the non-finite-metric guard — is owned by
    `Experiment.run`. This sweep just decides whether the (heavy) task needs materializing
    (only on a forced re-run via `ignore_cache` or a cache miss; on a default-mode hit
    `Experiment.run` short-circuits to the cached `results`, so an already-loaded task, if
    any, is reused), hands off to `run`, and tracks success/fail + (non-`ignore`) hits.
    """
    results: list[dict] = []
    for job in jobs:
        if ignore_cache or not job.cache_existed:
            task, eval_metric_name, task_name = job.lazy_task.materialize()
        else:
            task, eval_metric_name, task_name = job.lazy_task.current

        out = job.model_experiment.run(
            task=task,
            fold=job.fold,
            task_name=task_name,
            cache_task_key=job.cache_task_key,
            repeat=job.repeat,
            cacher=job.cacher,
            ignore_cache=ignore_cache,
            raise_on_failure=raise_on_failure,
            debug_mode=debug_mode,
            eval_metric_name=eval_metric_name,
        )

        if job.cache_existed and not ignore_cache:
            stats.cache_exists += 1
        if out is not None:
            stats.success += 1
            results.append(out)
        else:
            stats.fail += 1
    return results


def run_experiments_new(
    *,
    output_dir: str,
    model_experiments: list[Experiment],
    tasks: list[int | UserTask],
    repetitions_mode: Literal["TabArena-Lite", "TabArena", "matrix", "individual"],
    tasks_metadata: pd.DataFrame | None = None,
    repetitions_mode_args: tuple | list | None = None,
    cache_mode: Literal["default", "ignore", "only", "only_strict"] = "default",
    cache_cls: type[AbstractCacheFunction] = CacheFunctionPickle,
    cache_cls_kwargs: dict | None = None,
    raise_on_failure: bool = True,
    debug_mode: bool = False,
) -> list[dict]:
    """Run model experiments for a set of tasks.

    Parameters
    ----------
    output_dir: str
        Path to the local directory where experiment artifacts are cached.
    model_experiments: list[Experiment]
        List of model experiments to run. Each element must be an instance of the
        Experiment class. Each instance contains the configuration of the model
        and experiment to run.
    tasks: list[int | UserTask]
        The OpenML task IDs or UserTask instances to run the experiments on.
        See `tabarena.benchmark.task.user_task` for more details on how to define
        UserTask.
    repetitions_mode: Literal["TabArena-Lite", "TabArena", "matrix", "individual"]
        Determines how to run repeats of experiments:
            - "TabArena-Lite": Preset setting, run the first fold of the first repeat
                for all tasks.
            - "TabArena": Full TabArena benchmark setting. Recommended for the final
                evaluation. Requires `tasks_metadata` to be set.
            - "matrix": Allows you to specify a matrix of folds and repeats to run all
                combinations. See `repetitions_mode_args`.
            - "individual": Allows you to specific individual fold-repeats pairs to run.
                See `repetitions_mode_args`.
    tasks_metadata: pd.DataFrame | None, default None
        Metadata about each task in `tasks`. Required if `repetitions_mode="TabArena"`.

        If None, we assume that the `tasks` contain tasks from the official curated
        TabArena benchmark and load the metadata internally. If it contains tasks
        not in the official benchmark, an error will be raised.

        If pd.DataFrame, we assume the users passes custom metadata. This dataframe
        must contain the following columns:
            "task_id": str
                The task ID for the task as an int.
                If a local task, we assume this to be `UserTask.task_id_str`.
            "tabarena_num_repeats": int
                The number of repeats for the task based on the protocol from TabArena.
                See tabarena.nips2025_utils.fetch_metadata._get_n_repeats for details.
            "num_folds": int
                The number of folds for the task.
    repetitions_mode_args: list | tuple | None, default None
        Determine how many repetitions of the experiments to run per task, i.e., how
        many folds and repeats to run for each task. Note, all tasks come with
        pre-defined splits, so this parameter does not control how the data is split
        and will error if the numbers are not compatible with the pre-defined splits.
        This parameter's behavior depends on the `repetitions_mode`.

        If `repetitions_mode` is "TabArena-Lite", this parameter is ignored.

        If `repetitions_mode` is "matrix", this parameter defines a list of folds and
        a list of repeats for which we run all combinations. For example, if you pass
        `repetitions_mode_args = ([0, 1, 2], [0, 1])`, we will run the folds 0, 1,
        and 2 for repeats 0 and 1. The options to specify the folds and repeats are:
            - tuple[list[int], list[int]]: A tuple of two lists, where the first list
                contains the folds to run, and the second the repeats to run.
                For example, ([0, 2], [0, 3]) will run the first and third fold of the
                first and third repeats for each task. We start counting from 0.
                Set (list[int], [0]) to run folds for the first repeat.
            - tuple[int, int]: The first element is the number of folds to run, and
                the second the number of repeats to run. For example, (5, 3) will run
                the first 5 folds of the first 3 repeats for each task.
                We start counting from 0, so (2, 3) will run folds 0-1 and repeats 0-2.
                Set (X, 1) to run only the first X folds of the first repeat.
            - list[Any]: A list of tuples, where each of the elements follows one of
                the above formats, that specifies the repeat and fold pairs to run for
                each task. We assume the list is ordered the same as the tasks, so the
                first tuple corresponds to the first task, and so on.

        If `repetitions_mode` is "individual", this parameter defines a list of
        individual folds-repeat to run. For example, if you pass
        `repetitions_mode_args = [(0,0), (2,3)]`, we will run the first fold of
        the first repeat and the third fold of the fourth repeats for each task.
        The options to specify individual folds and repeats are:
            - list[tuple[int,int]]: A list of tuples, where the each tuple
                represents one fold-repeat pair to run for all tasks. Each
                tuple contains two integers, the first one is the fold index and
                second one is the repeat index.
            - list[Any]: A list of lists, where each of the elements follows the
                above format, that specifies the repeat and fold pairs to run for
                each task. We assume the list is ordered the same as the tasks, so the
                first tuple corresponds to the first task, and so on.
    cache_mode: Literal["default", "ignore", "only", "only_strict"], default "default"
        Determines how to handle the cache:
            - "default": Skip experiment if cache exists, otherwise run the experiment.
            - "ignore": Ignore the cache and always run the experiment. This will
                overwrite the cache file upon completion.
            - "only": Only load results from cache. This does not run the experiment
                if cache does not exist; missing experiments are silently skipped.
            - "only_strict": Like "only", but raise if any requested experiment is
                missing from the cache (after listing all of the missing experiments).
    cache_cls: type[AbstractCacheFunction], default CacheFunctionPickle
        The cache class used to read/write each experiment's `results`. Must accept
        `cache_name` and `cache_path` constructor arguments (e.g. `CacheFunctionPickle`
        or a drop-in replacement).
    cache_cls_kwargs: dict | None, default None
        Extra keyword arguments forwarded to `cache_cls(...)` (e.g. `compress`).
    raise_on_failure: bool, default True
        If True, will raise exceptions that occur during experiments, stopping all runs.
        If False, will ignore exceptions and continue fitting queued experiments.
        Experiments with exceptions will not be included in the output list.
    debug_mode: bool, default False
        Determine how to run the experiments:
            - If True, operates in a manner best suited for local model development.
                This mode is friendly to local debuggers and avoids subprocesses/threads
                and complex try/except logic.
            - If False, operates in a manner best suited for large-scale benchmarking.
                This mode tries to record information when method's fail and might not
                work well with local debuggers.

    Each experiment carries its own `dynamic_tabarena_validation_protocol` flag
    (see `Experiment`): when True, that experiment's validation split is
    dynamically configured based on the task type and dataset type at run time.

    Returns:
    -------
    result_lst: list[dict]
        Containing all metrics from fit() and predict() of all the given tasks
    """
    base_cache_path = output_dir

    assert all(isinstance(exp, Experiment) for exp in model_experiments), (
        "All `model_experiments` elements must be instances of Experiment class"
    )
    assert len({exp.name for exp in model_experiments}) == len(model_experiments), (
        "Duplicate experiment name found in `model_experiments`. All names must be unique."
    )
    assert all(isinstance(task, (int, UserTask)) for task in tasks), (
        f"Not all tasks are int or UserTask instances! Got: {tasks}"
    )

    fold_repeat_pairs_per_task = _parse_repetitions_mode_and_args(
        repetitions_mode=repetitions_mode,
        repetitions_mode_args=repetitions_mode_args,
        tasks=tasks,
        tasks_metadata=tasks_metadata,
    )
    n_splits = sum(len(pairs) for pairs in fold_repeat_pairs_per_task)

    print(
        f"Running Experiments, saving to: '{output_dir}'..."
        f"\n\tFitting {len(tasks)} tasks with a total of {n_splits} fold-repeat pairs"
        f"\n\tFitting {len(model_experiments)} methods with {n_splits} fold-repeat pairs for a total of {n_splits * len(model_experiments)} jobs..."
        f"\n\tTIDs    : {tasks}"
        f"\n\tRepeat-Fold-Pairs-Per-Task (first 20): {fold_repeat_pairs_per_task[:20]}"
        f"\n\tMethods : {[method.name for method in model_experiments]}",
    )

    stats = _RunStats(total=n_splits * len(model_experiments))
    jobs = _iter_jobs(
        tasks=tasks,
        fold_repeat_pairs_per_task=fold_repeat_pairs_per_task,
        model_experiments=model_experiments,
        stats=stats,
        base_cache_path=base_cache_path,
        cache_cls=cache_cls,
        cache_cls_kwargs=cache_cls_kwargs,
    )

    # Split point: a load-only sweep never fits a model and only reads the cache; every
    # other mode runs (with `Experiment.run` short-circuiting a default-mode cache hit).
    # Below here the loading and running paths — and their result tracking — are disjoint.
    if cache_mode in ("only", "only_strict"):
        return _load_sweep(
            jobs,
            stats=stats,
            strict=cache_mode == "only_strict",
        )
    return _run_sweep(
        jobs,
        stats=stats,
        ignore_cache=cache_mode == "ignore",
        debug_mode=debug_mode,
        raise_on_failure=raise_on_failure,
    )
