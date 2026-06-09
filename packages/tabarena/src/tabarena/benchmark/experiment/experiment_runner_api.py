from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from tabarena.benchmark.experiment import Experiment, ExperimentBatchRunner
from tabarena.benchmark.task.openml import OpenMLTaskWrapper
from tabarena.benchmark.task.user_task import UserTask
from tabarena.utils.cache import AbstractCacheFunction, CacheFunctionPickle

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection


def _as_int_list(value: object) -> list[int]:
    """Validate one matrix axis is a non-empty list of ints and return it unchanged."""
    assert isinstance(value, list) and len(value) > 0 and all(isinstance(x, int) for x in value), (  # noqa: PT018
        f"`repetitions_mode_args` for 'matrix' must be two ints or two non-empty int lists; got {value!r}."
    )
    return value


def _clean_repetitions_mode_args_for_matrix(
    repetitions_mode_args: tuple,
) -> tuple[list[int], list[int]]:
    """Normalize a matrix ``(folds, repeats)`` spec into ``(list[int], list[int])``.

    Both elements must be the same kind: either ints ``n`` (expanded to ``range(n)``) or
    non-empty lists of ints.
    """
    assert isinstance(repetitions_mode_args, tuple) and len(repetitions_mode_args) == 2, (  # noqa: PT018
        "`repetitions_mode_args` for 'matrix' must be a tuple of two elements: (folds, repeats)."
    )
    folds, repeats = repetitions_mode_args
    if isinstance(folds, int) and isinstance(repeats, int):
        return list(range(folds)), list(range(repeats))
    # Mixed int/list is rejected here too: `_as_int_list` fails on the int element.
    return _as_int_list(folds), _as_int_list(repeats)


def _assert_fold_repeat_pairs(pairs: object) -> None:
    """Assert ``pairs`` is a non-empty list of ``(fold_index, repeat_index)`` int tuples."""
    assert isinstance(pairs, list) and len(pairs) > 0, (  # noqa: PT018
        "Each 'individual' repetition list must be a non-empty list of (fold, repeat) tuples."
    )
    for rep in pairs:
        assert isinstance(rep, tuple) and len(rep) == 2 and all(isinstance(i, int) for i in rep), (  # noqa: PT018
            "Each 'individual' repetition must be a (fold_index, repeat_index) tuple of two ints."
        )


def _parse_tabarena_mode(
    *,
    tasks: list[int | UserTask],
    tasks_metadata: TaskMetadataCollection | None,
) -> list[list[tuple[int, int]]]:
    """Each task's ``(fold, repeat)`` pairs taken from its splits in `tasks_metadata`.

    Reads each task's *actual* splits from the collection (not a ``n_folds`` x ``n_repeats``
    product), so a task with a sparse / non-rectangular set of splits is respected. Tasks are
    matched to the collection by ``task_id_str`` (the ``UserTask`` id for a local task, the
    stringified OpenML tid otherwise).
    """
    if tasks_metadata is None:
        from tabarena.benchmark.task.metadata import default_task_metadata_collection

        tasks_metadata = default_task_metadata_collection()

    # Aggregate splits per task id, so both the "unrolled" (one entry per split) and the
    # multi-split forms of the collection behave the same.
    splits_by_id: dict[str, list[tuple[int, int]]] = {}
    for t in tasks_metadata:
        splits_by_id.setdefault(t.task_id_str, []).extend(
            (split.fold, split.repeat) for split in t.splits_metadata.values()
        )

    fold_repeat_pairs_per_task = []
    for task in tasks:
        t_id = task.task_id_str if isinstance(task, UserTask) else str(task)
        pairs = splits_by_id.get(t_id)
        assert pairs, f"Task ID '{t_id}' from `tasks` not found in `tasks_metadata`."
        fold_repeat_pairs_per_task.append(pairs)
    return fold_repeat_pairs_per_task


def _parse_individual_mode(
    *,
    repetitions_mode_args: tuple | list | None,
    n_tasks: int,
) -> list[list[tuple[int, int]]]:
    """Parse 'individual' args into a per-task list of (fold, repeat) pairs."""
    assert isinstance(repetitions_mode_args, list) and len(repetitions_mode_args) > 0, (  # noqa: PT018
        "`repetitions_mode_args` for 'individual' must be a non-empty list."
    )

    # A flat list of (fold, repeat) pairs is broadcast to every task.
    if isinstance(repetitions_mode_args[0], tuple):
        _assert_fold_repeat_pairs(repetitions_mode_args)
        return [repetitions_mode_args] * n_tasks

    # Otherwise: one list of (fold, repeat) pairs per task, aligned with `tasks`.
    assert len(repetitions_mode_args) == n_tasks, (
        "`repetitions_mode_args` for 'individual' (a list of per-task lists) must match the number of tasks."
    )
    for pairs in repetitions_mode_args:
        _assert_fold_repeat_pairs(pairs)
    return repetitions_mode_args


def _parse_repetitions_mode_and_args(
    *,
    repetitions_mode: Literal["TabArena-Lite", "TabArena", "matrix", "individual"],
    repetitions_mode_args: tuple | list | None,
    tasks: list[int | UserTask],
    tasks_metadata: TaskMetadataCollection | None = None,
) -> list[list[tuple[int, int]]]:
    """Resolve `repetitions_mode`/`repetitions_mode_args` into the folds/repeats per task.

    Returns one element per task, each a list of ``(fold_index, repeat_index)`` tuples.
    """
    n_tasks = len(tasks)

    if repetitions_mode == "TabArena-Lite":
        # First fold of the first repeat for each task.
        return [[(0, 0)] for _ in range(n_tasks)]

    if repetitions_mode == "TabArena":
        return _parse_tabarena_mode(tasks=tasks, tasks_metadata=tasks_metadata)

    if repetitions_mode == "matrix":
        assert repetitions_mode_args is not None, "`repetitions_mode_args` is required for 'matrix'."
        if isinstance(repetitions_mode_args, tuple):
            specs = [repetitions_mode_args] * n_tasks
        else:
            assert isinstance(repetitions_mode_args, list), (
                "`repetitions_mode_args` for 'matrix' must be a tuple or a list of per-task tuples."
            )
            assert len(repetitions_mode_args) == n_tasks, (
                "`repetitions_mode_args` list for 'matrix' must match the number of tasks."
            )
            specs = repetitions_mode_args
        cleaned = [_clean_repetitions_mode_args_for_matrix(spec) for spec in specs]
        return [[(f, r) for f in folds for r in repeats] for folds, repeats in cleaned]

    if repetitions_mode == "individual":
        return _parse_individual_mode(repetitions_mode_args=repetitions_mode_args, n_tasks=n_tasks)

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


@dataclass(frozen=True)
class _JobSpec:
    """A single (method, task, fold, repeat) unit, task identified by tid / UserTask.

    The tid-keyed counterpart of the public `Job` (which is dataset-*name* keyed): the
    name->tid resolution has already happened upstream. `_run_job_specs` is the engine that
    consumes a flat list of these; both front doors (`run_experiments_new` for a rectangular
    grid, `ExperimentBatchRunner.run_jobs` for a sparse job list) build specs and hand them
    over.
    """

    model_experiment: Experiment
    task: int | UserTask
    fold: int
    repeat: int


@dataclass
class _Job:
    """One (method, task, fold, repeat) unit of work, with its results cacher resolved.

    The shared scaffold (`_iter_jobs`) builds these; the load and run sweeps consume them.
    `cache_existed` is sampled once at build time so neither sweep re-stats the cache.
    `input_index` is the position of the originating spec in the engine's input list, so the
    run-order result stream can be reordered back to input order.
    """

    model_experiment: Experiment
    lazy_task: _LazyTask
    cache_task_key: int | str
    fold: int
    repeat: int
    cacher: AbstractCacheFunction
    cache_existed: bool
    input_index: int


def _iter_jobs(
    job_specs: list[_JobSpec],
    *,
    stats: _RunStats,
    base_cache_path: str,
    cache_cls: type[AbstractCacheFunction],
    cache_cls_kwargs: dict | None,
) -> Iterator[_Job]:
    """Yield a `_Job` per spec, grouped by task — the scaffold shared by both sweeps.

    Specs are bucketed by their cache key (`_task_cache_key`), preserving first-seen task
    order; one `_LazyTask` is built per bucket and shared across all of that task's
    method/fold/repeat jobs. This per-dataset grouping is what `_run_sweep`'s `.current`
    reuse and the one-dataset memory footprint rely on: once a task's contiguous block is
    consumed, its `_LazyTask` (and the loaded `OpenMLTaskWrapper`) is no longer referenced
    and can be collected before the next task loads. Each job carries its spec's
    `input_index` so the consumer can restore input order; a non-rectangular / interleaved
    spec list is fine. The load-vs-run split happens in the consumer, not here.
    """
    # Group by task, keeping each task's specs in input order and remembering one task
    # object per bucket to load lazily.
    indexed_specs_by_task: dict[int | str, list[tuple[int, _JobSpec]]] = {}
    task_by_key: dict[int | str, int | UserTask] = {}
    for input_index, spec in enumerate(job_specs):
        cache_task_key = _task_cache_key(spec.task)
        indexed_specs_by_task.setdefault(cache_task_key, []).append((input_index, spec))
        task_by_key.setdefault(cache_task_key, spec.task)

    n_tasks = len(indexed_specs_by_task)
    for dataset_index, (cache_task_key, indexed_specs) in enumerate(indexed_specs_by_task.items()):
        lazy_task = _LazyTask(task_by_key[cache_task_key])
        print(f"Starting Dataset {dataset_index + 1}/{n_tasks}...")

        for input_index, spec in indexed_specs:
            stats.started += 1
            print(
                stats.progress_line(
                    cache_task_key=cache_task_key,
                    repeat=spec.repeat,
                    fold=spec.fold,
                    method_name=spec.model_experiment.name,
                ),
            )
            cacher = _build_results_cacher(
                cache_cls=cache_cls,
                cache_cls_kwargs=cache_cls_kwargs,
                base_cache_path=base_cache_path,
                method_name=spec.model_experiment.name,
                cache_task_key=cache_task_key,
                fold=spec.fold,
                repeat=spec.repeat,
            )
            yield _Job(
                model_experiment=spec.model_experiment,
                lazy_task=lazy_task,
                cache_task_key=cache_task_key,
                fold=spec.fold,
                repeat=spec.repeat,
                cacher=cacher,
                cache_existed=cacher.exists,
                input_index=input_index,
            )


def _load_sweep(
    jobs: Iterable[_Job],
    *,
    stats: _RunStats,
    strict: bool,
) -> list[tuple[int, dict]]:
    """Load-only workflow: read each job's cached `results`, never fitting a model.

    Only the loading path lives here. A missing cache file is counted (and, when
    `strict`, collected and raised after the full sweep); a present one is loaded and
    counted as a success. The non-finite-metric guard is a fit-time concern (see
    `_run_sweep`) — cached results were already validated when written, so this path
    just reads them back. Each loaded result is paired with its job's `input_index` so the
    caller can restore input order.
    """
    results: list[tuple[int, dict]] = []
    missing: list[tuple] = []
    for job in jobs:
        if not job.cache_existed:
            stats.missing += 1
            if strict:
                missing.append((job.model_experiment.name, job.cache_task_key, job.fold, job.repeat))
            continue

        stats.cache_exists += 1
        stats.success += 1
        results.append((job.input_index, job.cacher.load_cache()))

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
) -> list[tuple[int, dict]]:
    """Run workflow: fit each job, reusing a cached result when one is already present.

    Only the running path lives here, and it is thin: each job's full fit lifecycle —
    task configuration, the failure guard, and the non-finite-metric guard — is owned by
    `Experiment.run`. This sweep just decides whether the (heavy) task needs materializing
    (only on a forced re-run via `ignore_cache` or a cache miss; on a default-mode hit
    `Experiment.run` short-circuits to the cached `results`, so an already-loaded task, if
    any, is reused), hands off to `run`, and tracks success/fail + (non-`ignore`) hits. Each
    successful result is paired with its job's `input_index` so the caller can restore input
    order.
    """
    results: list[tuple[int, dict]] = []
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
            results.append((job.input_index, out))
        else:
            stats.fail += 1
    return results


def _run_job_specs(
    job_specs: list[_JobSpec],
    *,
    base_cache_path: str,
    cache_mode: Literal["default", "ignore", "only", "only_strict"],
    cache_cls: type[AbstractCacheFunction],
    cache_cls_kwargs: dict | None,
    raise_on_failure: bool,
    debug_mode: bool,
) -> list[dict]:
    """Execute a flat list of `_JobSpec`s — the engine shared by both front doors.

    Run order is grouped by task (see `_iter_jobs`) so a task shared by several specs is
    loaded once and only one dataset is resident at a time. The returned results are then
    reordered to match the order of `job_specs`, so a caller's `results[i]` lines up with
    their input; failed or skipped jobs simply drop out, leaving the survivors in input
    order.

    Each (method.name, task, fold, repeat) must be unique: it is the results-cache key, so a
    duplicate would mean two jobs racing the same cache file.
    """
    cache_keys = [(s.model_experiment.name, _task_cache_key(s.task), s.fold, s.repeat) for s in job_specs]
    assert len(cache_keys) == len(set(cache_keys)), (
        "Duplicate (method, task, fold, repeat) job spec; this is the results-cache key and must be unique."
    )

    stats = _RunStats(total=len(job_specs))
    jobs = _iter_jobs(
        job_specs,
        stats=stats,
        base_cache_path=base_cache_path,
        cache_cls=cache_cls,
        cache_cls_kwargs=cache_cls_kwargs,
    )

    # Split point: a load-only sweep never fits a model and only reads the cache; every
    # other mode runs (with `Experiment.run` short-circuiting a default-mode cache hit).
    # Both return (input_index, result) pairs; sorting by index restores input order.
    if cache_mode in ("only", "only_strict"):
        indexed_results = _load_sweep(
            jobs,
            stats=stats,
            strict=cache_mode == "only_strict",
        )
    else:
        indexed_results = _run_sweep(
            jobs,
            stats=stats,
            ignore_cache=cache_mode == "ignore",
            debug_mode=debug_mode,
            raise_on_failure=raise_on_failure,
        )
    return [result for _, result in sorted(indexed_results, key=lambda pair: pair[0])]


def run_experiments_new(
    *,
    output_dir: str,
    model_experiments: list[Experiment],
    tasks: list[int | UserTask],
    repetitions_mode: Literal["TabArena-Lite", "TabArena", "matrix", "individual"],
    tasks_metadata: TaskMetadataCollection | None = None,
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
    tasks_metadata: TaskMetadataCollection | None, default None
        Native task metadata, used only when `repetitions_mode="TabArena"` to look up each
        task's splits. Tasks in `tasks` are matched to the collection by ``task_id_str``
        (the ``UserTask`` id for a local task, the stringified OpenML tid otherwise), and
        each task runs exactly that collection task's ``(fold, repeat)`` splits — so a
        sparse / non-rectangular set of splits is respected.

        If None, the official curated TabArena collection is loaded internally; passing a
        task not present in it raises. Wrap a legacy DataFrame with
        ``TaskMetadataCollection.from_legacy_df(df)`` to pass custom metadata.
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

    # Flatten the rectangular grid into per-(method, task, fold, repeat) specs in
    # task -> split -> method order. `_run_job_specs` regroups them by task for execution
    # and returns results in this same spec order, so the result ordering is identical to
    # the original nested loop.
    job_specs = [
        _JobSpec(model_experiment=model_experiment, task=task, fold=fold, repeat=repeat)
        for task, fold_repeat_pairs in zip(tasks, fold_repeat_pairs_per_task, strict=True)
        for (fold, repeat) in fold_repeat_pairs
        for model_experiment in model_experiments
    ]
    return _run_job_specs(
        job_specs,
        base_cache_path=base_cache_path,
        cache_mode=cache_mode,
        cache_cls=cache_cls,
        cache_cls_kwargs=cache_cls_kwargs,
        raise_on_failure=raise_on_failure,
        debug_mode=debug_mode,
    )
