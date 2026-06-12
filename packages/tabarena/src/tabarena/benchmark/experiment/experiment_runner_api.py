from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from tabarena.benchmark.experiment import Experiment, ExperimentBatchRunner
from tabarena.benchmark.task.openml import OpenMLTaskWrapper
from tabarena.benchmark.task.user_task import UserTask
from tabarena.utils.cache import AbstractCacheFunction, CacheFunctionPickle

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from tabarena.benchmark.task import TaskWrapper


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


def task_cache_key_from_task_id_str(task_id_str: str) -> int | str:
    """The results-cache task key for a serialized task id (``task_id_str``).

    The string-side counterpart of :func:`_task_cache_key`: an OpenML id string maps to
    its int, a ``UserTask`` id string to the task's ``slug``. This is the *one*
    normalization shared by the cache writer (the run engine) and any cache-hit
    pre-check (e.g. the SLURM dispatch filter) — they must agree or cached jobs are
    needlessly re-run.
    """
    try:
        return int(task_id_str)
    except ValueError:
        return UserTask.from_task_id_str(task_id_str).slug


def job_cache_exists(
    *,
    output_dir: str,
    method_name: str,
    task_id_str: str,
    fold: int,
    repeat: int,
    cache_cls: type[AbstractCacheFunction] = CacheFunctionPickle,
    cache_cls_kwargs: dict | None = None,
) -> bool:
    """Whether the ``results`` cache for one (method, task, fold, repeat) unit exists.

    Built on the same `_build_results_cacher` the run engine writes through, so the
    check can never drift from the writer's cache layout. ``task_id_str`` is the
    serialized task id (see :func:`task_cache_key_from_task_id_str`).
    """
    cacher = _build_results_cacher(
        cache_cls=cache_cls,
        cache_cls_kwargs=cache_cls_kwargs,
        base_cache_path=output_dir,
        method_name=method_name,
        cache_task_key=task_cache_key_from_task_id_str(task_id_str),
        fold=fold,
        repeat=repeat,
    )
    return cacher.exists


def job_cache_exists_batch(
    *,
    items: list[tuple[str, str, int, int]],
    output_dir: str,
) -> list[bool]:
    """Batched :func:`job_cache_exists` over ``(method_name, task_id_str, fold, repeat)`` tuples.

    Module-level and tuple-based so dispatch filters can fan it out across workers
    (e.g. Ray in ``tabflow_slurm``) without pickling live experiments.
    """
    return [
        job_cache_exists(
            output_dir=output_dir,
            method_name=method_name,
            task_id_str=task_id_str,
            fold=fold,
            repeat=repeat,
        )
        for method_name, task_id_str, fold, repeat in items
    ]


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
    """Running tallies for a `_run_job_specs` sweep, plus its progress line.

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

    The run engine only needs the heavy `OpenMLTaskWrapper` when it actually
    fits a model; fully-cached (method, fold, repeat) jobs load straight from disk. This
    wrapper defers and memoizes that load, and exposes whatever has been loaded so far
    via `current` so a default-mode cache hit reuses a prior load without forcing one
    (mirroring the original loop's per-dataset `task is None` reuse).
    """

    def __init__(self, task_id_or_object: int | UserTask) -> None:
        self._spec = task_id_or_object
        self._loaded = False
        self._task: TaskWrapper | None = None
        self._eval_metric_name: str | None = None
        self._task_name: str | None = None

    @property
    def current(self) -> tuple[TaskWrapper | None, str | None, str | None]:
        """The `(task, eval_metric_name, task_name)` loaded so far (`None`s if never)."""
        return self._task, self._eval_metric_name, self._task_name

    def materialize(self) -> tuple[TaskWrapper, str, str]:
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
    consumes a flat list of these; `ExperimentBatchRunner`'s front doors (`_run_individual`
    for the name-keyed grid entry points, `run_jobs` for a sparse job list) build specs and
    hand them over.
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
    """Execute a flat list of `_JobSpec`s — the one engine behind every front door.

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
