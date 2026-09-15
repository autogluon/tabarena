"""Ray helpers: task mapping utilities and the untimed Ray warm-up used by the benchmark harness.

``ray`` is imported inside every function, never at module import time, so this module can be
imported by code that must stay light (``tabarena.models.warmup``, the model wrappers) and by
processes that never start Ray.

The warm-up helpers mirror what AutoGluon's parallel fold fitting does on its own inside the
timed fit: :func:`ensure_ray_initialized` starts the Ray runtime with the arguments
``ParallelLocalFoldFittingStrategy._get_ray_init_args`` would use, and
:func:`warmup_ray_workers` pre-starts worker interpreters that have already imported the model
library. Both are best effort and both honor the ``TABARENA_DISABLE_RAY_WARMUP`` kill switch.
TabArena never calls ``ray.shutdown()``: AutoGluon does not either, a SLURM job's Ray belongs to
the process, in-process sweeps reuse it, and Ray's own atexit hook stops it at interpreter exit.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import os
import sys
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

DEFAULT_REMOTE_KWARGS = {
    "max_calls": 1,
    "retry_exceptions": True,
    "max_retries": 0,
}

#: Kill switch: when truthy, no warm-up helper starts Ray or a worker pool (set by ``tests/conftest.py``).
DISABLE_RAY_WARMUP_ENV = "TABARENA_DISABLE_RAY_WARMUP"
#: Opt-in for the import-only worker pool; off until its effect is measured on the cluster.
RAY_WORKER_WARMUP_ENV = "TABARENA_RAY_WORKER_WARMUP"
#: Seconds a warm worker waits at the barrier before it gives up and returns.
RAY_WORKER_WARMUP_TIMEOUT_S: float = 120.0

_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUTHY


def ray_warmup_disabled() -> bool:
    """Whether ``TABARENA_DISABLE_RAY_WARMUP`` forbids the warm-up from starting Ray or a worker pool."""
    return _env_flag(DISABLE_RAY_WARMUP_ENV)


def ray_worker_warmup_enabled() -> bool:
    """Whether ``TABARENA_RAY_WORKER_WARMUP`` opts into the import-only worker pool (default off)."""
    return _env_flag(RAY_WORKER_WARMUP_ENV)


def try_import_ray():
    """AutoGluon's version-gated Ray import.

    Returns the ``ray`` module when it is installed inside the version window AutoGluon accepts and
    raises otherwise, exactly like ``BaggedEnsembleModel._get_default_fold_fitting_strategy`` decides
    between parallel and sequential fold fitting. Tests monkeypatch this function to inject a fake.
    """
    from autogluon.common.utils.try_import import try_import_ray as _try_import_ray

    return _try_import_ray()


def ensure_ray_initialized(*, num_cpus: int | None, num_gpus: float | None) -> dict[str, Any]:
    """Start Ray the way ``ParallelLocalFoldFittingStrategy`` would, unless it is already running.

    Mirrors ``fold_fitting_strategy.py`` ``_get_ray_init_args``: ``log_to_driver=False``,
    ``logging_level=logging.ERROR``, ``num_cpus`` always, ``num_gpus`` only when it is positive. The
    one deliberate addition is ``include_dashboard=False``; AutoGluon leaves the dashboard at Ray's
    default and SLURM jobs already disable it, so this only removes harness processes. Passing the
    same resource arguments matters: AutoGluon skips its own ``ray.init`` when Ray is up, so a
    pre-started runtime with different resources would change how the folds are scheduled. When
    ``num_cpus`` is unknown nothing is started, because AutoGluon must then own the resource
    detection. On SLURM with ``setup_slurm_job`` Ray is already initialized and this records that.

    Returns:
        A dict with ``already_imported``, ``already_initialized``, ``initialized_by_warmup``,
        ``init_args`` (the arguments passed to ``ray.init``, or ``None``) and, when nothing was
        done, ``skipped`` with the reason.
    """
    out: dict[str, Any] = {
        "already_imported": "ray" in sys.modules,
        "already_initialized": False,
        "initialized_by_warmup": False,
        "init_args": None,
    }
    if ray_warmup_disabled():
        out["skipped"] = DISABLE_RAY_WARMUP_ENV
        return out
    if num_cpus is None:
        out["skipped"] = "num_cpus unknown; AutoGluon owns the resource detection"
        return out
    ray = try_import_ray()
    if ray.is_initialized():
        out["already_initialized"] = True
        return out
    init_args: dict[str, Any] = {
        "log_to_driver": False,
        "logging_level": logging.ERROR,
        "include_dashboard": False,
        "num_cpus": int(num_cpus),
    }
    if num_gpus is not None and num_gpus > 0:
        init_args["num_gpus"] = num_gpus
    ray.init(**init_args)
    out["initialized_by_warmup"] = True
    out["init_args"] = init_args
    return out


def plan_ray_worker_pool(*, num_cpus: int, num_jobs: int) -> tuple[int, int]:
    """Size an import-only worker pool like AutoGluon sizes the fold tasks of a bag.

    Mirrors ``CpuResourceCalculator.get_resources_per_job`` for the default case (no user resources
    per job): ``cpus_per_worker = max(1, num_cpus // num_jobs)`` and as many workers as can hold a
    CPU lease at once, capped at ``num_jobs``. Warm tasks must request the same ``num_cpus`` as the
    fold tasks because Ray derives ``OMP_NUM_THREADS`` from the assigned CPUs and OpenMP/BLAS read it
    when the library loads. Never returns more workers than fit, which would deadlock the barrier.

    Returns:
        ``(num_workers, cpus_per_worker)``.
    """
    if num_cpus < 1 or num_jobs < 1:
        return 0, 1
    cpus_per_worker = max(1, num_cpus // num_jobs)
    num_workers = max(0, min(num_jobs, num_cpus // cpus_per_worker))
    return num_workers, cpus_per_worker


class _WarmPoolBarrier:
    """Async actor body: releases every warm task once ``n`` of them have arrived (or on timeout).

    Decorated with ``ray.remote(num_cpus=0)`` inside :func:`warmup_ray_workers`, because ``ray`` may
    be absent when this module is imported. ``num_cpus=0`` matters: with ``num_workers *
    cpus_per_worker == num_cpus`` a CPU-bearing actor would leave one warm task unschedulable and the
    barrier would only release on timeout.
    """

    def __init__(self, n: int) -> None:
        self._n = n
        self._arrived = 0
        self._event = asyncio.Event()

    async def arrive(self, timeout_s: float) -> int:
        self._arrived += 1
        if self._arrived >= self._n:
            self._event.set()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._event.wait(), timeout_s)
        return self._arrived


def _import_in_worker(module_names: tuple[str, ...], barrier: Any, timeout_s: float) -> dict[str, Any]:
    """Runs inside a Ray worker: import ``module_names``, wait at the barrier, report what happened.

    Import only: never a CUDA context, never a model, never data. ``OMP_NUM_THREADS`` is removed
    from the worker's environment afterwards so a fold task later scheduled on this interpreter does
    not inherit the thread count Ray set for the warm task.
    """
    failed: dict[str, str] = {}
    for name in module_names:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failed[name] = f"{type(exc).__name__}: {exc}"
    ray = try_import_ray()
    with contextlib.suppress(Exception):
        ray.get(barrier.arrive.remote(timeout_s))
    omp = os.environ.pop("OMP_NUM_THREADS", None)
    return {
        "pid": os.getpid(),
        "imported": [name for name in module_names if name in sys.modules],
        "failed": failed,
        "omp_num_threads": omp,
    }


def warmup_ray_workers(
    module_names: Sequence[str],
    num_workers: int,
    *,
    cpus_per_worker: int = 1,
    timeout_s: float = RAY_WORKER_WARMUP_TIMEOUT_S,
) -> dict[str, Any]:
    """Pre-start ``num_workers`` Ray worker interpreters that have imported ``module_names``.

    Best effort and opt-in (see :func:`ray_worker_warmup_enabled`). Every warm task imports the
    modules and then waits at a barrier actor until all tasks have arrived: without the barrier a
    fast task finishes and its worker is re-leased before the raylet starts the next one, so fewer
    distinct interpreters get warmed. When AutoGluon recycles a worker after one fold (GPU bags, and
    CPU bags on AutoGluon versions without worker reuse) the pool covers the first batch of folds only;
    where workers are reused a warm worker stays warm for every fold it serves. Ray must already be
    initialized (see :func:`ensure_ray_initialized`).

    Returns:
        ``{"requested", "skipped"}`` when nothing ran, otherwise ``{"requested", "cpus_per_worker",
        "modules", "pids", "distinct_pids", "failed_imports", "elapsed_s"}``.
    """
    out: dict[str, Any] = {"requested": int(num_workers)}
    if ray_warmup_disabled():
        out["skipped"] = DISABLE_RAY_WARMUP_ENV
        return out
    if num_workers < 1:
        out["skipped"] = "no workers requested"
        return out
    ray = try_import_ray()
    if not ray.is_initialized():
        out["skipped"] = "ray is not initialized"
        return out
    modules = tuple(dict.fromkeys(name for name in module_names if name != "__main__"))
    start = time.perf_counter()
    barrier = ray.remote(num_cpus=0)(_WarmPoolBarrier).remote(num_workers)
    task = ray.remote(num_cpus=cpus_per_worker, max_retries=0)(_import_in_worker)
    refs = [task.remote(modules, barrier, timeout_s) for _ in range(num_workers)]
    results: list[dict[str, Any]] = []
    try:
        results = list(ray.get(refs, timeout=timeout_s + 30))
    except Exception as exc:
        logger.warning("Ray worker warm-up did not finish cleanly (%r); collecting the finished tasks.", exc)
        with contextlib.suppress(Exception):
            finished, _unfinished = ray.wait(refs, num_returns=len(refs), timeout=0)
            results = [ray.get(ref) for ref in finished]
    finally:
        with contextlib.suppress(Exception):
            ray.kill(barrier)
    pids = sorted(int(result["pid"]) for result in results if "pid" in result)
    failed_imports: dict[str, str] = {}
    for result in results:
        failed_imports.update(result.get("failed", {}))
    out.update(
        {
            "cpus_per_worker": int(cpus_per_worker),
            "modules": list(modules),
            "pids": pids,
            "distinct_pids": len(set(pids)),
            "failed_imports": failed_imports,
            "elapsed_s": time.perf_counter() - start,
        }
    )
    return out


def run_function_as_ray_task(
    *,
    func: Callable,
    num_cpus: int,
    num_gpus: int,
    func_kwargs: dict,
):
    """Run a function as a ray task with the given number of cpus and gpus. Blocks until the function is done."""
    import ray

    remote_func = ray.remote(**DEFAULT_REMOTE_KWARGS)(func)

    return ray.get(
        remote_func.options(num_cpus=num_cpus, num_gpus=num_gpus).remote(**func_kwargs),
    )


def ray_map_list(
    list_to_map: list,
    *,
    func: Callable,
    func_element_key_string: str,
    num_workers: int,
    num_cpus_per_worker: int,
    num_gpus_per_worker: int = 0,
    func_kwargs: dict | None = None,
    func_put_kwargs: dict | None = None,
    put_list_elements: bool = False,
    output_handler: Callable | None = None,
    track_progress: bool = False,
    tqdm_kwargs: dict | None = None,
    ray_remote_kwargs: dict | None = None,
) -> list:
    """Map a function over a list using ray. Blocks until all functions are done.

    Arguments:
    ----------
    list_to_map: list
        The list to map the function over.
    func: callable
        The function to map over the list.
    func_element_key_string: str
        The key string to use to pass the element as a key-value pair to the function.
        That is, the function will for example use `{func_element_key_string: list_to_map[0]}` as kwargs for `func`.
    num_workers: int
        The number of workers to use.
    num_cpus_per_worker: int
        The number of cpus to use per worker.
    func_kwargs: dict, default=None
        Additional kwargs to pass to the function.
    func_put_kwargs: dict, default=None
        Additional kwargs to pass to the function, where the values are put into the object store.
    put_list_elements: bool, default=False
        If True, put the elements of `list_to_map` into the object store before passing them to the function.
    output_handler: callable, default=None
        If not None, this should be a function that takes the output of `func` as input and uses it.
        For example, this could log the output or save it to a file.
    track_progress: bool, default=False
        Track the progress of working on the list.
    tqdm_kwargs: dict | None, default=None
        Additional kwargs to pass to tqdm if `track_progress` is True.
        For example, the decription for the progress bar: {"desc": "Processing list"}.
    """
    import ray

    assert num_workers > 0, "Number of workers must be at least 1!"
    remote_kwargs = deepcopy(DEFAULT_REMOTE_KWARGS)
    if ray_remote_kwargs is not None:
        remote_kwargs.update(ray_remote_kwargs)
    if ray_remote_kwargs is None or "max_calls" not in ray_remote_kwargs:
        remote_kwargs["max_calls"] = max(len(list_to_map) // num_workers, 1)
    remote_p = ray.remote(**remote_kwargs)(func)
    remote_p_options = {
        "num_cpus": num_cpus_per_worker,
        "num_gpus": num_gpus_per_worker,
    }

    job_refs = []
    job_refs_map = {}
    job_index = 0

    return_results = []

    if track_progress:
        from tqdm import tqdm

        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        pbar = tqdm(total=len(list_to_map), **tqdm_kwargs)

    # Setup Kwargs
    job_kwargs = {}
    if func_kwargs is not None:
        for key, value in func_kwargs.items():
            job_kwargs[key] = value
    if func_put_kwargs is not None:
        for key, value in func_put_kwargs.items():
            job_kwargs[key] = ray.put(value)

    # Start initial jobs
    for list_element in list_to_map[:num_workers]:
        result_ref = remote_p.options(**remote_p_options).remote(
            **{
                func_element_key_string: ray.put(list_element) if put_list_elements else list_element,
            },
            **job_kwargs,
        )
        job_refs.append(result_ref)
        job_refs_map[result_ref] = job_index
        job_index += 1

    # Worker loop
    unfinished_list = list_to_map[num_workers:]
    unfinished = job_refs
    while unfinished:
        finished, unfinished = ray.wait(unfinished, num_returns=1)
        job_i = job_refs_map[finished[0]]
        job_res = ray.get(finished[0])

        # Handle output (e.g. logging)
        if output_handler is not None:
            output_handler(job_res)
        if track_progress:
            pbar.update(n=1)

        return_results.append((job_i, job_res))

        # Re-schedule workers
        while unfinished_list and (len(unfinished) < num_workers):
            list_element = unfinished_list[0]
            unfinished_list = unfinished_list[1:]
            result_ref = remote_p.options(**remote_p_options).remote(
                **{
                    func_element_key_string: ray.put(list_element) if put_list_elements else list_element,
                },
                **job_kwargs,
            )
            unfinished.append(result_ref)
            job_refs_map[result_ref] = job_index
            job_index += 1

    if func_put_kwargs is not None:
        ray.internal.free(object_refs=[job_kwargs[key] for key in func_put_kwargs])
        for key in func_put_kwargs:
            del job_kwargs[key]

    if track_progress:
        pbar.close()

    return [r for _, r in sorted(return_results, key=lambda x: x[0])]


def to_batch_list(lst: list, batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]
