"""Run one benchmark work unit on a (SLURM) compute node.

The unit is a core ``Job``: an experiment (referenced *by name*) on one
``(dataset, fold, repeat)`` split. Both sides of the reference live in the shipped
:class:`~tabarena.benchmark.experiment.job_batch.JobBatch` artifact
(``--job_batch_dir``): the experiment is loaded from its ``experiments.yaml`` and the
dataset resolves against its ``task_metadata.csv``, so this runner executes through
exactly the same :meth:`ExperimentBatchRunner.run_jobs` path (same task resolution,
results naming, and cache layout) as a local benchmark run.

Process order in ``__main__``: line-buffered stdout, thread variables aligned to the CPU
affinity (before any heavy import and before Ray starts), the offline-weights environment
(before the first ``huggingface_hub`` import), the job (which applies the batch's cache
config), then Ray only when the experiment's fit can reach it, then the fit. The submit
template runs this process with ``TMPDIR`` inside a per-job node-local scratch directory
that the shell removes afterwards, also when the process is killed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from tabflow_slurm.slurm_utils import apply_offline_weights_env, setup_slurm_job

if TYPE_CHECKING:
    from tabarena.benchmark.experiment import Job, JobBatch

#: Set to keep the Ray temp dir of an item for inspection instead of removing it.
KEEP_RAY_TEMP_DIR_ENV = "TABARENA_KEEP_RAY_TEMP_DIR"
#: Directory the Ray worker logs of a failed item are copied into (exported by the submit template).
RAY_LOG_DIR_ENV = "TABARENA_RAY_LOG_DIR"


def load_job(
    *,
    job_batch_dir: str,
    experiment_name: str,
    dataset: str,
    fold: int,
    repeat: int,
) -> tuple[JobBatch, Job]:
    """Load the batch, apply its cache config and return the job named by the coordinates.

    The coordinates are a lookup key into the batch's serialized job list: the job that runs is
    the one loaded from disk, and coordinates that do not name a job of the batch (e.g. a stale
    job JSON against a regenerated batch) fail loudly here.
    """
    from tabarena.benchmark.experiment import JobBatch

    batch = JobBatch.load(job_batch_dir)
    # Point this worker at the caches the run was configured with (OpenML / HuggingFace /
    # TabArena / TabPFN) before the task is loaded and before Ray starts, so fold workers inherit
    # them too. The config was embedded in the batch at setup time; absent means library defaults.
    if batch.cache_config is not None:
        batch.cache_config.apply()
    wanted = (experiment_name, dataset, fold, repeat)
    job = next(
        (j for j in batch.jobs if (j.experiment.name, *j.task.as_triple()) == wanted),
        None,
    )
    if job is None:
        experiment_names = sorted({j.experiment.name for j in batch.jobs})
        if experiment_name not in experiment_names:
            raise ValueError(
                f"Experiment {experiment_name!r} is not in the job batch at {job_batch_dir!r} "
                f"(has: {experiment_names}).",
            )
        raise ValueError(
            f"(experiment, dataset, fold, repeat) = {wanted} is not a job of the batch at "
            f"{job_batch_dir!r}. The job JSON may be stale relative to a regenerated batch.",
        )
    return batch, job


def job_problem_type(batch: JobBatch, job: Job) -> str | None:
    """The problem type of the job's dataset from the batch's task metadata; ``None`` when unknown."""
    try:
        return batch.task_metadata.task_metadata_by_dataset()[job.task.dataset].problem_type
    except Exception:
        return None


def resolve_setup_ray(requested: bool, batch: JobBatch, job: Job) -> bool:
    """Whether to start Ray before the fit: only when requested and the experiment can reach Ray.

    ``Experiment.uses_ray`` answers False for fits that never touch Ray (a foundation model with
    sequential fold fitting, a direct model fit), which saves the Ray startup wall-clock and keeps
    the Ray daemons out of the fit's memory numbers. Any error answers True, the safe default.
    """
    if not requested:
        return False
    try:
        return bool(job.experiment.uses_ray(problem_type=job_problem_type(batch, job)))
    except Exception:
        return True


def run_job(
    batch: JobBatch,
    job: Job,
    *,
    output_dir: str,
    ignore_cache: bool,
    cleanup_on_failure: bool = True,
    cpu_budget_check: Literal["error", "warn", "off"] = "warn",
    require_warmup: bool = True,
) -> list[dict]:
    """Run one loaded job through ``ExperimentBatchRunner.run_jobs`` and return its results.

    ``cleanup_on_failure``, ``cpu_budget_check`` and ``require_warmup`` are the benchmark-worker settings of
    ``ExperimentRunner``; they are placed on the loaded experiment's ``experiment_kwargs`` (only
    where the experiment does not set them itself), which ``Experiment.run`` forwards to the runner.
    The loaded experiment lives only in this process, so the shipped batch is untouched.
    """
    from tabarena.benchmark.experiment import ExperimentBatchRunner

    experiment_kwargs = getattr(job.experiment, "experiment_kwargs", None)
    if isinstance(experiment_kwargs, dict):
        experiment_kwargs.setdefault("cleanup_on_failure", cleanup_on_failure)
        experiment_kwargs.setdefault("cpu_budget_check", cpu_budget_check)
        experiment_kwargs.setdefault("require_warmup", require_warmup)

    runner = ExperimentBatchRunner(
        expname=output_dir,
        task_metadata=batch.task_metadata,
        cache_mode="ignore" if ignore_cache else "default",
        # Benchmark mode: record model failures instead of debugger-friendly behavior. A failure
        # still raises (raise_on_failure defaults to True), so the process exits non-zero and the
        # submit template counts the item as failed.
        debug_mode=False,
    )
    results_lst = runner.run_jobs([job])
    for results in results_lst:
        print("Metric error:", results.get("metric_error"))
    return results_lst


def run_experiment(
    *,
    job_batch_dir: str,
    experiment_name: str,
    dataset: str,
    fold: int,
    repeat: int,
    output_dir: str,
    ignore_cache: bool,
    cleanup_on_failure: bool = True,
    cpu_budget_check: Literal["error", "warn", "off"] = "warn",
    require_warmup: bool = True,
) -> list[dict]:
    """Run a single ``(experiment, dataset, fold, repeat)`` work unit from a job batch.

    :func:`load_job` followed by :func:`run_job`; the in-process mode of ``run_local`` calls this.
    Compute resources, fold-fitting strategy, preprocessing, and the dynamic validation protocol
    are baked into each serialized experiment at build time, so they are not passed here.

    Parameters
    ----------
    job_batch_dir : str
        Directory of the ``JobBatch`` artifact written at setup time (experiments +
        task metadata + job coordinates).
    experiment_name : str
        The experiment to run, by its unique name in the batch's ``experiments.yaml``.
    dataset : str
        The dataset to run on (the collection's ``tabarena_task_name``, also the results
        ``dataset`` key). A local ``UserTask`` (e.g. a materialized Data Foundry task) is
        auto-resolved from the collection's ``task_id_str``.
    fold : int
        The fold to run.
    repeat : int
        The repeat to run. Here, repeat 0 means the first set of folds without any repeats.
    output_dir : str
        The path to the output directory where the results will be saved (and cached).
    ignore_cache : bool
        Whether to ignore the cache or not. If True, the cache will be ignored and the
        experiment will be run from scratch and potentially overwrite existing results.
    cleanup_on_failure, cpu_budget_check, require_warmup :
        See :func:`run_job`.
    """
    batch, job = load_job(
        job_batch_dir=job_batch_dir,
        experiment_name=experiment_name,
        dataset=dataset,
        fold=fold,
        repeat=repeat,
    )
    return run_job(
        batch,
        job,
        output_dir=output_dir,
        ignore_cache=ignore_cache,
        cleanup_on_failure=cleanup_on_failure,
        cpu_budget_check=cpu_budget_check,
        require_warmup=require_warmup,
    )


def copy_ray_worker_logs(ray_temp_dir: str | None, dest_dir: str | None, job: Job | None = None) -> Path | None:
    """Copy Ray's ``session_latest/logs`` of a failed item into ``dest_dir``; best effort.

    The copy lands in ``<dest_dir>/<experiment>/<dataset>/<repeat>_<fold>`` when ``job`` is
    given. Returns the destination, or ``None`` when nothing was copied.
    """
    if not ray_temp_dir or not dest_dir:
        return None
    logs = Path(ray_temp_dir) / "session_latest" / "logs"
    if not logs.is_dir():
        return None
    dest = Path(dest_dir)
    if job is not None:
        dest = dest / job.experiment.name / job.task.dataset / f"{job.task.repeat}_{job.task.fold}"
    try:
        shutil.copytree(logs, dest, dirs_exist_ok=True, symlinks=True)
    except OSError as exc:
        print(f"WARNING: could not copy Ray worker logs to {dest}: {exc}")
        return None
    print(f"Ray worker logs copied to {dest}")
    return dest


def remove_ray_temp_dir(ray_temp_dir: str) -> None:
    """Remove an item's Ray temp dir, retrying once after Ray's asynchronous log flush."""
    # Ray's shutdown flushes worker logs asynchronously, so an immediate delete can race it
    # (OSError: Directory not empty). Retry once after the flush settles, then best effort.
    try:
        shutil.rmtree(ray_temp_dir)
    except OSError:
        time.sleep(10)
        shutil.rmtree(ray_temp_dir, ignore_errors=True)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _parse_int_or_none(s):
    if (s is None) or (s.lower() == "none") or (s.lower() == "null"):
        return None
    return int(s)


def _strip_quotes(value: str) -> str:
    """Strip matching surrounding quotes that can survive when the command is built as a
    string and passed through an extra (e.g. SLURM submission) layer without unquoting.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # Required work-unit coordinates
    parser.add_argument(
        "--job_batch_dir",
        type=str,
        required=True,
        help="Directory of the JobBatch artifact (experiments + task metadata + jobs).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Name of the experiment (in the batch's experiments.yaml) to run.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (the collection's tabarena_task_name) to run on.",
    )
    parser.add_argument("--fold", type=int, required=True, help="Fold of CV to run.")
    parser.add_argument(
        "--repeat",
        type=int,
        required=True,
        help="Repeat of CV to run. Here, repeat 0 means the first set of folds without any repeats.",
    )
    parser.add_argument(
        "--ignore_cache",
        type=_str2bool,
        default=False,
        help="Whether to ignore the cache or not. If True, the cache will be ignored and "
        "the experiment will be run from scratch and potentially overwrite existing results.",
    )
    # Experiment environment settings
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory where the results will be saved.",
        default=str(Path(__file__).parent / "run_tabarena_experiment_output"),
    )
    # Hardware settings
    parser.add_argument(
        "--num_cpus",
        type=_parse_int_or_none,
        help="Number of CPUs to use for the experiment. "
        "If None, Ray will automatically detect the number of CPUs and use that.",
        default=1,
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        help="Number of GPUs to use for the experiment (SLURM node allocation and Ray).",
        default=0,
    )
    parser.add_argument(
        "--memory_limit",
        type=_parse_int_or_none,
        help="Memory limit to use for the experiment. Given in GB. If None, no memory limit will be set.",
        default=10,
    )
    parser.add_argument(
        "--setup_ray_for_slurm_shared_resources_environment",
        type=_str2bool,
        help="If True, setup Ray to work well in a shared resources environment with SLURM "
        "(started only when the experiment's fit can reach Ray).",
        default=False,
    )
    parser.add_argument(
        "--ray_temp_root",
        type=str,
        default=None,
        help="Directory to create the Ray temp dir under (the per-job node-local scratch). Default: TMPDIR.",
    )
    parser.add_argument(
        "--cpu_budget_check",
        choices=["error", "warn", "off"],
        default="warn",
        help="What to do when the fit's num_cpus differs from the CPUs this process may run on: "
        "'warn' (default) prints the mismatch and records it in the experiment metadata, 'error' aborts the item so "
        "the protocol's budget must match the node, 'off' skips the check.",
    )
    parser.add_argument(
        "--require_warmup",
        type=_str2bool,
        default=True,
        help="Abort the item before the timed fit when the method's warm-up raised or left steps failed (default), "
        "so a cold timed fit is never recorded silently. Pass false to record the warm-up report and fit anyway.",
    )
    parser.add_argument(
        "--offline_weights",
        type=_str2bool,
        default=False,
        help="Export HF_HUB_OFFLINE=1, HF_HUB_DISABLE_PROGRESS_BARS=1 and AG_FETCH_PRETRAINED_WEIGHTS=false before "
        "any model library import. Set by the setup when every selected model had its weights prefetched.",
    )
    return parser


if __name__ == "__main__":
    # Every print reaches the SLURM log as soon as it is written, also from Ray-less processes.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    # Seed the OMP-style thread variables from the affinity mask before any library creates its
    # pools and before Ray starts, so daemons and fold workers inherit them.
    from tabarena.utils.thread_utils import align_thread_env_to_affinity

    align_thread_env_to_affinity()

    args = _build_parser().parse_args()
    # Before the first huggingface_hub import (see apply_offline_weights_env).
    apply_offline_weights_env(args.offline_weights)

    num_cpus = args.num_cpus
    if num_cpus is None:
        from tabarena.utils.resources import detect_num_cpus

        num_cpus = detect_num_cpus()
        print(f"Number of CPUs not provided, using detected number of CPUs: {num_cpus}")

    memory_limit = args.memory_limit
    if memory_limit is None:
        from tabarena.utils.resources import detect_memory_limit_gb

        memory_limit = detect_memory_limit_gb()
        print(f"Memory limit not provided, using detected memory size: {memory_limit} GB")

    batch, job = load_job(
        job_batch_dir=_strip_quotes(args.job_batch_dir),
        experiment_name=_strip_quotes(args.experiment),
        dataset=_strip_quotes(args.dataset),
        fold=args.fold,
        repeat=args.repeat,
    )
    setup_ray = resolve_setup_ray(args.setup_ray_for_slurm_shared_resources_environment, batch, job)
    if args.setup_ray_for_slurm_shared_resources_environment and not setup_ray:
        print(f"Ray not started: experiment {job.experiment.name!r} does not use Ray for this fit.")
    ray_temp_dir = setup_slurm_job(
        setup_ray_for_slurm_shared_resources_environment=setup_ray,
        num_cpus=num_cpus,
        num_gpus=args.num_gpus,
        memory_limit=memory_limit,
        ray_temp_root=args.ray_temp_root,
        log_to_driver=False,
    )
    failed = True
    try:
        run_job(
            batch,
            job,
            output_dir=args.output_dir,
            ignore_cache=args.ignore_cache,
            cleanup_on_failure=True,
            cpu_budget_check=args.cpu_budget_check,
            require_warmup=args.require_warmup,
        )
        failed = False
    finally:
        if ray_temp_dir is not None:
            if failed:
                copy_ray_worker_logs(ray_temp_dir, os.environ.get(RAY_LOG_DIR_ENV), job)
            if os.environ.get(KEEP_RAY_TEMP_DIR_ENV):
                print(f"Keeping Ray temp dir {ray_temp_dir} ({KEEP_RAY_TEMP_DIR_ENV} is set).")
            else:
                remove_ray_temp_dir(ray_temp_dir)
