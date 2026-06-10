"""Run one benchmark work unit on a (SLURM) compute node.

The unit is a core ``Job``: an experiment (referenced *by name*) on one
``(dataset, fold, repeat)`` split. Both sides of the reference live in the shipped
:class:`~tabarena.benchmark.experiment.job_batch.JobBatch` artifact
(``--job_batch_dir``): the experiment is loaded from its ``experiments.yaml`` and the
dataset resolves against its ``task_metadata.csv`` — so this runner executes through
exactly the same :meth:`ExperimentBatchRunner.run_jobs` path (same task resolution,
results naming, and cache layout) as a local benchmark run.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tabflow_slurm.slurm_utils import setup_slurm_job


def run_experiment(
    *,
    job_batch_dir: str,
    experiment_name: str,
    dataset: str,
    fold: int,
    repeat: int,
    output_dir: str,
    ignore_cache: bool,
) -> list[dict]:
    """Run a single ``(experiment, dataset, fold, repeat)`` work unit from a job batch.

    Compute resources, fold-fitting strategy, preprocessing, and the dynamic
    validation protocol are baked into each serialized experiment at build time,
    so they are not passed here.

    Parameters
    ----------
    job_batch_dir : str
        Directory of the ``JobBatch`` artifact written at setup time (experiments +
        task metadata + job coordinates).
    experiment_name : str
        The experiment to run, by its unique name in the batch's ``experiments.yaml``.
    dataset : str
        The dataset to run on (the collection's ``tabarena_task_name`` — also the
        results ``dataset`` key). A local ``UserTask`` (e.g. a materialized Data
        Foundry task) is auto-resolved from the collection's ``task_id_str``.
    fold : int
        The fold to run.
    repeat : int
        The repeat to run. Here, repeat 0 means the first set of folds without any repeats.
    output_dir : str
        The path to the output directory where the results will be saved (and cached).
    ignore_cache : bool
        Whether to ignore the cache or not. If True, the cache will be ignored and the
        experiment will be run from scratch and potentially overwrite existing results.
    """
    from tabarena.benchmark.experiment import ExperimentBatchRunner, Job, JobBatch

    batch = JobBatch.load(job_batch_dir)
    experiment_by_name = {experiment.name: experiment for experiment in batch.experiments}
    experiment = experiment_by_name.get(experiment_name)
    if experiment is None:
        raise ValueError(
            f"Experiment {experiment_name!r} is not in the job batch at {job_batch_dir!r} "
            f"(has: {sorted(experiment_by_name)}).",
        )

    runner = ExperimentBatchRunner(
        expname=output_dir,
        task_metadata=batch.task_metadata,
        cache_mode="ignore" if ignore_cache else "default",
        # Benchmark mode: record model failures instead of debugger-friendly behavior.
        debug_mode=False,
    )
    results_lst = runner.run_jobs([Job.create(experiment, dataset, fold=fold, repeat=repeat)])
    for results in results_lst:
        print("Metric error:", results.get("metric_error"))
    return results_lst


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


if __name__ == "__main__":
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
    # TODO: debug zone
    parser.add_argument(
        "--ignore_cache",
        type=_str2bool,
        default=False,
        help="Whether to ignore the cache or not. If True, the cache will be ignored and "
        "the experiment will be run from scratch and potentially overwrite existing results.",
    )
    # Experiment environment settings
    parser.add_argument(
        "--openml_cache_dir",
        type=str,
        help="Path to the OpenML cache directory or 'auto'.",
        default="auto",
    )
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
        help="If True, setup Ray to work well in a shared resources environment with SLURM.",
        default=False,
    )
    args = parser.parse_args()

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

    ray_temp_dir = setup_slurm_job(
        openml_cache_dir=args.openml_cache_dir,
        setup_ray_for_slurm_shared_resources_environment=args.setup_ray_for_slurm_shared_resources_environment,
        num_cpus=num_cpus,
        num_gpus=args.num_gpus,
        memory_limit=memory_limit,
    )
    try:
        run_experiment(
            job_batch_dir=_strip_quotes(args.job_batch_dir),
            experiment_name=_strip_quotes(args.experiment),
            dataset=_strip_quotes(args.dataset),
            fold=args.fold,
            repeat=args.repeat,
            output_dir=args.output_dir,
            ignore_cache=args.ignore_cache,
        )
    finally:
        if ray_temp_dir is not None:
            shutil.rmtree(ray_temp_dir)
