from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from tabflow_slurm.slurm_utils import setup_slurm_job


def _parse_task_id(task_id_str: str) -> int | object:
    """Parse the task id from a string and return either an int or a TabArena UserTask object.

    Parameters
    ----------
    task_id_str : str
        The task id of the OpenML task to run, or a string defining a local UserTask.

    Returns:
    -------
    int | openml.tasks.OpenMLTask
        The parsed task id, either as an int (for OpenML task IDs) or as
    """
    # Strip matching surrounding quotes that can survive when the command is built as a
    # string and passed through an extra (e.g. SLURM submission) layer without unquoting.
    # The task id itself never contains leading/trailing quotes, so this is always safe.
    task_id_str = task_id_str.strip()
    if len(task_id_str) >= 2 and task_id_str[0] == task_id_str[-1] and task_id_str[0] in ("'", '"'):
        task_id_str = task_id_str[1:-1]

    try:
        task_id_or_object = int(task_id_str)
    except ValueError:
        from tabarena.benchmark.task.user_task import UserTask

        task_id_or_object = UserTask.from_task_id_str(task_id_str)
        print(f"Loaded: User task with task hash: {task_id_or_object.task_id} from {task_id_str}")

    return task_id_or_object


def run_experiment(
    *,
    task_id: str,
    fold: int,
    repeat: int,
    configs_yaml_file: str,
    config_index: list[int] | None,
    output_dir: str,
    ignore_cache: bool,
):
    """Run an individual experiment for a given task id and dataset name.

    Compute resources, fold-fitting strategy, preprocessing, and the dynamic
    validation protocol are baked into each Experiment at build time, so they
    are no longer passed here.

    Parameters
    ----------
    task_id : str
        The task id of the OpenML task to run.
        If castable into int, it is assumed to be an OpenML task ID.
        If str, it defines a local OpenML task file.
    fold : int
        The fold to run.
    repeat : int
        The repeat to run. Here, repeat 0 means the first set of folds without any repeats.
    configs_yaml_file : str
        The path to the YAML file containing the configurations of all methods to run for the experiment.
    config_index : int | None
        The index of the configuration from the YAML file to run. If None, all configurations will be run.
    output_dir : str
        The path to the output directory where the results will be saved (and cached).
    ignore_cache : bool
        Whether to ignore the cache or not. If True, the cache will be ignored and the experiment will be
        run from scratch and potentially overwrite existing results.
    """
    from tabarena.benchmark.experiment import YamlExperimentSerializer, run_experiments_new

    task_id_or_object = _parse_task_id(task_id)
    methods = YamlExperimentSerializer.from_yaml(
        path=configs_yaml_file,
        config_index=config_index,
    )

    results_lst: dict[str, Any] = run_experiments_new(
        output_dir=output_dir,
        model_experiments=methods,
        tasks=[task_id_or_object],
        repetitions_mode="individual",
        repetitions_mode_args=[(fold, repeat)],
        cache_mode="ignore" if ignore_cache else "default",
        failure_on_non_finite_metric_error=True,
    )[0]
    print("Metric error:", results_lst["metric_error"])
    return results_lst


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _parse_int_list(s):
    return [int(item) for item in s.split(",")]


def _parse_int_or_none(s):
    if (s is None) or (s.lower() == "none") or (s.lower() == "null"):
        return None
    return int(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Require tasks settings
    parser.add_argument(
        "--task_id",
        type=str,
        required=True,
        help="OpenML Task ID for the task to run.",
    )
    parser.add_argument("--fold", type=int, required=True, help="Fold of CV to run.")
    parser.add_argument(
        "--repeat",
        type=int,
        required=True,
        help="Repeat of CV to run. Here, repeat 0 means the first set of folds without any repeats.",
    )
    parser.add_argument(
        "--configs_yaml_file",
        type=str,
        required=True,
        help="Path to the YAML file containing the configurations of all methods to run for the experiment.",
    )
    # Misc task setting
    parser.add_argument(
        "--config_index",
        type=_parse_int_list,
        help="List of index of the configuration from YAML file to run.",
        # Can be ommited to be None.
        default=None,
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
            config_index=args.config_index,
            task_id=args.task_id,
            fold=args.fold,
            repeat=args.repeat,
            configs_yaml_file=args.configs_yaml_file,
            output_dir=args.output_dir,
            ignore_cache=args.ignore_cache,
        )
    finally:
        if ray_temp_dir is not None:
            shutil.rmtree(ray_temp_dir)
