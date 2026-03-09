from __future__ import annotations

import argparse
import os
import shutil
from typing import Any

import openml


def setup_slurm_job(
    *,
    openml_cache_dir: str,
    tabrepo_cache_dir: str,
    setup_ray_for_slurm_shared_resources_environment: bool,
    num_cpus: int,
    num_gpus: int,
    memory_limit: int,
) -> None | str:
    """Ensure correct caching and usage of directories for OpenML and TabRepo.

    Parameters
    ----------
    openml_cache_dir : str
        The path to the OpenML cache directory.
    tabrepo_cache_dir : str
        The path to the TabRepo cache directory.
    num_cpus : int
        The number of CPUs to use for the experiment (needed for proper Ray setup).
    num_gpus : int
        The number of GPUs to use for the experiment (needed for proper Ray setup).
    memory_limit : int
        The memory limit to use for the experiment (needed for proper Ray setup).
    setup_ray_for_slurm_shared_resources_environment : bool
        If running on a SLURM cluster, we need to initialize Ray with extra options and a unique tempr dir.
        Otherwise, given the shared filesystem, Ray will try to use the same temp dir for all workers and
        crash (semi-randomly).
    """
    openml.config.set_root_cache_directory(root_cache_directory=openml_cache_dir)
    os.environ["TABREPO_CACHE"] = tabrepo_cache_dir

    # SLURM save Ray setup in a shared resource system
    ray_dir = None
    if setup_ray_for_slurm_shared_resources_environment:
        print("Setting up Ray for SLURM job in a shared resources environment.")
        import logging
        import tempfile

        import ray

        ray_dir = tempfile.mkdtemp() + "/ray"
        ray_mem_in_b = int(int(memory_limit) * (1024.0**3))
        ray.init(
            address="local",
            _memory=ray_mem_in_b,
            object_store_memory=int(ray_mem_in_b * 0.3),
            _temp_dir=ray_dir,
            include_dashboard=False,
            logging_level=logging.INFO,
            log_to_driver=True,
            num_gpus=num_gpus,
            num_cpus=num_cpus,
        )
    return ray_dir


def run_experiment(
    *,
    task_id: str,
    fold: int,
    repeat: int,
    configs_yaml_file: str,
    config_index: list[int] | None,
    output_dir: str,
    ignore_cache: bool,
    num_cpus: int,
    num_gpus: int,
    memory_limit: int,
    sequential_local_fold_fitting: bool,
):
    """Run an individual experiment for a given task id and dataset name.

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
    num_cpus : int
        The number of CPUs to use for the experiment.
    num_gpus : int
        The number of GPUs to use for the experiment.
    memory_limit : int
        The memory limit to use for the experiment.
    sequential_local_fold_fitting : bool
        Whether to use sequential local fold fitting or not. If True, the experiment will be run without
        Ray. This might create a large speedup for some models.
    """
    from tabarena.benchmark.experiment import run_experiments_new
    from tabarena.benchmark.experiment.experiment_constructor import (
        YamlExperimentSerializer,
        YamlSingleExperimentSerializer,
    )

    try:
        task_id_or_object = int(task_id)
    except ValueError:
        from tabarena.benchmark.task.user_task import UserTask

        task_id_or_object = UserTask.from_task_id_str(task_id)

    yaml_out = YamlExperimentSerializer.load_yaml(path=configs_yaml_file)
    methods = []
    for m_i, method in enumerate(yaml_out):
        if (config_index is not None) and (m_i not in config_index):
            continue

        if "method_kwargs" not in method:
            method["method_kwargs"] = {}

        # Logic to handle resources and special model cases
        if "fit_kwargs" not in method["method_kwargs"]:
            method["method_kwargs"]["fit_kwargs"] = {}
        method["method_kwargs"]["fit_kwargs"]["num_cpus"] = num_cpus
        method["method_kwargs"]["fit_kwargs"]["num_gpus"] = num_gpus
        method["method_kwargs"]["fit_kwargs"]["memory_limit"] = memory_limit

        if "model_hyperparameters" not in method:
            method["model_hyperparameters"] = {}
        if sequential_local_fold_fitting:
            if "ag_args_ensemble" not in method["model_hyperparameters"]:
                method["model_hyperparameters"]["ag_args_ensemble"] = {}
            method["model_hyperparameters"]["ag_args_ensemble"][
                "fold_fitting_strategy"
            ] = "sequential_local"

        methods.append(YamlSingleExperimentSerializer.parse_method(method))

    for m_i in range(len(methods)):
        preprocessing_name = methods[m_i].method_kwargs.pop(
            "preprocessing_pipeline", None
        )

        if preprocessing_name is not None:
            print("Adding preprocessing to the config:", preprocessing_name)
            from tabarena.benchmark.feature_selection_methods.feature_selection_methods_register import (
                PREPROCESSING_METHODS,
            )

            methods[m_i] = PREPROCESSING_METHODS[preprocessing_name](methods[m_i])

    results_lst: dict[str, Any] = run_experiments_new(
        output_dir=output_dir,
        model_experiments=methods,
        tasks=[task_id_or_object],
        repetitions_mode="individual",
        repetitions_mode_args=[(fold, repeat)],
        cache_mode="ignore" if ignore_cache else "default",
    )[0]
    print("Metric error:", results_lst["metric_error"])
    return results_lst


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_int_list(s):
    return [int(item) for item in s.split(",")]


if __name__ == "__main__":
    # TODO: provide defaults or a default CLI command to run this.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_id", type=str, required=True, help="OpenML Task ID for the task to run."
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
    # TODO: make sure that this can be None / missing
    parser.add_argument(
        "--config_index",
        type=parse_int_list,
        help="List of index of the configuration from YAML file to run.",
    )

    # TODO: improve usage, but required for a good setup
    parser.add_argument(
        "--openml_cache_dir", type=str, help="Path to the OpenML cache directory."
    )
    parser.add_argument(
        "--tabrepo_cache_dir", type=str, help="Path to the TabRepo cache directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory where the results will be saved.",
    )
    parser.add_argument(
        "--num_cpus", type=int, help="Number of CPUs to use for the experiment."
    )
    parser.add_argument(
        "--num_gpus", type=int, help="Number of GPUs to use for the experiment."
    )
    parser.add_argument(
        "--memory_limit", type=int, help="Memory limit to use for the experiment."
    )
    parser.add_argument(
        "--setup_ray_for_slurm_shared_resources_environment",
        type=str2bool,
        help="If True, setup Ray to work well in a shared resources environment with SLURM.",
    )

    # TODO: debug zone
    parser.add_argument(
        "--ignore_cache",
        type=str2bool,
        default=False,
        help="Whether to ignore the cache or not. If True, the cache will be ignored and the experiment will be run from scratch and potentially overwrite existing results.",
    )
    parser.add_argument(
        "--sequential_local_fold_fitting",
        type=str2bool,
        default=False,
        help="Whether to use sequential local fold fitting or not. If True, the experiment will be run without Ray. This might create a large speedup for some models.",
    )

    args = parser.parse_args()
    ray_temp_dir = setup_slurm_job(
        openml_cache_dir=args.openml_cache_dir,
        tabrepo_cache_dir=args.tabrepo_cache_dir,
        setup_ray_for_slurm_shared_resources_environment=args.setup_ray_for_slurm_shared_resources_environment,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        memory_limit=args.memory_limit,
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
            num_cpus=args.num_cpus,
            num_gpus=args.num_gpus,
            memory_limit=args.memory_limit,
            sequential_local_fold_fitting=args.sequential_local_fold_fitting,
        )
    finally:
        if ray_temp_dir is not None:
            shutil.rmtree(ray_temp_dir)

# python3 run_tabarena_fs_slurm.py --task_id 2 --fold 0 --repeat 0 --configs_yaml_file ../../../tabflow/configs/model_configs.yaml --openml_cache_dir ./openml_cache/ --tabrepo_cache_dir ./tabrepo_cache/ --output_dir ./results/ --num_cpus 4 --num_gpus 0 --memory_limit 16000 --ignore_cache False --sequential_local_fold_fitting True
