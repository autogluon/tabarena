from __future__ import annotations

import argparse
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import openml


def setup_slurm_job(
    *,
    openml_cache_dir: str,
    num_cpus: int,
    num_gpus: int,
    memory_limit: int,
    setup_ray_for_slurm_shared_resources_environment: bool,
) -> None | str:
    """Ensure correct caching and usage of directories for OpenML and TabRepo.

    Parameters
    ----------
    openml_cache_dir : str
        The path to the OpenML cache directory, or "auto" to use the default OpenML cache directory.
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
    if openml_cache_dir == "auto":
        print("Using the default OpenML cache directory.")
    else:
        print(f"Setting OpenML cache directory to: {openml_cache_dir}")
        openml.config.set_root_cache_directory(root_cache_directory=openml_cache_dir)

    # SLURM save Ray setup in a shared resource system
    ray_dir = None
    if setup_ray_for_slurm_shared_resources_environment:
        print("Setting up Ray for SLURM job in a shared resources environment.")
        import logging
        import tempfile

        import ray

        ray_dir = tempfile.mkdtemp() + "/ray"

        min_plasma_storage_size = int(memory_limit * 0.5)
        ray_mem_in_b = int(int(memory_limit) * (1024.0**3))

        _plasma_directory = None
        dev_shm_size = ray._private.utils.get_shared_memory_bytes() / 1e9
        if dev_shm_size < min_plasma_storage_size:
            print(
                "WARNING: /dev/shm is full, switching to /tmp usage! "
                f"Available shared memory size: {dev_shm_size} GB, "
                f"Required minimum for Ray plasma store: {min_plasma_storage_size} GB."
            )
            # Likely slower but runs at least.
            _plasma_directory = ray_dir

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
            _plasma_directory=_plasma_directory,
        )
    return ray_dir


def _parse_yaml_config(
    *,
    configs_yaml_file: str,
    config_index: list[int] | None,
    num_cpus: int,
    num_gpus: int,
    memory_limit: int,
    sequential_local_fold_fitting: bool,
) -> list:
    """Parse the YAML configuration file and return a list of method configurations to run.

    Parameters
    ----------
    configs_yaml_file
        The path to the YAML file containing the configurations of all methods to run for the experiment.
    config_index
        The index of the configuration from the YAML file to run. If None, all configurations will be run.
    num_cpus
        The number of CPUs to use for the experiment.
    num_gpus
        The number of GPUs to use for the experiment.
    memory_limit
        The memory limit to use for the experiment.
    sequential_local_fold_fitting
        Whether to force to use sequential local fold fitting or not.

    Returns:
    -------
    methods: list
        Parsed TabArena experiment configurations to run, with resources and special model cases handled.
    """
    from tabarena.benchmark.experiment.experiment_constructor import (
        YamlExperimentSerializer,
        YamlSingleExperimentSerializer,
    )

    yaml_out = YamlExperimentSerializer.load_yaml(path=configs_yaml_file)
    methods = []
    for m_i, method in enumerate(yaml_out):
        if (config_index is not None) and (m_i not in config_index):
            continue

        if method["type"] == "AGExperiment":
            method["fit_kwargs"]["num_cpus"] = num_cpus
            method["fit_kwargs"]["num_gpus"] = num_gpus
            method["fit_kwargs"]["memory_limit"] = memory_limit
            methods.append(YamlSingleExperimentSerializer.parse_method(method))
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

    # TODO: Update
    #   - Make this a general purpose logic inside of TabArena code base to edit feature generator
    for m_i in range(len(methods)):
        preprocessing_name = methods[m_i].method_kwargs.pop(
            "preprocessing_pipeline", None
        )

        if (preprocessing_name is None) or (preprocessing_name == "default"):
            continue

        if preprocessing_name == "tabarena_default":
            print("=== Using new TabArena default preprocessing pipeline for method!")
            from tabarena.benchmark.preprocessing import (
                TabArenaModelAgnosticPreprocessing,
                TabArenaModelSpecificPreprocessing,
            )

            new_experiment = deepcopy(methods[m_i])
            new_experiment.method_kwargs["fit_kwargs"]["feature_generator"] = (
                TabArenaModelAgnosticPreprocessing()
            )
            new_experiment.method_kwargs["model_hyperparameters"] = (
                TabArenaModelSpecificPreprocessing.add_to_hyperparameters(
                    new_experiment.method_kwargs["model_hyperparameters"]
                )
            )
        elif preprocessing_name.startswith("FSBench__"):
            # Logic for feature selection benchmark
            from tabarena.benchmark.feature_selection_methods.feature_selection_benchmark_utils import (
                apply_fs_bench_preprocessing,
            )

            new_experiment = apply_fs_bench_preprocessing(
                preprocessing_name=preprocessing_name,
                experiment=methods[m_i],
            )

        else:
            raise ValueError(
                f"Preprocessing pipeline name '{preprocessing_name}' not recognized."
            )

        methods[m_i] = new_experiment

    return methods


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
    try:
        task_id_or_object = int(task_id_str)
    except ValueError:
        from tabarena.benchmark.task.user_task import UserTask

        task_id_or_object = UserTask.from_task_id_str(task_id_str)

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
    num_cpus: int,
    num_gpus: int,
    memory_limit: int,
    sequential_local_fold_fitting: bool,
    dynamic_tabarena_validation_protocol: bool,
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

    task_id_or_object = _parse_task_id(task_id)
    methods = _parse_yaml_config(
        configs_yaml_file=configs_yaml_file,
        config_index=config_index,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        memory_limit=memory_limit,
        sequential_local_fold_fitting=sequential_local_fold_fitting,
    )

    results_lst: dict[str, Any] = run_experiments_new(
        output_dir=output_dir,
        model_experiments=methods,
        tasks=[task_id_or_object],
        repetitions_mode="individual",
        repetitions_mode_args=[(fold, repeat)],
        cache_mode="ignore" if ignore_cache else "default",
        failure_on_non_finite_metric_error=True,
        dynamic_tabarena_validation_protocol=dynamic_tabarena_validation_protocol,
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


def _parse_int_list_or_none(s):
    if (s is None) or (s.lower() == "none") or (s.lower() == "null"):
        return None
    return _parse_int_list(s)


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
    parser.add_argument(
        "--sequential_local_fold_fitting",
        type=_str2bool,
        default=False,
        help="Whether to force to use sequential local fold fitting or not. If True, the "
        "experiment will be run without Ray. This might create a large speedup for some models.",
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
        help="Number of GPUs to use for the experiment.",
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
    parser.add_argument(
        "--dynamic_tabarena_validation_protocol",
        type=_str2bool,
        help="Whether to use the dynamic TabArena validation protocol or not. "
        "If True, the validation protocol will be dynamically updated based "
        "on the characteristics of the data for an experiment.",
        default=False,
    )
    args = parser.parse_args()

    num_cpus = args.num_cpus
    if num_cpus is None:
        from autogluon.common.utils.cpu_utils import get_available_cpu_count

        num_cpus = get_available_cpu_count(only_physical_cores=False)
        print(f"Number of CPUs not provided, using detected number of CPUs: {num_cpus}")

    memory_limit = args.memory_limit
    if memory_limit is None:
        from autogluon.common.utils.resource_utils import ResourceManager

        memory_limit = int(ResourceManager.get_memory_size(format="GB"))
        print(
            f"Memory limit not provided, using detected memory size: {memory_limit} GB"
        )

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
            num_cpus=num_cpus,
            num_gpus=args.num_gpus,
            memory_limit=memory_limit,
            sequential_local_fold_fitting=args.sequential_local_fold_fitting,
            dynamic_tabarena_validation_protocol=args.dynamic_tabarena_validation_protocol,
        )
    finally:
        if ray_temp_dir is not None:
            shutil.rmtree(ray_temp_dir)
