"""Code with hardcoded benchmark setup for generating input data to our SLURM job submission script.

Running this code will generate `slurm_run_data.json` with all the data required to run the array jobs
via `submit_template_gpu.sh`.

See `run_setup_slurm_jobs.py` for an example of how to use this code.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import ray
import yaml
from tabarena.benchmark.experiment.experiment_utils import check_cache_hit
from tabarena.utils.cache import CacheFunctionPickle
from tabarena.utils.ray_utils import ray_map_list, to_batch_list

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class BenchmarkSetup:
    """Manually set the parameters for the benchmark run."""

    # Required user input
    benchmark_name: str

    # Cluster Settings
    # ----------------
    base_path = "/work/dlclarge2/purucker-w/"
    """Base path for the project, code, and results. Within this directory, all results, code, and logs for TabArena will
    be saved. Adjust below as needed if more than one base path is desired. On a typical SLURM system, this base path
    should point to a persistent workspace that all your jobs can access.

    For our system, we used a structure as follows:
        - BASE_PATH
            - code              -- contains all code for the project (the dev install from TabArena and AutoGluon)
            - venvs             -- contains all virtual environments
            - input_data        -- contains all input data (e.g. TabRepo artifacts), this is not used so far
            - output            -- contains all output data from running the benchmark
            - slurm_out         -- contains all SLURM output logs
            - .openml-cache     -- contains the OpenML cache
    """
    python_from_base_path: str = "venvs/tabarena_fs/bin/python"
    """Python executable and environment to use for the SLURM jobs. This should point to a Python
    executable within a (virtual) environment."""
    run_script_from_base_path: str = "examples/running_tabarena_feature_selection_methods/slurm/tabflow_slurm/run_tabarena_fs_slurm.py"
    """Python script to run the benchmark. This should point to the script that runs the benchmark
    for TabArena."""
    openml_cache_from_base_path: str = ".openml-cache"
    """OpenML cache directory. This is used to store dataset and tasks data from OpenML."""
    tabrepo_cache_dir_from_base_path: str = "input_data/tabrepo"
    """TabRepo cache directory."""
    slurm_log_output_from_base_path: str = "slurm_out/new_runs"
    """Directory for the SLURM output logs. This is used to store the output logs from the
    SLURM jobs."""
    output_dir_base_from_base_path: str = "output/"
    """Output directory for the benchmark. In this folder a `benchmark_name` folder will be created."""
    configs_path_from_base_path: str = "code/tabarena_new/tabarena/tabflow_slurm/benchmark_configs_"
    """YAML file with the configs to run. Generated from parameters above in code below.
    File path is f"{self.base_path}{self.configs_path_from_base_path}{self.benchmark_name}.yaml"
    """
    slurm_script: str = "submit_template.sh"
    """Name of the SLURM (array) script that to run on the cluster (only used to print the command
     to run)."""
    slurm_gpu_partition: str = "alldlc2_gpu-l40s"
    """SLURM partition to use for GPU jobs. Adjust as needed for your cluster setup."""
    slurm_cpu_partition: str = "bosch_cpu-cascadelake"
    """SLURM partition to use for CPU jobs. Adjust as needed for your cluster setup."""

    # Task/Data Settings
    # ------------------
    custom_metadata: pd.DataFrame | None = None
    """Custom metadata to use for defining the tasks and datasets to run.

    The metadata must have the following columns:
        "tabarena_num_repeats": int
            The number of repeats for the task based on the protocol from TabArena.
            See tabarena.nips2025_utils.fetch_metadata._get_n_repeats for details.
        "num_folds": int
            The number of folds for the task.
        "task_id": str
            The task ID for the task as an int.
            If a local task, we assume this to be `UserTask.task_id_str`.
        "num_instances": int
            The number of instances/samples in the dataset.
        "num_features" : int
            The number of features in the dataset.
        "num_classes": int
            The number of classes in the dataset. For regression tasks, this value is
            ignored.
        "problem_type": str
            The problem type of the task. Options: "binary", "regression", "multiclass"
    """
    tabarena_lite: bool = False
    """Run only TabArena-Lite, that is: only the first split of each dataset, and the default
    configuration and up to `tabarena_lite_n_configs` random configs."""
    problem_types_to_run: list[str] = field(
        # Options: "binary", "regression", "multiclass"
        default_factory=lambda: [
            "binary",
            "multiclass",
            "regression",
        ]
    )

    # Benchmark Settings
    # ------------------
    """Problem types to run in the benchmark. Adjust as needed to run only specific problem types."""
    num_cpus: int = 8
    """Number of CPUs to use for the SLURM jobs. The number of CPUs available on the node."""
    num_gpus: int = 0
    """Number of GPUs to use for the SLURM jobs. The number of GPUs available on the node."""
    memory_limit: int = 32
    """Memory/RAM limit for the SLURM jobs in GB. The memory limit available on the node."""
    time_limit: int = 3600
    """Time limit for each fit (all 8 folds) of a model in seconds. By default, 3600 seconds is used."""
    n_random_configs: int = 200
    """Number of random hyperparameter configurations to run for each model"""
    models: list[tuple[str, int | str]] = field(default_factory=list)
    """List of models to run in the benchmark with metadata.
    Metadata keys from left to right:
        - model name: str
        - number of random hyperparameter configurations to run: int or str
            Some special cases are:
                - If 0, only the default configuration is run.
                - If "all", `n_random_configs`-many configurations are run.

    Remove or comment out models to that you do not want to run.
    Examples from the current state of TabArena are:
    default_factory=lambda: [
            ("TabDPT", "all"),
            ("TabICL", "all"),
            ("TabPFNv2", "all"),
            ("Mitra", "all"),
            # -- Neural networks
            ("TabM", "all"),
            ("RealMLP", "all"),
            ("ModernNCA", "all"),
            ("FastaiMLP", "all"),
            ("TorchMLP", "all"),
            # -- Tree-based models
            ("CatBoost", "all"),
            ("EBM", "all"),
            ("ExtraTrees", "all"),
            ("LightGBM", "all"),
            ("RandomForest", "all"),
            ("XGBoost", "all"),
            # -- Baselines
            ("KNN", "all"),
            ("Linear", "all"),
            # -- Other
            ("xRFM", "all"),
        ]
    )
    For the newest set of available models, see:
    `tabarena.models.utils.get_configs_generator_from_name`
    """
    fs_methods: list[tuple[str, int | str]] = field(default_factory=list)
    """List of fs_methods to run in the benchmark with metadata.
        Metadata keys from left to right:
            - fs_method name: str
            - number of random hyperparameter configurations to run: int or str
                Some special cases are:
                    - If 0, only the default configuration is run.
                    - If "all", `n_random_configs`-many configurations are run."""
    configs_per_job: int = 5
    """Batching of several experiments per job. This is used to reduce the number of SLURM jobs.
    Adjust the time limit in the slurm_script accordingly."""
    setup_ray_for_slurm_shared_resources_environment: bool = True
    """Prepare Ray for a SLURM shared resource environment. This is used to setup Ray for SLURM
    shared resources. Recommended to set to True if sequential_local_fold_fitting is False."""
    preprocessing_pipelines: list[str] = field(default_factory=lambda: ["default"])
    """EXPERIMENTAL, REQUIRES A CUSTOM AUTOGLUON BRANCH!
    Preprocessing pipelines to add to the configurations we want to run.

    Each options multiplies the number of configurations to run by the number of
    pipelines. For example, if we have 10 configurations and 2 pipelines, we will
    run 20 configurations.

    Options:
        - "default": Use the default preprocessing pipeline.
        - Any other string registered in `tabarena.benchmark.preprocessing.preprocessing_register`.
    """
    custom_model_constraints: dict[str, dict[str, int]] | None = None
    """Custom mapping of model names to constraints to filter which models runs on
    what kind of datasets.

    For each model, provide a dictionary with the constraints for that model and
    the model AG Key as the name.

    Each constraint is a dictionary with keys:
        - "max_n_features": int
            Maximal number of features.
        - "max_n_samples_train_per_fold": int
        - "min_n_samples_train_per_fold": int
        - "max_n_classes": int
            Maximal number of classes.
        - "regression_support": bool
            False, if the model does not support regression.

    All keys are optional and can be omitted if there is no constraint for that key.

    Example for TabPFNv2:
        custom_model_constraints = {
            "TABPFNV2": {
                    "max_n_samples_train_per_fold": 10_000,
                    "max_n_features": 500,
                    "max_n_classes": 10,
            },
            "TABICL": {
                    "max_n_samples_train_per_fold": 100_000,
                    "max_n_features": 500,
                    "regression_support": False,
            }
        }
    """
    custom_fs_method_constraints: dict[str, dict[str, int]] | None = None
    """Custom mapping of fs_method names to constraints to filter which fs_methods runs on
    what kind of datasets.
    For each fs_method, provide a dictionary with the constraints for that fs_method as the name.
    Each constraint is a dictionary with keys:
        - "only_numerical_columns": bool,
        - "accepts_missing_values": bool
    All keys are optional and can be omitted if there is no constraint for that key."""

    # Misc Settings
    # -------------
    ignore_cache: bool = False
    """If True, will overwrite the cache and run all jobs again."""
    cache_cls: CacheFunctionPickle = CacheFunctionPickle
    """How to save the cache. Pickle is the current recommended default. This option and the two
    below must be in sync with the cache method in run_script."""
    cache_cls_kwargs: dict = field(default_factory=lambda: {"include_self_in_call": True})
    """Arguments for the cache class. This is used to setup the cache class for the benchmark."""
    cache_path_format: str = "name_first"
    """Path format for the cache. This is used to setup the cache path format for the benchmark."""
    num_ray_cpus = 8
    """Number of CPUs to use for checking the cache and generating the jobs. This should be set to the number of CPUs
    available to the python script."""
    sequential_local_fold_fitting: bool = False
    """Use Ray for local fold fitting. This is used to speed up the local fold fitting and force
    this behavior if True. If False the default strategy of running the local fold fitting is used,
    as determined by AutoGluon and the model's default_ag_args_ensemble parameters. Should only be used for
    debugging anymore."""
    model_artifacts_base_path: str | Path | None = "/tmp/ag"  # noqa: S108
    """Adapt the default temporary directory used for model artifacts in TabArena.
        - If None, the default temporary directory is used: "./AutoGluonModels".
        - If a string or Path, the directory is used as the base path for the temporary
        and any model artifacts will be stored in time-stamped subdirectories.
    """
    fs_method_artifacts_base_path: str | Path | None = "/tmp/ag_fs"  # noqa: S108
    """Adapt the default temporary directory used for model artifacts in TabArena.
        - If None, the default temporary directory is used: "./AutoGluonModels".
        - If a string or Path, the directory is used as the base path for the temporary
        and any model artifacts will be stored in time-stamped subdirectories.
    """
    seed_config: str = "fold-config-wise"
    """Method to set the seeds of models when benchmarking:
        - "static" for static seed across all model fits
        - "fold-wise" for different seed per fold
        - "fold-config-wise" for different seeds per fold and configuration

    "static" was the default for the original TabArena benchmark, but we recommend and
    default to "fold-config-wise" now to better capture variability across different
    folds and configurations in benchmarking.
    """
    fake_memory_for_estimates: int | None = None
    """Experimental parameter that is to be ignored!

    If not None, we use this value to fake the amount of available (CPU) memory
    such that the memory estimates for a model are compared this value instead
    of the actually available memory on the system.

    Values in GB as `memory_limit`.

    This can be useful if:
        - To test or overrule (bad) memory estimates.
        - For models that use CPU memory as a proxy for GPU memory (e.g. most TFMs),
          this can be used if the job has much more VRAM than CPU memory.
    """
    # TODO: solve this properly in the future.
    parallel_benchmark_fix: str | None = None
    """Experimental parameter that is to be ignored!

    Set this is to some string value to make sure you can run parallel jobs for the same
    benchmark name.
    """

    @property
    def _safe_benchmark_name(self) -> str:
        """Safe benchmark name for file paths."""
        benchmark_name = self.benchmark_name
        if self.parallel_benchmark_fix is not None:
            benchmark_name += f"_{self.parallel_benchmark_fix}"
        return benchmark_name

    @property
    def slurm_job_json(self) -> str:
        """JSON file with the job data to run used by SLURM. This is generated from the configs and metadata."""
        return f"slurm_run_data_{self._safe_benchmark_name}.json"

    @property
    def configs(self) -> str:
        """YAML file with the configs to run. Generated from parameters above in code below."""
        return f"{self.base_path}{self.configs_path_from_base_path}{self._safe_benchmark_name}.yaml"

    @property
    def output_dir(self) -> str:
        """Output directory for the benchmark."""
        return self.base_path + self.output_dir_base_from_base_path + self.benchmark_name

    @property
    def metadata(self) -> str:
        """Dataset/task Metadata for TabArena."""
        return self.base_path + self.metadata_from_base_path

    @property
    def python(self) -> str:
        """Python executable to use for the SLURM jobs."""
        return self.base_path + self.python_from_base_path

    @property
    def run_script(self) -> str:
        """Python script to run the benchmark."""
        return self.base_path + self.run_script_from_base_path

    @property
    def openml_cache(self) -> str:
        """OpenML cache directory."""
        return self.base_path + self.openml_cache_from_base_path

    @property
    def tabrepo_cache_dir(self) -> str:
        """TabRepo cache directory."""
        return self.base_path + self.tabrepo_cache_dir_from_base_path

    @property
    def slurm_log_output(self) -> str:
        """Directory for the SLURM output logs."""
        return self.base_path + self.slurm_log_output_from_base_path

    @property
    def slurm_base_command(self):
        """SLURM command to run the benchmark.

        We set the following parameters based on the benchmark setup:
            - slurm script
            - partition
            - gres (including GPUs)
            - time
            - cpus
            - memory
        """
        is_gpu_job = self.num_gpus > 0

        partition = self.slurm_gpu_partition if is_gpu_job else self.slurm_cpu_partition
        partition = "--partition=" + partition

        gres = f"gpu:{self.num_gpus},localtmp:100" if is_gpu_job else "localtmp:100"
        gres = f"--gres={gres}"

        time_in_h = self.time_limit // 3600 * self.configs_per_job + 1
        time_in_h = f"--time={time_in_h}:00:00"
        cpus = f"--cpus-per-task={self.num_cpus}"
        if is_gpu_job:
            mem = f"--mem-per-gpu={self.memory_limit}G"
        else:
            mem = f"--mem-per-cpu={self.memory_limit//self.num_cpus}G"
        script = str(Path(__file__).parent / self.slurm_script)

        return f"{partition} {gres} {time_in_h} {cpus} {mem} {script}"

    def get_jobs_to_run(self):  # noqa: C901
        """Determine all jobs to run by checking the cache and filtering
        invalid jobs.
        """
        Path(self.openml_cache).mkdir(parents=True, exist_ok=True)
        Path(self.tabrepo_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.slurm_log_output).mkdir(parents=True, exist_ok=True)

        if self.custom_metadata is None:
            from tabarena.nips2025_utils.fetch_metadata import (
                load_curated_task_metadata,
            )

            metadata = load_curated_task_metadata()
        else:
            metadata = deepcopy(self.custom_metadata)

        self.generate_configs_yaml()
        # Read YAML file and get the number of configs
        with open(self.configs) as file:
            configs = yaml.safe_load(file)["methods"]

        def yield_all_jobs():
            for row in metadata.itertuples():
                task_id = row.task_id
                n_samples_train_per_fold = int(row.num_instances - int(row.num_instances / row.num_folds))
                n_features = int(row.num_features)
                n_classes = int(row.num_classes) if row.problem_type in ["binary", "multiclass"] else 0

                # Quick, model independent skip.
                if row.problem_type not in self.problem_types_to_run:
                    continue

                repeats_folds = product(range(int(row.tabarena_num_repeats)), range(int(row.num_folds)))
                if self.tabarena_lite:  # Filter to only first split.
                    repeats_folds = list(repeats_folds)[:1]

                for repeat_i, fold_i in repeats_folds:
                    for config_index, config in list(enumerate(configs)):
                        yield {
                            "config_index": config_index,
                            "config": config,
                            "task_id": task_id,
                            "fold_i": fold_i,
                            "repeat_i": repeat_i,
                            "n_samples_train_per_fold": n_samples_train_per_fold,
                            "n_features": n_features,
                            "n_classes": n_classes,
                        }

        jobs_to_check = list(yield_all_jobs())

        # Check cache and filter invalid jobs in parallel using Ray
        if ray.is_initialized:
            ray.shutdown()
        ray.init(num_cpus=self.num_ray_cpus)
        output = ray_map_list(
            list_to_map=list(to_batch_list(jobs_to_check, 10_000)),
            func=should_run_job_batch,
            func_element_key_string="input_data_list",
            num_workers=self.num_ray_cpus,
            num_cpus_per_worker=1,
            func_kwargs={
                "output_dir": self.output_dir,
                "cache_path_format": self.cache_path_format,
                "cache_cls": self.cache_cls,
                "cache_cls_kwargs": self.cache_cls_kwargs,
                "models_to_constraints": self.models_to_constraints,
            },
            track_progress=True,
            tqdm_kwargs={"desc": "Checking Cache and Filter Invalid Jobs"},
        )
        output = [item for sublist in output for item in sublist]  # Flatten the batched list
        to_run_job_map = {}
        for run_job, job_data in zip(output, jobs_to_check, strict=True):
            if run_job:
                job_key = (
                    job_data["task_id"],
                    job_data["fold_i"],
                    job_data["repeat_i"],
                )
                if job_key not in to_run_job_map:
                    to_run_job_map[job_key] = []
                to_run_job_map[job_key].append(job_data["config_index"])

        # Convert the map to a list of jobs
        jobs = []
        to_run_jobs = 0
        for job_key, config_indices in to_run_job_map.items():
            to_run_jobs += len(config_indices)
            for config_batch in to_batch_list(config_indices, self.configs_per_job):
                jobs.append(
                    {
                        "task_id": job_key[0],
                        "fold": job_key[1],
                        "repeat": job_key[2],
                        "config_index": config_batch,
                    },
                )

        print(f"Generated {to_run_jobs} jobs to run without batching.")
        print(f"Jobs with batching: {len(jobs)}")
        return jobs

    def generate_configs_yaml(self):
        """Generate the YAML file with the configurations to run based
        on specific models to run.
        """
        from tabarena.benchmark.experiment import (
            AGModelBagExperiment,
            YamlExperimentSerializer,
        )
        from tabarena.models.utils import get_configs_generator_from_name

        experiments_lst = []
        fs_experiments_lst = []
        method_kwargs = {}
        fs_method_kwargs = {}
        if self.model_artifacts_base_path is not None:
            method_kwargs["init_kwargs"] = {"default_base_path": self.model_artifacts_base_path}
        if self.fs_method_artifacts_base_path is not None:
            fs_method_kwargs["init_kwargs"] = {"default_base_path": self.fs_method_artifacts_base_path}

        print(
            "Generating experiments for models & fs_methods...",
            f"\n\t`all` := number of configs: {self.n_random_configs}",
            f"\n\t{len(self.models) * len(self.fs_methods)} models: {self.models}, fs_methods: {self.fs_methods}",
            f"\n\t{len(self.preprocessing_pipelines)} preprocessing pipelines: {self.preprocessing_pipelines}",
            f"\n\tMethod kwargs: {method_kwargs}",
        )
        for preprocessing_name in self.preprocessing_pipelines:
            pipeline_method_kwargs = deepcopy(method_kwargs)

            name_id_suffix = ""
            if preprocessing_name != "default":
                pipeline_method_kwargs["preprocessing_pipeline"] = preprocessing_name
                name_id_suffix = f"_{preprocessing_name}"

            for model_name, n_configs in self.models:
                if isinstance(n_configs, str) and n_configs == "all":
                    n_configs = self.n_random_configs
                elif not isinstance(n_configs, int):
                    raise ValueError(
                        f"Invalid number of configurations for model {model_name}: {n_configs}. "
                        "Must be an integer or 'all'."
                    )
                config_generator = get_configs_generator_from_name(model_name)
                experiments_lst.append(
                    config_generator.generate_all_bag_experiments(
                        num_random_configs=n_configs,
                        add_seed=self.seed_config,
                        name_id_suffix=name_id_suffix,
                        method_kwargs=pipeline_method_kwargs,
                        time_limit=self.time_limit,
                    )
                )
            for experiment in experiments_lst:
                for fs_method_name, n_configs in self.fs_methods:
                    config_generator = get_configs_generator_from_name(fs_method_name)
                    fs_experiments_lst.append(
                        config_generator.generate_all_bag_experiments(
                            num_random_configs=n_configs,
                            add_seed=self.seed_config,
                            name_id_suffix=name_id_suffix,
                            method_kwargs=pipeline_method_kwargs,
                            time_limit=self.time_limit,
                        )
                    )

        # Post Process experiment list
        experiments_all: list[AGModelBagExperiment] = [
            exp for exp_family_lst in fs_experiments_lst for exp in exp_family_lst
        ]

        # Verify no duplicate names
        experiment_names = set()
        for experiment in experiments_all:
            if experiment.name not in experiment_names:
                experiment_names.add(experiment.name)
            else:
                raise AssertionError(
                    f"Found multiple instances of experiment named {experiment.name}. All experiment names must be unique!",
                )

        YamlExperimentSerializer.to_yaml(experiments=experiments_all, path=self.configs)

    def get_jobs_dict(self):
        """Get the jobs to run as a dictionary with default arguments and jobs."""
        jobs = list(self.get_jobs_to_run())

        # Fake memory limit for estimates if needed
        memory_limit = self.memory_limit
        if self.fake_memory_for_estimates is not None:
            memory_limit = self.fake_memory_for_estimates

        default_args = {
            "python": self.python,
            "run_script": self.run_script,
            "openml_cache_dir": self.openml_cache,
            "configs_yaml_file": self.configs,
            "tabrepo_cache_dir": self.tabrepo_cache_dir,
            "output_dir": self.output_dir,
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
            "memory_limit": memory_limit,
            "setup_ray_for_slurm_shared_resources_environment": self.setup_ray_for_slurm_shared_resources_environment,
            "ignore_cache": self.ignore_cache,
            "sequential_local_fold_fitting": self.sequential_local_fold_fitting,
        }
        return {"defaults": default_args, "jobs": jobs}

    def setup_jobs(self):
        """Setup the jobs to run by generating the SLURM job JSON file."""
        jobs_dict = self.get_jobs_dict()
        n_jobs = len(jobs_dict["jobs"])
        if n_jobs == 0:
            print("No jobs to run.")
            Path(self.slurm_job_json).unlink(missing_ok=True)
            Path(self.configs).unlink(missing_ok=True)
            return

        with open(self.slurm_job_json, "w") as f:
            json.dump(jobs_dict, f)

        print(
            f"##### Setup Jobs for {self._safe_benchmark_name}"
            "\nRun the following command to start the jobs:"
            f"\nsbatch --array=0-{n_jobs - 1}%100 {self.slurm_base_command} {self.slurm_job_json}"
            "\n"
        )

    @property
    def models_to_constraints(self) -> dict[str, dict[str, int]]:
        """Mapping of model names to their constraints.

        Returns:
        --------
        model_constraints: dict[str, dict[str, int]]
            Mapping of model names to their constraints.
            Each constraint is a dictionary with keys:
                - "max_n_features": int
                    Maximal number of features.
                - "max_n_samples_train_per_fold": int
                - "min_n_samples_train_per_fold": int
                - "max_n_classes": int
                    Maximal number of classes.
                - "regression_support": bool
                    False, if the model does not support regression.
            Keys are optional and will be omitted if there is no constraint for that key.
        """
        model_constrains = {}

        # TabICL Subset
        for model in ["TA-TABICL", "TABICL"]:
            model_constrains[model] = {
                "max_n_samples_train_per_fold": 100_000,
                "max_n_features": 500,
                "regression_support": False,
            }

        # TabPFNv2 Subset
        for model in ["TABPFNV2", "TA-TABPFNV2", "MITRA"]:
            model_constrains[model] = {
                "max_n_samples_train_per_fold": 10_000,
                "max_n_features": 500,
                "max_n_classes": 10,
            }

        if self.custom_model_constraints is not None:
            model_constrains = {
                **model_constrains,
                **self.custom_model_constraints,
            }

        return model_constrains

    @staticmethod
    def are_model_constraints_valid(
        *,
        model_cls: str,
        n_features: int,
        n_classes: int,
        n_samples_train_per_fold: int,
        models_to_constraints: dict[str, dict[str, int]],
    ) -> bool:
        """Check if the model constraints are valid for the given model and dataset.

        Arguments:
        ----------
        model_cls: str
            The name of the model class to check. AG key of abstract model class.
        n_features: int
            The number of features in the dataset.
        n_classes: int
            The number of classes in the dataset.
            0 for regression tasks.
        n_samples_train_per_fold: int
            The number of training samples per fold in the dataset.
        models_to_constraints: dict[str, dict[str, int]]
            Mapping of model names to their potential constraints.

        Returns:
        --------
        model_is_valid: bool
            True if the model can be run on the dataset, False otherwise.
        """
        model_constraints = models_to_constraints.get(model_cls)
        if model_constraints is None:
            return True  # No constraints for this model

        regression_support = model_constraints.get("regression_support", True)
        if (n_classes == 0) and (not regression_support):
            return False

        max_n_features = model_constraints.get("max_n_features", None)
        if (max_n_features is not None) and (n_features > max_n_features):
            return False

        max_n_samples_train_per_fold = model_constraints.get("max_n_samples_train_per_fold", None)
        if (max_n_samples_train_per_fold is not None) and (n_samples_train_per_fold > max_n_samples_train_per_fold):
            return False

        min_n_samples_train_per_fold = model_constraints.get("min_n_samples_train_per_fold", None)
        if (min_n_samples_train_per_fold is not None) and (n_samples_train_per_fold < min_n_samples_train_per_fold):
            return False

        max_n_classes = model_constraints.get("max_n_classes", None)
        if (max_n_classes is not None) and (n_classes > max_n_classes):
            return False

        # All constraints are valid
        return True


def should_run_job_batch(*, input_data_list: list[dict], **kwargs) -> list[bool]:
    """Batched version for Ray."""
    return [should_run_job(input_data=data, **kwargs) for data in input_data_list]


def should_run_job(
    *,
    input_data: dict,
    output_dir: str,
    cache_path_format: str,
    cache_cls: CacheFunctionPickle,
    cache_cls_kwargs: dict,
    models_to_constraints: dict,
) -> bool:
    """Check if a job should be run based on the configuration and cache.
    Must be not a class function to be used with Ray.
    """
    config = input_data["config"]
    task_id = input_data["task_id"]
    fs_method = input_data["fs_method"]
    fold_i = input_data["fold_i"]
    repeat_i = input_data["repeat_i"]

    # Check if local task or not
    try:
        task_id = int(task_id)
    except ValueError:
        task_id = task_id.split("|", 2)[1]  # Extract the local task ID if it is a UserTask.task_id_str

    # Filter out-of-constraints datasets
    if not BenchmarkSetup.are_model_constraints_valid(
        model_cls=config["model_cls"],
        n_features=input_data["n_features"],
        n_classes=input_data["n_classes"],
        n_samples_train_per_fold=input_data["n_samples_train_per_fold"],
        models_to_constraints=models_to_constraints,
    ):
        return False

    return not check_cache_hit(
        result_dir=output_dir,
        method_name=config["name"],
        task_id=task_id,
        fold=fold_i,
        repeat=repeat_i,
        cache_path_format=cache_path_format,
        cache_cls=cache_cls,
        cache_cls_kwargs=cache_cls_kwargs,
        mode="local",
    )