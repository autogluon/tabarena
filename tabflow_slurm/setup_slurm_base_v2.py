from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
import ray
import yaml
from tabarena.benchmark.experiment.experiment_utils import check_cache_hit
from tabarena.benchmark.task.user_task import SplitMetadata, TabArenaTaskMetadata
from tabarena.utils.cache import CacheFunctionPickle
from tabarena.utils.ray_utils import ray_map_list, to_batch_list


@dataclass
class PathSetup:
    """Configure paths for the benchmark."""

    base_path: str = "/work/dlclarge2/purucker-tabarena/"
    """Base path for the project, code, and results. Within this directory,
    all results, code, and logs for TabArena will be saved. Adjust below as needed if
    more than one base path is desired. On a typical SLURM system, this base path
    should point to a persistent workspace that all your jobs can access.

    For our system, we used a structure as follows:
        - BASE_PATH
            - code              -- contains all code for the project (the dev install
                                    from TabArena and AutoGluon)
            - venvs             -- contains all virtual environments
            - output            -- contains all output data from running the benchmark
            - slurm_out         -- contains all SLURM output logs
            - .openml-cache     -- contains the OpenML cache
    """
    venv_name: str = "tabarena_14022026"
    """Python venv to use for the SLURM jobs."""
    tabarena_repo_name: str = "tabarena_new"
    """Name of the local tabarena repository.
    Used to determine the path to the run script"""
    openml_cache_from_base_path: str | Literal["auto"] = ".openml-cache"
    """OpenML cache directory. This is used to store dataset and tasks data from OpenML.

    If "auto", we use the default cache from OpenML. If any other string, this is
    interpreted as the path to the folder for a custom OpenML cache.
    """
    slurm_log_output_from_base_path: str = "slurm_out/"
    """Directory for the SLURM output logs. In this folder a `benchmark_name`
    folder will be created and used to store the output logs from the SLURM jobs."""
    output_dir_base_from_base_path: str = "output/"
    """Output directory for the benchmark. In this folder a `benchmark_name`
    folder will be created."""

    @property
    def python_path(self) -> str:
        """Python executable to use for the SLURM jobs."""
        return self.base_path + f"venvs/{self.venv_name}/bin/python"

    @property
    def openml_cache_path(self) -> str:
        """OpenML cache directory."""
        if self.openml_cache_from_base_path == "auto":
            return self.openml_cache_from_base_path
        return self.base_path + self.openml_cache_from_base_path

    @property
    def run_script_path(self) -> str:
        """Python script to run the benchmark.
        This should point to the script that runs the benchmark for TabArena.
        """
        return self.base_path + (
            f"code/{self.tabarena_repo_name}/tabarena"
            f"/tabflow_slurm/run_tabarena_experiment.py"
        )

    @property
    def configs_base_path(self) -> str:
        """YAML file with the configs to run.

        File path is f"{self.base_path}{self.configs_path_from_base_path}
        {self._safe_benchmark_name}.yaml".
        """
        return (
            f"code/{self.tabarena_repo_name}/tabarena/tabflow_slurm/benchmark_configs_"
        )

    def get_slurm_job_json_path(self, safe_benchmark_name: str) -> str:
        """JSON file with the job data to run used by SLURM.
        This is generated from the configs and metadata.
        """
        # TODO: change UX for config and slurm paths.
        path_to_config_file = str(Path(self.configs_base_path).parent) + "/"
        return (
            f"{self.base_path}{path_to_config_file}"
            f"slurm_run_data_{safe_benchmark_name}.json"
        )

    def get_configs_path(self, safe_benchmark_name: str) -> str:
        """YAML file with the configs to run."""
        return f"{self.base_path}{self.configs_base_path}{safe_benchmark_name}.yaml"

    def get_output_path(self, benchmark_name: str) -> str:
        """Output directory for the benchmark."""
        return self.base_path + self.output_dir_base_from_base_path + benchmark_name

    def get_slurm_log_output_path(self, benchmark_name: str) -> str:
        """Directory for the SLURM output logs."""
        return self.base_path + self.slurm_log_output_from_base_path + benchmark_name

    def get_slurm_script_path(self, script_name: str) -> str:
        """Path to the SLURM script to run."""
        return (
            self.base_path
            + f"code/{self.tabarena_repo_name}/tabarena/tabflow_slurm/{script_name}"
        )


@dataclass
class SlurmSetup:
    """Setup for SLURM jobs.  Adjust as needed for your cluster setup."""

    script_name: str = "submit_template.sh"
    """Name of the SLURM (array) script that to run on the cluster
    (only used to print the command to run)."""
    gpu_partition: str = "alldlc2_gpu-l40s"
    """SLURM partition to use for GPU jobs."""
    cpu_partition: str = "alldlc2_cpu-epyc9655"
    """SLURM partition to use for CPU jobs."""
    extra_gres: str | None = "localtmp:100"
    """Extra SLURM gres to use for the jobs."""
    mem_per_handle: bool = False
    """How to pass memory constraints to SLURM jobs.
        - If True, we set SLURM memory per CPUs (or GPUs if available).
        - If False, we set SLURM memory per job.
    """
    exclusive_node: bool = False
    """If True, we assume we can use all resources on the node."""

    time_limit_overhead: int = 1
    """Overhead time in hours to add to the SLURM time limit to account for
    job scheduling and other non-model fitting overhead."""

    configs_per_job: int = 5
    """Batching of several experiments per job to reduce the number of  SLURM jobs."""

    max_array_size: int = 29_999
    """Maximum number of array tasks per SLURM array job.
    If the total number of jobs exceeds this limit, the jobs are split
    into multiple array jobs, each with its own JSON file and sbatch command."""

    setup_ray_for_slurm_shared_resources_environment: bool = True
    """Prepare Ray for a SLURM shared resource environment.
    Recommended to set to True if sequential_local_fold_fitting is False."""


@dataclass
class BenchmarkSetup2026:
    """Manually set the parameters for the benchmark run."""

    benchmark_name: str
    """Unique name of the benchmark; determines where output artifacts are stored."""

    task_metadata: Literal["tabarena-v0.1"] | pd.DataFrame | list[TabArenaTaskMetadata] | str | Path
    """Metadat that defines the tasks to benchmark.

    If str, we assume it is the path to a CSV file, which we load as DataFrame.

    This is either a pandas DataFrame or a TabArenaTaskMetadata object.
    We assume the DataFrame is created from a TabArenaTaskMetadata (or has all columns
    needed to parse each row via TabArenaTaskMetadata.from_row).
    """
    problem_types_to_run: list[str] = field(
        default_factory=lambda: [
            "binary",
            "multiclass",
            "regression",
        ]
    )
    """Problem types to run in the benchmark. Adjust as needed to run only
     specific problem types.
     Options: "binary", "regression", "multiclass".

    Can be understood as a filter on top of the TabArenaTaskMetadata.
    """
    split_indices_to_run: list[str] | Literal["lite"] | None = None
    """Split indices to run in the benchmark. Adjust as needed to run only specific
    splits. If None, we run all splits. If "lite", we run only the first split."""

    path_setup: PathSetup = field(default_factory=PathSetup)
    """Contains all path related to the benchmark."""
    slurm_setup: SlurmSetup = field(default_factory=SlurmSetup)
    """SLURM config information for the benchmark."""

    time_limit: int = 3600
    """Time limit for each fit of a model in seconds -- including time for validation.
    By default, 3600 seconds is used."""
    time_limit_for_model_agnostic_preprocessing: int | None = None
    """The time limit for the model agnostic preprocessing step."""
    time_limit_with_model_agnostic_preprocessing: bool = False
    """Whether the model agnostic preprocessing should influence the
    fit time of the model:
        - If False (default), we stop fitting a model after `time_limit`.
        - If True, we stop fitting a model after `time_limit` minus the
            time it took for model agnostic preprocessing.
    """
    num_cpus: int | None = 8
    """Number of CPUs to use for the job.
    If None, use all available CPUs."""
    num_gpus: int = 0
    """Number of GPUs to use for the jobs."""
    memory_limit: int | None = 32
    """Memory/RAM limit for the jobs in GB.
    If None, use all available memory."""
    n_random_configs: int = 50
    """Number of random hyperparameter configurations to run for each model"""
    models: list[tuple[str, int | str | dict]] = field(default_factory=list)
    """List of models to run in the benchmark with metadata.
    Metadata keys from left to right:
        - model name: str
        - number of random hyperparameter configurations to run: int or str
            Some special cases are:
                - If 0, only the default configuration is run.
                - If "all", `n_random_configs`-many configurations are run.
                - If dict, kwargs for AGExperiment
    Example usage:
        # Run all random configs for LightGBM, 10 random configs for Random Forest,
        and only the default for TabDPT.
        default_factory=lambda: [
                ("LightGBM", "all"),
                ("RandomForest", 10),
                ("TabDPT", 0),
            ]
        )

    Models one can use: "CatBoost", "EBM", "ExtraTrees", "FastaiMLP", "KNN", "LightGBM",
    "Linear", "ModernNCA", "TorchMLP", "RandomForest", "RealMLP", "TabDPT", "TabICL",
    "TabM", "TabPFNv2", "XGBoost", "Mitra", "xRFM", "RealTabPFN-v2.5", "SAP-RPT-OSS",
    "TabICLv2", "PerpetualBooster", "TabSTAR"

    For the newest set of available models, see:
    `tabarena.models.utils.get_configs_generator_from_name`
    """
    model_agnostic_preprocessing: bool = True
    """Whether to use model-agnostic preprocessing or not.
    By default, we use AutoGluon's automatic preprocessing for all models.
    This can be disabled by setting this to False. Warning: the model then needs
    to be able to handle this!
    """
    preprocessing_pipelines: list[str] = field(
        default_factory=lambda: ["tabarena_default"]
    )
    """EXPERIMENTAL!
    Preprocessing pipelines to add to the configurations we want to run.

    Each options multiplies the number of configurations to run by the number of
    pipelines. For example, if we have 10 configurations and 2 pipelines, we will
    run 20 configurations.

    Options:
        - "default": Use the default preprocessing pipeline.
        - "tabarena_default": new model agnostic and model specific preprocessing
            updates for TabArena (experimental, can be buggy!).
        - Any other string points to custom experimental code for now.
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

    # Misc Settings
    # -------------
    dynamic_tabarena_validation_protocol: bool = True
    """If True, the validation data will be adapted dynamically based on the task.
    WARNING: this can overwrite the configured validation of a configuration!"""
    shuffle_features: bool = True
    """Whether to shuffle the features of the datasets. Only here for backward compatibility
    with the original TabArena setup, but not recommended to change."""
    adapt_num_folds_to_n_classes: bool = True
    """Whether to adapt the number of folds to the number of classes for classification tasks.
    Ensures that each fold has at least one sample of each class.
    """
    parallel_benchmark_name: str | None = None
    """Set this is to some string value to make sure you can run parallel
    jobs for the same benchmark name.This ensures that the config and job .yaml/.json
    files are not overwritten, while SLURM output and TabArena output are still saved
    in the same folders.
    """
    ignore_cache: bool = False
    """If True, will overwrite the cache and run all jobs again."""
    num_ray_cpus = 8
    """Number of CPUs to use for checking the cache and generating the jobs.
    This should be set to the number of CPUs available to the python script."""
    sequential_local_fold_fitting: bool = False
    """Use Ray for local fold fitting. This is used to speed up the local fold fitting
    and force this behavior if True. If False the default strategy of running the
    local fold fitting is used, as determined by AutoGluon and the model's
    default_ag_args_ensemble parameters."""
    model_artifacts_base_path: str | Path | None = "/tmp/ag"  # noqa: S108
    """Adapt the default temporary directory used for model artifacts in TabArena.
        - If None, the default temporary directory is used: "./AutoGluonModels".
        - If a string or Path, the directory is used as the base path for the temporary
        and any model artifacts will be stored in time-stamped subdirectories.
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
    verbosity: int = 2
    """Verbosity level for logging and printing."""

    def __post_init__(self):
        # Max number of configs per job. Might be overridden.
        # Determines total time for a job.
        self._max_configs_per_job = self.slurm_setup.configs_per_job

    @property
    def time_limit_per_config(self):
        """Compute the maximal time limit per config plus some overhead."""
        time_limit_per_config = self.time_limit
        if self.time_limit_for_model_agnostic_preprocessing is not None:
            time_limit_per_config += self.time_limit_for_model_agnostic_preprocessing
        # Add a constant SLURM overhead.
        if self.time_limit_with_model_agnostic_preprocessing:
            time_limit_per_config += 60 * 15
        return time_limit_per_config

    @property
    def _parallel_safe_benchmark_name(self) -> str:
        """Safe benchmark name for file paths."""
        benchmark_name = self.benchmark_name
        if self.parallel_benchmark_name is not None:
            benchmark_name += f"_{self.parallel_benchmark_name}"
        return benchmark_name

    @staticmethod
    def _get_slurm_base_command(  # noqa: PLR0913
        *,
        num_gpus: int,
        num_cpus: int,
        time_limit_per_config: int,
        configs_per_job: int,
        time_limit_overhead: int,
        gpu_partition: str,
        cpu_partition: str,
        slurm_log_output: str,
        slurm_script_path: str,
        slurm_extra_gres: str,
        slurm_exclusive_node: bool,
        memory_limit: int,
        slurm_mem_per_handle: bool,
    ):
        """SLURM command to run the benchmark.

        We set the following parameters based on the benchmark setup:
            - slurm script
            - partition
            - gres (including GPUs)
            - time
            - cpus
            - memory

        Parameter
        --------
        num_gpus: int
            Number of GPUs to use for the jobs.
        num_cpus: int
            Number of CPUs to use for the job.
        time_limit_per_config: int
            Time limit for each fit of a config.

        """
        is_gpu_job = num_gpus > 0

        partition = gpu_partition if is_gpu_job else cpu_partition
        partition = "--partition=" + partition
        slurm_logs = f"--output={slurm_log_output}/%A/slurm-%A_%a.out"

        time_in_h = (
            time_limit_per_config // 3600 * configs_per_job + time_limit_overhead
        )
        time_in_h = f"--time={time_in_h}:00:00"

        # Handle GPU (same for exclusive and non-exclusive)
        gres = f"gpu:{num_gpus}" if is_gpu_job else ""
        if slurm_extra_gres:
            if len(gres) > 0:
                gres += ","
            gres += slurm_extra_gres
        gres = f"--gres={gres}" if len(gres) > 0 else None
        cmd_arg = f"{partition}"
        if gres is not None:
            cmd_arg += f" {gres}"

        if slurm_exclusive_node:
            return f"{cmd_arg} {time_in_h} --mem=0 --nodes=1 --exclusive {slurm_logs} {slurm_script_path}"

        # Handle CPU
        cpus = f"--cpus-per-task={num_cpus}"
        # Handle Memory
        if slurm_mem_per_handle:
            if is_gpu_job:
                mem = f"--mem-per-gpu={memory_limit // num_gpus}G"
            else:
                mem = f"--mem-per-cpu={memory_limit // num_cpus}G"
        else:
            mem = f"--mem={memory_limit}G"
        if memory_limit is None:
            mem = mem[:-1]

        return f"{cmd_arg} {time_in_h} {cpus} {mem} {slurm_logs} {slurm_script_path}"

    @property
    def slurm_base_command(self):
        """SLURM command to run the benchmark."""
        p_bm = self._parallel_safe_benchmark_name
        slurm_script_path = self.path_setup.get_slurm_script_path(
            self.slurm_setup.script_name
        )

        return self._get_slurm_base_command(
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
            memory_limit=self.memory_limit,
            time_limit_per_config=self.time_limit_per_config,
            configs_per_job=self._max_configs_per_job,
            time_limit_overhead=self.slurm_setup.time_limit_overhead,
            slurm_log_output=self.path_setup.get_slurm_log_output_path(p_bm),
            slurm_script_path=slurm_script_path,
            slurm_extra_gres=self.slurm_setup.extra_gres,
            slurm_exclusive_node=self.slurm_setup.exclusive_node,
            slurm_mem_per_handle=self.slurm_setup.mem_per_handle,
            gpu_partition=self.slurm_setup.gpu_partition,
            cpu_partition=self.slurm_setup.cpu_partition,
        )

    def _load_task_metadata(self) -> list[TabArenaTaskMetadata]:
        """Function to parse and filter the task metadata for jobs we want to run."""
        # Parse task metadata
        task_metadata = self.task_metadata

        if isinstance(task_metadata, str) and (task_metadata == "tabarena-v0.1"):
            print("Loading task metadata from TabArena v0.1 and converting to new TabArenaTaskMetadata fromat...")
            from tabarena.nips2025_utils.fetch_metadata import (
                load_curated_task_metadata,
            )

            metadata = load_curated_task_metadata()
            task_metadata = []
            for row in metadata.itertuples():
                num_classes = row.num_classes
                num_instances = row.num_instances
                num_features = row.num_features

                n_repeats = row.tabarena_num_repeats
                n_folds = row.num_folds

                metric_map = {
                    "binary": "roc_auc",
                    "multiclass": "log_loss",
                    "regression": "rmse",
                }
                eval_metric = metric_map[row.problem_type]

                for repeat_i in range(n_repeats):
                    for fold_i in range(n_folds):

                        split_index = SplitMetadata.get_split_index(repeat_i=repeat_i, fold_i=fold_i)
                        splits_metadata = {
                            split_index:
                            SplitMetadata(
                                repeat=repeat_i,
                                fold=fold_i,
                                num_instances_train=num_instances * 2/3,
                                num_instances_test=num_instances * 1/3,
                                num_instance_groups_train=num_instances * 2/3,
                                num_instance_groups_test=num_instances * 1/3,
                                num_classes_train=num_classes,
                                num_classes_test=num_classes,
                                num_features_train=num_features,
                                num_features_test=num_features,
                            )
                        }

                        task_metadata.append(
                            TabArenaTaskMetadata(
                                task_id_str=row.task_id,
                                dataset_name=row.dataset_name,
                                tabarena_task_name=row.dataset_name,
                                problem_type=row.problem_type,
                                is_classification=row.is_classification,
                                target_name=row.target_feature,
                                stratify_on=row.target_feature if row.is_classification else None,
                                split_time_horizon=None,
                                split_time_horizon_unit=None,
                                time_on=None,
                                group_on=None,
                                group_time_on=None,
                                group_labels=None,
                                multiclass_max_n_classes_over_splits=num_classes,
                                multiclass_min_n_classes_over_splits=num_classes,
                                class_consistency_over_splits=True,
                                num_instances=num_instances,
                                num_features=num_features,
                                num_instance_groups=num_instances,
                                num_classes=num_classes,
                                splits_metadata=splits_metadata,
                                eval_metric=eval_metric,
                            )
                        )
        if isinstance(task_metadata, (str, Path)):
            print(f"Loading task metadata from {task_metadata}...")
            task_metadata = pd.read_csv(task_metadata, index_col=False)
        if isinstance(task_metadata, pd.DataFrame):
            # Parse task_metadat
            task_metadata = [
                TabArenaTaskMetadata.from_row(row)
                for _, row in task_metadata.iterrows()
            ]
        assert all(isinstance(x, TabArenaTaskMetadata) for x in task_metadata)
        n_rolled_up_tasks = len(task_metadata)

        # Unify format to be unrolled
        task_metadata = [
            single_ttm for ttm in task_metadata for single_ttm in ttm.unroll_splits()
        ]
        n_unrolled_tasks = len(task_metadata)

        # -- Perform general filters/slices
        task_metadata = [
            ttm
            for ttm in task_metadata
            if ttm.problem_type in self.problem_types_to_run
        ]
        n_problem_types_filtered_tasks = len(task_metadata)

        if self.split_indices_to_run is not None:
            if self.split_indices_to_run == "lite":
                split_indices_to_run = [
                    SplitMetadata.get_split_index(repeat_i=0, fold_i=0)
                ]
            else:
                split_indices_to_run = self.split_indices_to_run

            # Assert split indices are valid
            split_index_pattern = re.compile(r"^r\d+f\d+$")
            for split_index in split_indices_to_run:
                assert (
                    split_index_pattern.match(split_index)
                ), f"Invalid SplitIndex format: {split_index!r}, expected 'r{{int}}f{{int}}'"

            task_metadata = [
                ttm for ttm in task_metadata if ttm.split_index in split_indices_to_run
            ]
        n_splits_filtered_tasks = len(task_metadata)

        # -- Sanity checks
        for ttm in task_metadata:
            if ttm.task_id_str is None:
                raise ValueError(
                    f"Task metadata for task {ttm.tabarena_task_name} does not have a task_id_str!"
                )

        print(
            f"Found {len(task_metadata)} tasks from metadata."
            f"\n\tTask Filter History:"
            f"\n\t(1) {n_rolled_up_tasks} datasets -> {n_unrolled_tasks} Tasks."
            f"\n\t(2) Filter to problem types: {n_problem_types_filtered_tasks}"
            f"\n\t(3) Filter to splits: {n_splits_filtered_tasks}."
        )
        return task_metadata

    def get_jobs_to_run(self):  # noqa: C901
        """Determine all jobs to run by checking the cache and filtering
        invalid jobs.
        """
        if self.path_setup.openml_cache_path != "auto":
            Path(self.path_setup.openml_cache_path).mkdir(parents=True, exist_ok=True)
        Path(self.path_setup.get_output_path(self.benchmark_name)).mkdir(
            parents=True, exist_ok=True
        )
        Path(self.path_setup.get_slurm_log_output_path(self.benchmark_name)).mkdir(
            parents=True, exist_ok=True
        )

        task_metadata_list = self._load_task_metadata()
        configs = self.generate_configs_yaml()

        def yield_all_jobs():
            for ta_task_metadata in task_metadata_list:
                task_id = ta_task_metadata.task_id_str
                split_md = ta_task_metadata.splits_metadata[
                    ta_task_metadata.split_index
                ]

                for config_index, config in list(enumerate(configs)):
                    yield {
                        "config_index": config_index,
                        "config": config,
                        "task_id": task_id,
                        "fold_i": split_md.fold,
                        "repeat_i": split_md.repeat,
                        "n_samples_train_per_fold": split_md.num_instances_train,
                        "n_features": split_md.num_features_train,
                        "n_classes": split_md.num_classes_train,
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
                "output_dir": self.path_setup.get_output_path(self.benchmark_name),
                "models_to_constraints": self.models_to_constraints,
                "ignore_cache": self.ignore_cache,
            },
            track_progress=True,
            tqdm_kwargs={"desc": "Checking Cache and Filter Invalid Jobs"},
        )
        output = [
            item for sublist in output for item in sublist
        ]  # Flatten the batched list
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
        max_config_batch = 1
        for job_key, config_indices in to_run_job_map.items():
            to_run_jobs += len(config_indices)
            for config_batch in to_batch_list(
                config_indices, self.slurm_setup.configs_per_job
            ):
                max_config_batch = max(max_config_batch, len(config_batch))
                jobs.append(
                    {
                        "task_id": job_key[0],
                        "fold": job_key[1],
                        "repeat": job_key[2],
                        "config_index": config_batch,
                    },
                )
        self._max_configs_per_job = max_config_batch
        print(f"Generated {to_run_jobs} jobs to run without batching.")
        print(f"Jobs with batching: {len(jobs)}")
        return jobs

    def _generate_autogluon_config(
        self, *, model_name: str, agexp_kwargs: dict, pipeline_method_kwargs: dict
    ) -> list:
        """Parse the AutoGluon config from the models."""
        from tabarena.benchmark.experiment.experiment_constructor import (
            AGExperiment,
        )

        for key in ["fit_kwargs", "init_kwargs"]:
            if key not in agexp_kwargs:
                agexp_kwargs[key] = {}
            if key in pipeline_method_kwargs:
                agexp_kwargs[key].update(pipeline_method_kwargs[key])
        agexp_kwargs["fit_kwargs"]["time_limit"] = self.time_limit

        return [
            AGExperiment(
                name=model_name,
                **agexp_kwargs,
            )
        ]

    def _generate_model_configs(
        self,
        *,
        model_name: str,
        n_configs: int | str,
        pipeline_method_kwargs: dict,
        name_id_suffix: str,
        default_seed_config: str = "fold-config-wise",
    ) -> list:

        from tabarena.models.utils import get_configs_generator_from_name

        if isinstance(n_configs, str) and n_configs == "all":
            n_configs = self.n_random_configs
        elif not isinstance(n_configs, int):
            raise ValueError(
                f"Invalid number of configurations for model {model_name}: {n_configs}. "
                "Must be an integer or 'all'."
            )
        config_generator = get_configs_generator_from_name(model_name)
        # TODO: add model agnostic time limit here
        return config_generator.generate_all_bag_experiments(
            num_random_configs=n_configs,
            add_seed=default_seed_config,
            name_id_suffix=name_id_suffix,
            method_kwargs=pipeline_method_kwargs,
            time_limit=self.time_limit,
            time_limit_with_preprocessing=self.time_limit_with_model_agnostic_preprocessing,
        )

    def generate_configs_yaml(self) -> list[dict]:
        """Generate the YAML file with the configurations to run based
        on specific models to run.
        """
        from tabarena.benchmark.experiment import (
            YamlExperimentSerializer,
        )

        experiments_lst = []
        method_kwargs = {
            "init_kwargs": {"verbosity": self.verbosity},
            "shuffle_features": self.shuffle_features,
            "fit_kwargs": dict(),
        }
        if self.model_artifacts_base_path is not None:
            method_kwargs["init_kwargs"]["default_base_path"] = (
                self.model_artifacts_base_path
            )
        if not self.model_agnostic_preprocessing:
            method_kwargs["fit_kwargs"]["feature_generator"] = None
        if self.adapt_num_folds_to_n_classes:
            method_kwargs["fit_kwargs"]["adapt_num_folds_to_n_classes"] = True

        print(
            "Generating experiments for models...",
            f"\n\t`all` := number of configs: {self.n_random_configs}",
            f"\n\t{len(self.models)} models: {self.models}",
            f"\n\t{len(self.preprocessing_pipelines)} preprocessing pipelines: "
            f"{self.preprocessing_pipelines}",
            f"\n\tMethod kwargs: {method_kwargs}",
        )
        for preprocessing_name in self.preprocessing_pipelines:
            pipeline_method_kwargs = deepcopy(method_kwargs)

            name_id_suffix = ""
            if preprocessing_name != "default":
                pipeline_method_kwargs["preprocessing_pipeline"] = preprocessing_name
            if preprocessing_name != "tabarena_default":
                name_id_suffix = f"_{preprocessing_name}"

            for model_name, n_configs_or_kwargs in self.models:
                # Resolve AutoGluon Config
                if model_name.startswith("AutoGluon"):
                    experiments_lst.append(
                        self._generate_autogluon_config(
                            model_name=model_name,
                            agexp_kwargs=n_configs_or_kwargs,
                            pipeline_method_kwargs=pipeline_method_kwargs,
                        )
                    )
                    continue

                # Resolve model configs
                experiments_lst.append(
                    self._generate_model_configs(
                        model_name=model_name,
                        n_configs=n_configs_or_kwargs,
                        pipeline_method_kwargs=pipeline_method_kwargs,
                        name_id_suffix=name_id_suffix,
                    )
                )

        # Verify no duplicate names
        experiments_all = [
            exp for exp_family_lst in experiments_lst for exp in exp_family_lst
        ]
        experiment_names = set()
        for experiment in experiments_all:
            if experiment.name not in experiment_names:
                experiment_names.add(experiment.name)
            else:
                raise AssertionError(
                    f"Found multiple instances of experiment named {experiment.name}. "
                    f"All experiment names must be unique!",
                )

        configs_path = self.path_setup.get_configs_path(
            self._parallel_safe_benchmark_name
        )
        YamlExperimentSerializer.to_yaml(
            experiments=experiments_all,
            path=configs_path,
        )

        # Read YAML file and get the number of configs
        with open(configs_path) as file:
            return yaml.safe_load(file)["methods"]

    def get_jobs_dict(self):
        """Get the jobs to run as a dictionary with default arguments and jobs."""
        jobs = list(self.get_jobs_to_run())

        # Fake memory limit for estimates if needed
        memory_limit = self.memory_limit
        if self.fake_memory_for_estimates is not None:
            memory_limit = self.fake_memory_for_estimates

        default_args = {
            "python": self.path_setup.python_path,
            "run_script": self.path_setup.run_script_path,
            "openml_cache_dir": self.path_setup.openml_cache_path,
            "configs_yaml_file": self.path_setup.get_configs_path(
                self._parallel_safe_benchmark_name
            ),
            "output_dir": self.path_setup.get_output_path(self.benchmark_name),
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
            "memory_limit": memory_limit,
            "setup_ray_for_slurm_shared_resources_environment": self.slurm_setup.setup_ray_for_slurm_shared_resources_environment,
            "ignore_cache": self.ignore_cache,
            "sequential_local_fold_fitting": self.sequential_local_fold_fitting,
            "dynamic_tabarena_validation_protocol": self.dynamic_tabarena_validation_protocol,
        }
        return {"defaults": default_args, "jobs": jobs}

    def setup_jobs(self, array_job_limit: int = 100) -> str | list[str]:
        """Setup the jobs to run by generating the SLURM job JSON file(s).

        If the number of jobs exceeds `slurm_setup.max_array_size`, the jobs
        are split into multiple array jobs, each with its own JSON file.

        Returns a single command string if one batch, or a list of command
        strings if multiple batches are needed.
        """
        jobs_dict = self.get_jobs_dict()
        base_json_path = self.path_setup.get_slurm_job_json_path(
            self._parallel_safe_benchmark_name
        )
        all_jobs = jobs_dict["jobs"]
        n_jobs = len(all_jobs)
        if n_jobs == 0:
            print("No jobs to run.")
            Path(base_json_path).unlink(missing_ok=True)
            Path(
                self.path_setup.get_configs_path(self._parallel_safe_benchmark_name)
            ).unlink(missing_ok=True)
            return "N/A"

        max_array_size = self.slurm_setup.max_array_size
        job_batches = list(to_batch_list(all_jobs, max_array_size))

        run_commands = []
        for batch_idx, batch_jobs in enumerate(job_batches):
            batch_jobs = list(batch_jobs)
            # Use original path for single batch, append _batch{i} for multiple
            if len(job_batches) == 1:
                json_path = base_json_path
            else:
                json_path = base_json_path.replace(
                    ".json", f"_batch{batch_idx}.json"
                )

            batch_dict = {"defaults": jobs_dict["defaults"], "jobs": batch_jobs}
            with open(json_path, "w") as f:
                json.dump(batch_dict, f)

            batch_size = len(batch_jobs)
            run_command = (
                f"sbatch --array=0-{batch_size - 1}%{array_job_limit}"
                f" {self.slurm_base_command} {json_path}"
            )
            run_commands.append(run_command)

        batch_info = ""
        if len(job_batches) > 1:
            batch_info = (
                f"\nSplit into {len(job_batches)} array job batches"
                f" (max {max_array_size} per batch)."
            )
        print(
            f"##### Setup Jobs for {self._parallel_safe_benchmark_name}"
            f"{batch_info}"
            "\nRun the following command(s) to start the jobs:"
            f"\n" + "\n".join(run_commands) + "\n"
        )

        if len(run_commands) == 1:
            return run_commands[0]
        return run_commands

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

        max_n_samples_train_per_fold = model_constraints.get(
            "max_n_samples_train_per_fold", None
        )
        if (max_n_samples_train_per_fold is not None) and (
            n_samples_train_per_fold > max_n_samples_train_per_fold
        ):
            return False

        min_n_samples_train_per_fold = model_constraints.get(
            "min_n_samples_train_per_fold", None
        )
        if (min_n_samples_train_per_fold is not None) and (
            n_samples_train_per_fold < min_n_samples_train_per_fold
        ):
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
    models_to_constraints: dict,
    ignore_cache: bool,
) -> bool:
    """Check if a job should be run based on the configuration and cache.
    Must be not a class function to be used with Ray.
    """
    config = input_data["config"]
    task_id = input_data["task_id"]
    fold_i = input_data["fold_i"]
    repeat_i = input_data["repeat_i"]

    # Check if local task or not
    try:
        task_id = int(task_id)
    except ValueError:
        # Extract the local task ID if it is a UserTask.task_id_str
        task_id = task_id.split("|", 2)[1]

    # Filter out-of-constraints datasets
    if "model_cls" in config:
        model_cls = config["model_cls"]
    else:
        assert config["name"].startswith("AutoGluon")
        model_cls = "AutoGluon"
    if not BenchmarkSetup2026.are_model_constraints_valid(
        model_cls=model_cls,
        n_features=input_data["n_features"],
        n_classes=input_data["n_classes"],
        n_samples_train_per_fold=input_data["n_samples_train_per_fold"],
        models_to_constraints=models_to_constraints,
    ):
        return False

    if ignore_cache:
        return True

    return not check_cache_hit(
        result_dir=output_dir,
        method_name=config["name"],
        task_id=task_id,
        fold=fold_i,
        repeat=repeat_i,
        cache_path_format="name_first",
        cache_cls=CacheFunctionPickle,
        cache_cls_kwargs={"include_self_in_call": True},
        mode="local",
    )
