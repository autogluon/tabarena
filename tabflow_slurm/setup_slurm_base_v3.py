from __future__ import annotations

import json
import os
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import pandas as pd
import ray
import yaml
from tabarena.benchmark.experiment.experiment_constructor import Experiment, resolve_class
from tabarena.benchmark.experiment.experiment_utils import check_cache_hit
from tabarena.benchmark.models.model_registry import infer_model_cls
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
        return self.base_path + (f"code/{self.tabarena_repo_name}/tabarena/tabflow_slurm/run_tabarena_experiment.py")

    @property
    def configs_base_path(self) -> str:
        """YAML file with the configs to run.

        File path is f"{self.base_path}{self.configs_path_from_base_path}
        {self._safe_benchmark_name}.yaml".
        """
        return f"code/{self.tabarena_repo_name}/tabarena/tabflow_slurm/benchmark_configs_"

    def get_slurm_job_json_path(self, safe_benchmark_name: str) -> str:
        """JSON file with the job data to run used by SLURM.
        This is generated from the configs and metadata.
        """
        # TODO: change UX for config and slurm paths.
        return f"{self.base_path}{Path(self.configs_base_path).parent}/slurm_run_data_{safe_benchmark_name}.json"

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
        return self.base_path + f"code/{self.tabarena_repo_name}/tabarena/tabflow_slurm/{script_name}"


@dataclass
class SchedulerSetup:
    """Base class for scheduler-specific job setup.

    Owns the generic job-batching knobs (`bundle_size`,
    `bundle_size_per_dataset`) shared by any array-style scheduler. Subclasses
    implement how to construct the run commands a user invokes and contribute
    any scheduler-specific entries to the per-job defaults via
    `get_extra_default_args`.
    """

    bundle_size: int = 5
    """Number of `(task_id, fold, repeat, config_index)` items batched into a
    single array task. Items may span different `(task_id, fold, repeat)`
    groups.

    Mirrors the design used by `tabflow` (the SageMaker side) — see
    `tabflow/cli/launch_jobs.py::get_tasks_batched`."""

    bundle_size_per_dataset: dict[str, int] | None = None
    """Optional per-dataset override of `bundle_size`, keyed by `dataset_name`.
    Items from datasets not listed here use `bundle_size`. Items belonging to
    different effective bundle sizes are not mixed in the same bundle."""

    def get_run_commands(
        self,
        *,
        jobs_dict: dict,
        path_setup: PathSetup,
        benchmark_name: str,
        parallel_safe_benchmark_name: str,
        resources_setup: ResourcesSetup,
    ) -> list[str] | None:
        """Persist `jobs_dict` and return the run commands a user should invoke.

        Returns `None` when there is no work to launch (so the caller can
        clean up any stale per-run artifacts).

        Schedulers receive `path_setup` and the benchmark name(s) so they can
        derive any scheduler-specific paths (log dirs, scripts, job-definition
        files, etc.) themselves without leaking those concerns into the
        interface. `parallel_safe_benchmark_name` is used to identify per-run
        artifacts that must not collide across parallel runs; `benchmark_name`
        is the bare name used for resources that should be shared across
        parallel runs (e.g. log directories).

        Any scheduler-specific batching metadata (e.g. the max number of
        configs assigned to a single array task) is expected to live inside
        `jobs_dict` alongside `jobs` and `defaults`.
        """
        raise NotImplementedError

    def get_extra_default_args(self) -> dict:
        """Extra key/value pairs to merge into the per-job `defaults` dict.

        Subclasses override this to surface scheduler-specific runtime flags
        (e.g. SLURM Ray-shared-resources hints) without coupling the caller
        to a specific scheduler. Default: no extras.
        """
        return {}

    def bundle_items(self, approved: list[JobCandidate]) -> tuple[list[dict], int]:
        """Group approved candidates into array-task bundles.

        Candidates are partitioned by their effective bundle size
        (`bundle_size_per_dataset[dataset_name]` if present, else
        `bundle_size`) so candidates with different effective sizes never
        share a task. Returns `(jobs, max_configs_per_job)` where each job is
        `{"items": [...]}` (JSON-serializable, only the identifying-tuple
        fields), and `max_configs_per_job` is the largest bundle observed
        (used by schedulers to budget the per-task time limit).
        """
        overrides = self.bundle_size_per_dataset or {}
        by_size: dict[int, list[JobCandidate]] = {}
        for c in approved:
            size = overrides.get(c.dataset_name, self.bundle_size)
            by_size.setdefault(size, []).append(c)

        jobs: list[dict] = []
        max_configs_per_job = 1
        # Sort sizes for deterministic job ordering across runs.
        for size in sorted(by_size):
            for bundle in to_batch_list(by_size[size], size):
                max_configs_per_job = max(max_configs_per_job, len(bundle))
                jobs.append(
                    {
                        "items": [
                            {
                                "task_id": c.task_id,
                                "fold": c.fold,
                                "repeat": c.repeat,
                                "config_index": c.config_index,
                            }
                            for c in bundle
                        ],
                    },
                )
        return jobs, max_configs_per_job


@dataclass
class SlurmSetup(SchedulerSetup):
    """Setup for SLURM jobs. Adjust as needed for your cluster setup."""

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

    max_array_size: int = 29_999
    """Maximum number of array tasks per SLURM array job.
    If the total number of jobs exceeds this limit, the jobs are split
    into multiple array jobs, each with its own JSON file and sbatch command."""

    array_job_limit: int = 100
    """Maximum number of concurrent SLURM array tasks per array job.
    Passed as the `%N` suffix to `sbatch --array=0-K%N`."""

    setup_ray_for_slurm_shared_resources_environment: bool = True
    """Prepare Ray for a SLURM shared resource environment.
    Recommended to set to True if sequential_local_fold_fitting is False."""

    def get_extra_default_args(self) -> dict:
        """Surface the SLURM-specific Ray-shared-resources hint to the per-job defaults."""
        return {
            "setup_ray_for_slurm_shared_resources_environment": self.setup_ray_for_slurm_shared_resources_environment,
        }

    def get_run_commands(
        self,
        *,
        jobs_dict: dict,
        path_setup: PathSetup,
        benchmark_name: str,
        parallel_safe_benchmark_name: str,
        resources_setup: ResourcesSetup,
    ) -> list[str] | None:
        """Persist `jobs_dict` to one or more JSON files and return matching sbatch commands.

        If `jobs_dict["jobs"]` exceeds `max_array_size`, the jobs are split across
        multiple array-job batches; each batch gets its own `_batch{i}.json`
        file and its own sbatch command (the single-batch case keeps the base
        path). All SLURM-specific paths (job JSON, log dir, script) are
        derived from `path_setup` + the two benchmark name flavors and
        `self.script_name`.

        The time-budget computation needs to know the worst-case number of
        configs handled by a single array task; this is read from
        `jobs_dict["max_configs_per_job"]` (populated by the caller).

        Returns the list of sbatch commands to run, or `None` if there are
        no jobs (in which case the base JSON file is also removed if it exists).
        """
        base_json_path = path_setup.get_slurm_job_json_path(parallel_safe_benchmark_name)

        all_jobs = jobs_dict["jobs"]
        if not all_jobs:
            print("No jobs to run.")
            Path(base_json_path).unlink(missing_ok=True)
            return None

        base_command = self._build_sbatch_prefix(
            resources_setup=resources_setup,
            configs_per_job=jobs_dict["max_configs_per_job"],
            slurm_log_output=path_setup.get_slurm_log_output_path(benchmark_name),
            slurm_script_path=path_setup.get_slurm_script_path(self.script_name),
        )

        run_commands = self._write_job_batches_and_build_commands(
            all_jobs=all_jobs,
            defaults=jobs_dict["defaults"],
            base_json_path=base_json_path,
            base_command=base_command,
        )

        # --- Print summary ---
        n_batches = len(run_commands)
        batch_info = (
            f"\nSplit into {n_batches} array job batches (max {self.max_array_size} per batch)."
            if n_batches > 1
            else ""
        )
        print(
            "##### Setup Jobs"
            f"{batch_info}"
            "\nRun the following command(s) to start the jobs:"
            "\n" + "\n".join(run_commands) + "\n"
        )
        return run_commands

    def _write_job_batches_and_build_commands(
        self,
        *,
        all_jobs: list,
        defaults: dict,
        base_json_path: str,
        base_command: str,
    ) -> list[str]:
        """Split `all_jobs` into `max_array_size`-sized batches, persist each as JSON,
        and return the matching `sbatch --array=...` commands.

        With one batch the JSON is written to `base_json_path`; with multiple
        batches the path is suffixed with `_batch{i}` so each batch has its own
        file.
        """
        job_batches = [list(b) for b in to_batch_list(all_jobs, self.max_array_size)]
        multi_batch = len(job_batches) > 1

        run_commands: list[str] = []
        for batch_idx, batch_jobs in enumerate(job_batches):
            json_path = (
                base_json_path.replace(".json", f"_batch{batch_idx}.json")
                if multi_batch
                else base_json_path
            )
            with open(json_path, "w") as f:
                json.dump({"defaults": defaults, "jobs": batch_jobs}, f)

            run_commands.append(
                f"sbatch --array=0-{len(batch_jobs) - 1}%{self.array_job_limit} {base_command} {json_path}"
            )
        return run_commands

    def _build_sbatch_prefix(
        self,
        *,
        resources_setup: ResourcesSetup,
        configs_per_job: int,
        slurm_log_output: str,
        slurm_script_path: str,
    ) -> str:
        """Build the space-separated sbatch flag string used as the prefix of every
        per-batch `sbatch --array=... <prefix> <json>` invocation.

        Resource amounts come from `resources_setup`; partitions, gres,
        exclusivity, time overhead and memory style come from this `SlurmSetup`.
        """
        is_gpu_job = resources_setup.num_gpus > 0
        partition = self.gpu_partition if is_gpu_job else self.cpu_partition

        time_in_h = (
            resources_setup.time_limit_per_config // 3600 * configs_per_job
            + self.time_limit_overhead
        )

        gres_items: list[str] = []
        if is_gpu_job:
            gres_items.append(f"gpu:{resources_setup.num_gpus}")
        if self.extra_gres:
            gres_items.append(self.extra_gres)

        flag_parts: list[str] = [f"--partition={partition}"]
        if gres_items:
            flag_parts.append(f"--gres={','.join(gres_items)}")
        flag_parts.append(f"--time={time_in_h}:00:00")

        if self.exclusive_node:
            flag_parts += ["--mem=0", "--nodes=1", "--exclusive"]
        else:
            flag_parts.append(f"--cpus-per-task={resources_setup.num_cpus}")
            memory_limit = resources_setup.memory_limit
            if memory_limit is not None:
                if self.mem_per_handle:
                    divisor = resources_setup.num_gpus if is_gpu_job else resources_setup.num_cpus
                    prefix = "--mem-per-gpu" if is_gpu_job else "--mem-per-cpu"
                    flag_parts.append(f"{prefix}={memory_limit // divisor}G")
                else:
                    flag_parts.append(f"--mem={memory_limit}G")

        flag_parts.append(f"--output={slurm_log_output}/%A/slurm-%A_%a.out")
        flag_parts.append(slurm_script_path)
        return " ".join(flag_parts)


@dataclass
class ResourcesSetup:
    """Compute and time-budget resources for the benchmark jobs."""

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
    """Number of GPUs to use for the jobs (SLURM allocation and Ray)."""
    num_gpus_model: int | None = None
    """Number of GPUs passed to a model for fitting.
    If None (default), uses the same value as ``num_gpus``.
    Set to 0 to reserve the GPU for preprocessing (e.g. sentence-transformer
    encoding) while fitting models on CPU only."""
    memory_limit: int | None = 32
    """Memory/RAM limit for the jobs in GB.
    If None, use all available memory."""
    fake_memory_for_estimates: int | None = None
    """Experimental parameter that is to be ignored!

    If not None, this value is reported to models in place of `memory_limit`
    so the model's internal memory estimates are compared against it instead
    of the actually available memory on the system. Values in GB as
    `memory_limit`.

    This can be useful if:
        - To test or overrule (bad) memory estimates.
        - For models that use CPU memory as a proxy for GPU memory (e.g. most
          TFMs), this can be used if the job has much more VRAM than CPU memory.
    """

    @property
    def time_limit_per_config(self) -> int:
        """Maximal time limit per config plus some overhead."""
        total = self.time_limit
        if self.time_limit_for_model_agnostic_preprocessing is not None:
            total += self.time_limit_for_model_agnostic_preprocessing
        if self.time_limit_with_model_agnostic_preprocessing:
            total += 60 * 15  # constant SLURM overhead
        return total

    @property
    def effective_memory_limit(self) -> int | None:
        """Memory limit reported to models (honors the `fake_memory_for_estimates` override)."""
        return self.fake_memory_for_estimates if self.fake_memory_for_estimates is not None else self.memory_limit


@dataclass
class TasksToRunSetup:
    """Defines which tasks (datasets x splits) to run in the benchmark.

    Encapsulates the source of task metadata and any filters applied on top of it.
    """

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
    required_dtypes_to_run: list[str] | None = None
    """Adjust as needed to run only datasets with at least one column of data types.
    Options: "numeric", "categorical", "text", "datetime".
    If None, we do not require any data types.
    """
    forbidden_dtypes_to_run: list[str] | None = None
    """Adjust as needed to run only datasets without any columns of data types.
    Options: "numeric", "categorical", "text", "datetime".
    If None, we do not forbid any data types.
    """
    n_train_samples_to_run: tuple[int | None, int | None] | None = None
    """Tuple of lower and upper limit for the number of training samples of datasets run in the benchmark.
    Adjust as needed to run only datasets with a certain number of training samples.
    If None, we run all datasets.
    Lower limit is exclusive, upper limit is inclusive. For example, (0, 1000) runs only datasets with less
    than 1000 training samples. If a tuple value is None, there is no limit in that direction.
    """
    dataset_names_to_run: list[str] | None = None
    """List of dataset names to run in the benchmark. Adjust as needed to run only specific datasets.
    If None, we run all datasets. Matches against `dataset_name` of the task metadata.
    """

    def load_task_metadata(self) -> list[TabArenaTaskMetadata]:
        """Parse and filter the task metadata for jobs we want to run."""
        task_metadata = self._parse_task_metadata()

        # Unify format to be unrolled (one entry per split).
        task_metadata = [single_ttm for ttm in task_metadata for single_ttm in ttm.unroll_splits()]

        filter_steps = [
            ("problem types", self._filter_by_problem_types),
            ("splits", self._filter_by_split_indices),
            ("dataset names", self._filter_by_dataset_names),
            ("dtypes", self._filter_by_dtypes),
            ("dataset size", self._filter_by_train_samples),
        ]

        filter_history: list[tuple[str, int]] = [("Starting", len(task_metadata))]
        for label, filter_fn in filter_steps:
            task_metadata = filter_fn(task_metadata)
            filter_history.append((f"Filter to {label}", len(task_metadata)))

        self._sanity_check_task_ids(task_metadata)
        self._print_filter_history(filter_history)
        return task_metadata

    def _parse_task_metadata(self) -> list[TabArenaTaskMetadata]:
        """Resolve the configured `task_metadata` into a list of TabArenaTaskMetadata."""
        task_metadata = self.task_metadata

        if isinstance(task_metadata, str) and (task_metadata == "tabarena-v0.1"):
            task_metadata = self._load_tabarena_v0_1_task_metadata()
        if isinstance(task_metadata, (str, Path)):
            print(f"Loading task metadata from {task_metadata}...")
            task_metadata = pd.read_csv(task_metadata, index_col=False)
        if isinstance(task_metadata, pd.DataFrame):
            task_metadata = [TabArenaTaskMetadata.from_row(row) for _, row in task_metadata.iterrows()]
        assert all(isinstance(x, TabArenaTaskMetadata) for x in task_metadata)
        return task_metadata

    def _filter_by_problem_types(self, task_metadata: list[TabArenaTaskMetadata]) -> list[TabArenaTaskMetadata]:
        return [ttm for ttm in task_metadata if ttm.problem_type in self.problem_types_to_run]

    def _filter_by_split_indices(self, task_metadata: list[TabArenaTaskMetadata]) -> list[TabArenaTaskMetadata]:
        if self.split_indices_to_run is None:
            return task_metadata

        if self.split_indices_to_run == "lite":
            split_indices_to_run = [SplitMetadata.get_split_index(repeat_i=0, fold_i=0)]
        else:
            split_indices_to_run = self.split_indices_to_run

        split_index_pattern = re.compile(r"^r\d+f\d+$")
        for split_index in split_indices_to_run:
            assert split_index_pattern.match(split_index), (
                f"Invalid SplitIndex format: {split_index!r}, expected 'r{{int}}f{{int}}'"
            )

        return [ttm for ttm in task_metadata if ttm.split_index in split_indices_to_run]

    def _filter_by_dataset_names(self, task_metadata: list[TabArenaTaskMetadata]) -> list[TabArenaTaskMetadata]:
        if self.dataset_names_to_run is None:
            return task_metadata

        requested = set(self.dataset_names_to_run)
        available = {ttm.dataset_name for ttm in task_metadata}
        missing = requested - available
        if missing:
            raise ValueError(
                f"Requested dataset names not found in task metadata: {sorted(missing)}. "
                f"Available dataset names: {sorted(available)}"
            )
        return [ttm for ttm in task_metadata if ttm.dataset_name in requested]

    def _filter_by_dtypes(self, task_metadata: list[TabArenaTaskMetadata]) -> list[TabArenaTaskMetadata]:
        if (self.forbidden_dtypes_to_run is None) and (self.required_dtypes_to_run is None):
            return task_metadata
        return [
            ttm
            for ttm in task_metadata
            if ttm.has_supported_dtypes(
                required_dtypes=self.required_dtypes_to_run,
                forbidden_dtypes=self.forbidden_dtypes_to_run,
            )
        ]

    def _filter_by_train_samples(self, task_metadata: list[TabArenaTaskMetadata]) -> list[TabArenaTaskMetadata]:
        if self.n_train_samples_to_run is None:
            return task_metadata

        lb, ub = self.n_train_samples_to_run
        lb = lb if lb is not None else 0
        ub = ub if ub is not None else float("inf")
        return [
            ttm
            for ttm in task_metadata
            if lb < ttm.splits_metadata[ttm.split_index].num_instances_train <= ub
        ]

    @staticmethod
    def _sanity_check_task_ids(task_metadata: list[TabArenaTaskMetadata]) -> None:
        for ttm in task_metadata:
            if ttm.task_id_str is None:
                raise ValueError(f"Task metadata for task {ttm.tabarena_task_name} does not have a task_id_str!")

    @staticmethod
    def _print_filter_history(filter_history: list[tuple[str, int]]) -> None:
        lines = [
            f"Found {filter_history[-1][1]} tasks to run.",
            "\tTask Filter History:",
            *(f"\t({i}) {label}: {count}." for i, (label, count) in enumerate(filter_history, start=1)),
        ]
        print("\n".join(lines))

    @staticmethod
    def _load_tabarena_v0_1_task_metadata() -> list[TabArenaTaskMetadata]:
        """Load TabArena v0.1 task metadata and convert it to the new
        TabArenaTaskMetadata format (one entry per task, with splits unrolled).
        """
        print("Loading task metadata from TabArena v0.1 and converting to new TabArenaTaskMetadata format...")
        from tabarena.nips2025_utils.fetch_metadata import (
            load_curated_task_metadata,
        )

        metric_map = {
            "binary": "roc_auc",
            "multiclass": "log_loss",
            "regression": "rmse",
        }

        metadata = load_curated_task_metadata()
        task_metadata: list[TabArenaTaskMetadata] = []
        for row in metadata.itertuples():
            num_classes = row.num_classes
            num_instances = row.num_instances
            num_features = row.num_features

            n_repeats = row.tabarena_num_repeats
            n_folds = row.num_folds

            eval_metric = metric_map[row.problem_type]

            for repeat_i in range(n_repeats):
                for fold_i in range(n_folds):
                    split_index = SplitMetadata.get_split_index(repeat_i=repeat_i, fold_i=fold_i)
                    splits_metadata = {
                        split_index: SplitMetadata(
                            repeat=repeat_i,
                            fold=fold_i,
                            num_instances_train=num_instances * 2 / 3,
                            num_instances_test=num_instances * 1 / 3,
                            num_instance_groups_train=num_instances * 2 / 3,
                            num_instance_groups_test=num_instances * 1 / 3,
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
        return task_metadata


@dataclass(frozen=True)
class ModelConstraints:
    """Per-model dataset-compatibility constraints.

    A constraint is "active" only when its corresponding field is set
    (non-`None`); unset fields impose no restriction. `regression_support`
    defaults to True — set False for classification-only models.
    """

    max_n_features: int | None = None
    max_n_samples_train_per_fold: int | None = None
    min_n_samples_train_per_fold: int | None = None
    max_n_classes: int | None = None
    regression_support: bool = True

    def applies(
        self,
        *,
        n_features: int,
        n_classes: int,
        n_samples_train_per_fold: int,
        problem_type: str | None = None,
    ) -> bool:
        """True if a dataset with these properties is compatible with the model.

        For regression datasets, `problem_type == "regression"` is the
        authoritative signal — `n_classes` from metadata can be 0/-1/None.
        """
        if problem_type == "regression" and not self.regression_support:
            return False
        if self.max_n_features is not None and n_features > self.max_n_features:
            return False
        if self.max_n_samples_train_per_fold is not None and n_samples_train_per_fold > self.max_n_samples_train_per_fold:
            return False
        if self.min_n_samples_train_per_fold is not None and n_samples_train_per_fold < self.min_n_samples_train_per_fold:
            return False
        return not (self.max_n_classes is not None and n_classes > self.max_n_classes)


# Shared constraints for model families (used by ModelPipelinesToRunSetup.DEFAULT_MODEL_CONSTRAINTS).
_TABICL_CONSTRAINTS = ModelConstraints(
    max_n_samples_train_per_fold=100_000,
    max_n_features=500,
    regression_support=False,
)
_TABPFNV2_CONSTRAINTS = ModelConstraints(
    max_n_samples_train_per_fold=10_000,
    max_n_features=500,
    max_n_classes=10,
)


@dataclass(frozen=True)
class JobCandidate:
    """A single (task split x config) work unit.

    Carries the identifying tuple (`task_id`, `dataset_name`, `fold`,
    `repeat`, `config_index`), the resolved `config` dict, and the dataset
    shape inputs needed by the cache/constraint filter on the Ray side
    (`should_run_job`). Used end-to-end: enumeration -> Ray filtering ->
    scheduler bundling. Only `task_id`/`fold`/`repeat`/`config_index` survive
    into the per-array-task JSON consumed by the runner script.
    """

    task_id: str
    dataset_name: str
    fold: int
    repeat: int
    config_index: int
    config: dict
    n_features: int
    n_classes: int
    n_samples_train_per_fold: int
    problem_type: str


@dataclass
class ModelPipelinesToRunSetup:
    """Defines which models to run in the benchmark, plus model-related behavior.

    Encapsulates the list of models with per-model config counts, preprocessing
    pipelines applied to each model, and miscellaneous model-level settings
    (predict batching, verbosity, artifact paths, fake memory, fold fitting).
    """

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
    preprocessing_pipelines: list[str] = field(default_factory=lambda: ["tabarena_default"])
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
    max_predict_batch_size: int | None = None
    """Maximal batch size for the predict function of the models.
    This is used at validation and test predict time. Thus, it trades off speed for memory usage.
    If None, no limit is applied.
    """
    sequential_local_fold_fitting: bool = False
    """Use Ray for local fold fitting. This is used to speed up the local fold fitting
    and force this behavior if True. If False the default strategy of running the
    local fold fitting is used, as determined by AutoGluon and the model's
    default_ag_args_ensemble parameters."""
    model_artifacts_base_path: str | Path | None = "/tmp"  # noqa: S108
    """Adapt the default temporary directory used for model artifacts in TabArena.
        - If None, the default temporary directory is used: "./AutoGluonModels".
        - If a string or Path, the directory is used as the base path for the temporary
        and any model artifacts will be stored in time-stamped subdirectories.
    """
    model_verbosity: int | None = None
    """Verbosity level passed to the model via model_hyperparameters['verbose'].
    Controls model-level logging (e.g. CatBoost iteration logs, LightGBM verbosity)
    independently of AutoGluon's overall verbosity. If None, no model-level verbosity is set."""
    adapt_num_folds_to_n_classes: bool = True
    """Whether to adapt the number of folds to the number of classes for classification tasks.
    Ensures that each fold has at least one sample of each class.
    """
    dynamic_tabarena_validation_protocol: bool = True
    """If True, the validation data will be adapted dynamically based on the task.
    WARNING: this can overwrite the configured validation of a configuration!"""

    custom_model_constraints: dict[str, ModelConstraints] = field(default_factory=dict)
    """Per-model overrides of dataset compatibility, keyed by AG model key.

    Entries here are merged on top of `DEFAULT_MODEL_CONSTRAINTS` (custom wins
    on key collisions). Models not listed in either map are considered
    compatible with every dataset.
    """

    DEFAULT_MODEL_CONSTRAINTS: ClassVar[dict[str, ModelConstraints]] = {
        "TABICL": _TABICL_CONSTRAINTS,
        "TA-TABICL": _TABICL_CONSTRAINTS,
        "TABPFNV2": _TABPFNV2_CONSTRAINTS,
        "TA-TABPFNV2": _TABPFNV2_CONSTRAINTS,
        "MITRA": _TABPFNV2_CONSTRAINTS,
    }

    @property
    def model_constraints(self) -> dict[str, ModelConstraints]:
        """Effective model constraints (defaults overridden by `custom_model_constraints`)."""
        return {**self.DEFAULT_MODEL_CONSTRAINTS, **self.custom_model_constraints}

    def generate_configs_yaml(
        self,
        *,
        configs_path: str,
        resources_setup: ResourcesSetup,
        verbosity: int,
        shuffle_features: bool,
    ) -> list[dict]:
        """Build experiments, write them to `configs_path`, and return the parsed methods list.

        `verbosity` and `shuffle_features` are bench-level toggles threaded
        through into the per-method kwargs alongside the model-specific
        overrides this dataclass owns.
        """
        from tabarena.benchmark.experiment import (
            YamlExperimentSerializer,
        )

        base_method_kwargs = {
            "init_kwargs": {"verbosity": verbosity},
            "shuffle_features": shuffle_features,
            "fit_kwargs": {},
            "extra_model_hyperparameters": {},
        }

        experiments_all = self.generate_experiments(
            base_method_kwargs=base_method_kwargs,
            resources_setup=resources_setup,
        )

        # Verify no duplicate names
        experiment_names = set()
        for experiment in experiments_all:
            if experiment.name in experiment_names:
                raise AssertionError(
                    f"Found multiple instances of experiment named {experiment.name}. "
                    f"All experiment names must be unique!",
                )
            experiment_names.add(experiment.name)

        YamlExperimentSerializer.to_yaml(
            experiments=experiments_all,
            path=configs_path,
        )

        # Read YAML file and return the methods list.
        with open(configs_path) as file:
            return yaml.safe_load(file)["methods"]

    def generate_experiments(
        self,
        *,
        base_method_kwargs: dict,
        resources_setup: ResourcesSetup,
    ) -> list:
        """Build the full list of experiment configs (one per model x pipeline x random config)."""
        method_kwargs = self._enrich_method_kwargs(base_method_kwargs)
        self._print_experiment_summary(method_kwargs)
        return [
            exp
            for pipeline_name in self.preprocessing_pipelines
            for exp in self._build_experiments_for_pipeline(pipeline_name, method_kwargs, resources_setup)
        ]

    def _enrich_method_kwargs(self, base_method_kwargs: dict) -> dict:
        """Layer this dataclass's model-level overrides on top of the bench-level base."""
        mk = deepcopy(base_method_kwargs)
        if self.model_artifacts_base_path is not None:
            mk["init_kwargs"]["default_base_path"] = self.model_artifacts_base_path
        if not self.model_agnostic_preprocessing:
            mk["fit_kwargs"]["feature_generator"] = None
        if self.adapt_num_folds_to_n_classes:
            mk["fit_kwargs"]["adapt_num_bag_folds_to_n_classes"] = True
        if self.max_predict_batch_size is not None:
            mk["extra_model_hyperparameters"]["ag.max_batch_size"] = self.max_predict_batch_size
        if self.model_verbosity is not None:
            mk["extra_model_hyperparameters"]["ag.verbosity"] = self.model_verbosity
        return mk

    def _print_experiment_summary(self, method_kwargs: dict) -> None:
        print(
            "Generating experiments for models...",
            f"\n\t`all` := number of configs: {self.n_random_configs}",
            f"\n\t{len(self.models)} models: {self.models}",
            f"\n\t{len(self.preprocessing_pipelines)} preprocessing pipelines: {self.preprocessing_pipelines}",
            f"\n\tMethod kwargs: {method_kwargs}",
        )

    def _build_experiments_for_pipeline(
        self,
        pipeline_name: str,
        method_kwargs: dict,
        resources_setup: ResourcesSetup,
    ) -> list:
        """Per-pipeline overrides + per-model dispatch."""
        pipeline_kwargs = deepcopy(method_kwargs)
        name_id_suffix = ""
        if self.model_agnostic_preprocessing:
            if pipeline_name != "default":
                pipeline_kwargs["preprocessing_pipeline"] = pipeline_name
            if pipeline_name != "tabarena_default":
                name_id_suffix = f"_{pipeline_name}"

        experiments: list = []
        for model in self.models:
            experiments.extend(
                self._build_experiments_for_model(model, pipeline_kwargs, name_id_suffix, resources_setup)
            )
        return experiments

    def _build_experiments_for_model(
        self,
        model: Experiment | tuple,
        pipeline_method_kwargs: dict,
        name_id_suffix: str,
        resources_setup: ResourcesSetup,
    ) -> list:
        if isinstance(model, Experiment):
            return [model]
        model_name, n_configs_or_kwargs = model[0], model[1]
        if isinstance(model_name, str) and model_name.startswith("AutoGluon"):
            return self._generate_autogluon_config(
                model_name=model_name,
                agexp_kwargs=n_configs_or_kwargs,
                pipeline_method_kwargs=pipeline_method_kwargs,
                resources_setup=resources_setup,
            )
        return self._generate_model_configs(
            model_name=model_name,
            n_configs=n_configs_or_kwargs,
            pipeline_method_kwargs=pipeline_method_kwargs,
            name_id_suffix=name_id_suffix,
            resources_setup=resources_setup,
        )

    @staticmethod
    def _generate_autogluon_config(
        *,
        model_name: str,
        agexp_kwargs: dict,
        pipeline_method_kwargs: dict,
        resources_setup: ResourcesSetup,
    ) -> list:
        """Parse the AutoGluon config from the models."""
        from tabarena.benchmark.experiment.experiment_constructor import (
            AGExperiment,
        )
        # deepcopy: shallow .copy() leaves nested `fit_kwargs` / `init_kwargs` shared with
        # the caller's dict, and the subsequent .update() / item assignment would mutate
        # the user's `self.models` entry across calls.
        agexp_kwargs = deepcopy(agexp_kwargs)
        for key in ["fit_kwargs", "init_kwargs"]:
            agexp_kwargs.setdefault(key, {})
            if key in pipeline_method_kwargs:
                agexp_kwargs[key].update(pipeline_method_kwargs[key])
        agexp_kwargs["fit_kwargs"]["time_limit"] = resources_setup.time_limit

        return [AGExperiment(name=model_name, **agexp_kwargs)]

    def _generate_model_configs(
        self,
        *,
        model_name: str,
        n_configs: int | str,
        pipeline_method_kwargs: dict,
        name_id_suffix: str,
        resources_setup: ResourcesSetup,
        default_seed_config: str = "fold-config-wise",
    ) -> list:
        from tabarena.models.utils import get_configs_generator_from_name

        if isinstance(n_configs, str) and n_configs == "all":
            n_configs = self.n_random_configs
        elif not isinstance(n_configs, int):
            raise ValueError(
                f"Invalid number of configurations for model {model_name}: {n_configs}. Must be an integer or 'all'."
            )
        if isinstance(model_name, str):
            config_generator = get_configs_generator_from_name(model_name)
        else:
            config_generator = deepcopy(model_name)
        return config_generator.generate_all_bag_experiments(
            num_random_configs=n_configs,
            add_seed=default_seed_config,
            name_id_suffix=name_id_suffix,
            method_kwargs=pipeline_method_kwargs,
            time_limit=resources_setup.time_limit,
            time_limit_with_preprocessing=resources_setup.time_limit_with_model_agnostic_preprocessing,
        )


@dataclass
class TabArenaBenchmarkSetup:
    """Manually set the parameters for the benchmark run."""

    benchmark_name: str
    """Unique name of the benchmark; determines where output artifacts are stored."""

    tasks_to_run_setup: TasksToRunSetup
    """Defines which tasks to run in the benchmark, including the source of
    task metadata and any filters applied on top of it."""

    path_setup: PathSetup = field(default_factory=PathSetup)
    """Contains all path related to the benchmark."""
    scheduler_setup: SchedulerSetup = field(default_factory=SlurmSetup)
    """Scheduler-specific config for the benchmark (defaults to SLURM)."""
    resources_setup: ResourcesSetup = field(default_factory=ResourcesSetup)
    """Compute and time-budget resources for the benchmark jobs."""
    model_pipelines_to_run_setup: ModelPipelinesToRunSetup = field(default_factory=ModelPipelinesToRunSetup)
    """Defines which models and preprocessing pipelines to run in the benchmark,
    along with related fit/validation behavior."""

    # Misc Settings
    # -------------
    shuffle_features: bool = True
    """Whether to shuffle the features of the datasets. Only here for backward compatibility
    with the original TabArena setup, but not recommended to change."""
    parallel_benchmark_name: str | None = None
    """Set this is to some string value to make sure you can run parallel
    jobs for the same benchmark name. This ensures that the config and job .yaml/.json
    files are not overwritten, while SLURM output and TabArena output are still saved
    in the same folders.
    """
    ignore_cache: bool = False
    """If True, will overwrite the cache and run all jobs again."""
    num_ray_cpus: int | Literal["auto"] = "auto"
    """Number of CPUs to use for checking the cache and generating the jobs.
    This should be set to the number of CPUs available to the python script.
    If "auto", we use all available CPUs."""
    verbosity: int = 2
    """Verbosity level for logging and printing."""

    @property
    def _parallel_safe_benchmark_name(self) -> str:
        """Safe benchmark name for file paths."""
        benchmark_name = self.benchmark_name
        if self.parallel_benchmark_name is not None:
            benchmark_name += f"_{self.parallel_benchmark_name}"
        return benchmark_name

    def get_jobs_to_run(self) -> tuple[list[dict], int]:
        """Resolve the work to run for this benchmark.

        Pipeline:
            1. Create the output / log / OpenML-cache directories if missing.
            2. Load task metadata + generate the experiment configs YAML.
            3. Enumerate the cartesian product (task split x config) as
               candidate items.
            4. Drop cache hits and constraint-violating items in parallel via
               Ray (`should_run_job`).
            5. Bundle the survivors into array tasks via
               `scheduler_setup.bundle_items`.

        Returns `(jobs, max_configs_per_job)`: `jobs` is a list of
        `{"items": [...]}` per array task; `max_configs_per_job` is the
        largest bundle observed and is used to budget the per-task time limit.
        """
        self._ensure_runtime_dirs()
        task_metadata_list = self.tasks_to_run_setup.load_task_metadata()
        configs = self.model_pipelines_to_run_setup.generate_configs_yaml(
            configs_path=self.path_setup.get_configs_path(self._parallel_safe_benchmark_name),
            resources_setup=self.resources_setup,
            verbosity=self.verbosity,
            shuffle_features=self.shuffle_features,
        )

        candidates = self._enumerate_candidates(task_metadata_list, configs)
        approved = self._filter_via_ray(candidates)

        jobs, max_configs_per_job = self.scheduler_setup.bundle_items(approved)

        print(
            f"Approved {len(approved)} (task, fold, repeat, config) items"
            f" -> {len(jobs)} array tasks (max {max_configs_per_job} items/task)."
        )
        return jobs, max_configs_per_job

    def _ensure_runtime_dirs(self) -> None:
        """Create the output, log, and (optional) OpenML cache directories."""
        if self.path_setup.openml_cache_path != "auto":
            Path(self.path_setup.openml_cache_path).mkdir(parents=True, exist_ok=True)
        Path(self.path_setup.get_output_path(self.benchmark_name)).mkdir(parents=True, exist_ok=True)
        Path(self.path_setup.get_slurm_log_output_path(self.benchmark_name)).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _enumerate_candidates(
        task_metadata_list: list[TabArenaTaskMetadata],
        configs: list[dict],
    ) -> list[JobCandidate]:
        """Cartesian product of task splits x configs."""
        candidates: list[JobCandidate] = []
        for tm in task_metadata_list:
            split_md = tm.splits_metadata[tm.split_index]
            for config_index, config in enumerate(configs):
                candidates.append(
                    JobCandidate(
                        task_id=tm.task_id_str,
                        dataset_name=tm.dataset_name,
                        fold=split_md.fold,
                        repeat=split_md.repeat,
                        config_index=config_index,
                        config=config,
                        n_features=split_md.num_features_train,
                        n_classes=split_md.num_classes_train,
                        n_samples_train_per_fold=split_md.num_instances_train,
                        problem_type=tm.problem_type,
                    )
                )
        return candidates

    def _filter_via_ray(self, candidates: list[JobCandidate]) -> list[JobCandidate]:
        """Fan out `should_run_job` across Ray workers; return the approved subset."""
        num_ray_cpus = (
            len(os.sched_getaffinity(0))
            if self.num_ray_cpus == "auto"
            else self.num_ray_cpus
        )
        if ray.is_initialized():
            ray.shutdown()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            ray.init(num_cpus=num_ray_cpus)

        batched = ray_map_list(
            list_to_map=list(to_batch_list(candidates, 10_000)),
            func=should_run_job_batch,
            func_element_key_string="candidates",
            num_workers=num_ray_cpus,
            num_cpus_per_worker=1,
            func_kwargs={
                "output_dir": self.path_setup.get_output_path(self.benchmark_name),
                "model_constraints": self.model_pipelines_to_run_setup.model_constraints,
                "ignore_cache": self.ignore_cache,
            },
            track_progress=True,
            tqdm_kwargs={"desc": "Checking Cache and Filter Invalid Jobs"},
        )
        keep_flags = [b for batch in batched for b in batch]

        return [c for keep, c in zip(keep_flags, candidates, strict=True) if keep]

    def get_jobs_dict(self) -> dict:
        """Build the dict consumed by `scheduler_setup.get_run_commands`.

        Contains three pieces of state:
            - `defaults`: per-job runtime arguments shared across all array tasks.
            - `jobs`: list of `{"items": [...]}` array-task bundles.
            - `max_configs_per_job`: worst-case bundle size; used by the
              scheduler to budget the per-task time limit.
        """
        jobs, max_configs_per_job = self.get_jobs_to_run()
        return {
            "defaults": self._build_default_args(),
            "jobs": jobs,
            "max_configs_per_job": max_configs_per_job,
        }

    def _build_default_args(self) -> dict:
        """Per-job runtime defaults serialized into every array-task JSON."""
        return {
            "python": self.path_setup.python_path,
            "run_script": self.path_setup.run_script_path,
            "openml_cache_dir": self.path_setup.openml_cache_path,
            "configs_yaml_file": self.path_setup.get_configs_path(self._parallel_safe_benchmark_name),
            "output_dir": self.path_setup.get_output_path(self.benchmark_name),
            "num_cpus": self.resources_setup.num_cpus,
            "num_gpus": self.resources_setup.num_gpus,
            "num_gpus_model": self.resources_setup.num_gpus_model,
            "memory_limit": self.resources_setup.effective_memory_limit,
            "ignore_cache": self.ignore_cache,
            "sequential_local_fold_fitting": self.model_pipelines_to_run_setup.sequential_local_fold_fitting,
            "dynamic_tabarena_validation_protocol": self.model_pipelines_to_run_setup.dynamic_tabarena_validation_protocol,
            **self.scheduler_setup.get_extra_default_args(),
        }

    def setup_jobs(self) -> list[str] | None:
        """Generate the scheduler job file(s) and return the run commands.

        Delegates persistence and command construction to
        `scheduler_setup.get_run_commands`. Returns `None` when there are no
        jobs to run; in that case the configs YAML for this parallel run is
        also removed so it can be re-prepared cleanly on the next invocation.
        """
        run_commands = self.scheduler_setup.get_run_commands(
            jobs_dict=self.get_jobs_dict(),
            path_setup=self.path_setup,
            benchmark_name=self.benchmark_name,
            parallel_safe_benchmark_name=self._parallel_safe_benchmark_name,
            resources_setup=self.resources_setup,
        )
        if run_commands is None:
            Path(self.path_setup.get_configs_path(self._parallel_safe_benchmark_name)).unlink(missing_ok=True)
        return run_commands

def should_run_job_batch(*, candidates: list[JobCandidate], **kwargs) -> list[bool]:
    """Batched version for Ray."""
    return [should_run_job(candidate=c, **kwargs) for c in candidates]


def _resolve_ag_key(config: dict) -> str:
    """Resolve the AutoGluon model key from a serialized experiment config.

    `model_cls` (AGModelExperiment) and `method_cls` (plain Experiment) both
    carry the class identifier as a dotted import path. Resolve back to the
    class and use `ag_key` so the lookup matches the AG-key-based
    `model_constraints` dict. Fall back to "AutoGluon" for the full-pipeline
    AutoGluon experiments (which expose neither field).
    """
    raw_cls = config.get("model_cls") or config.get("method_cls")
    if raw_cls is not None:
        try:
            cls_obj = resolve_class(raw_cls, registry_resolver=infer_model_cls)
            return getattr(cls_obj, "ag_key", None) or raw_cls
        except (ImportError, AttributeError, ValueError, TypeError):
            return raw_cls
    if config.get("name", "").startswith("AutoGluon"):
        return "AutoGluon"
    return config.get("name", "")


def should_run_job(
    *,
    candidate: JobCandidate,
    output_dir: str,
    model_constraints: dict[str, ModelConstraints],
    ignore_cache: bool,
) -> bool:
    """Decide whether a candidate's job should run (skip on cache hit / constraint violation).

    Module-level so Ray workers can pickle it; reads everything it needs off
    the `JobCandidate` dataclass.
    """
    # Normalize task_id: numeric OpenML IDs are ints; UserTask IDs are
    # "<source>|<local_id>|..." strings that we split to the local part.
    try:
        task_id = int(candidate.task_id)
    except ValueError:
        task_id = candidate.task_id.split("|", 2)[1]

    constraints = model_constraints.get(_resolve_ag_key(candidate.config))
    if constraints is not None and not constraints.applies(
        n_features=candidate.n_features,
        n_classes=candidate.n_classes,
        n_samples_train_per_fold=candidate.n_samples_train_per_fold,
        problem_type=candidate.problem_type,
    ):
        return False

    if ignore_cache:
        return True

    return not check_cache_hit(
        result_dir=output_dir,
        method_name=candidate.config["name"],
        task_id=task_id,
        fold=candidate.fold,
        repeat=candidate.repeat,
        cache_path_format="name_first",
        cache_cls=CacheFunctionPickle,
        cache_cls_kwargs={"include_self_in_call": True},
        mode="local",
    )
