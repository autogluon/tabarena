"""Scheduler abstractions: generic batching plus the SLURM implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from tabarena.utils.ray_utils import to_batch_list

if TYPE_CHECKING:
    from tabarena.benchmark.experiment import Job
    from tabarena.benchmark.task.metadata import SplitMetadata, TaskMetadataCollection
    from tabflow_slurm.setup.paths import PathSetup
    from tabflow_slurm.setup.resources import ResourcesSetup


@dataclass(kw_only=True)
class SchedulerSetup:
    """Base class for scheduler-specific job setup.

    Owns the generic job-batching knobs (`bundle_size`,
    `bundle_size_per_dataset`) shared by any array-style scheduler. Subclasses
    implement how to construct the run commands a user invokes and contribute
    any scheduler-specific entries to the per-job defaults via
    `get_extra_default_args`.
    """

    bundle_size: int = 5
    """Number of `(experiment, dataset, fold, repeat)` items batched into a
    single array task. Items may span different `(dataset, fold, repeat)`
    groups.

    Mirrors the design used by `tabflow` (the SageMaker side) — see
    `tabflow/cli/launch_jobs.py::get_tasks_batched`."""

    bundle_size_per_dataset: dict[str, int] | None = None
    """Optional per-dataset override of `bundle_size`, keyed by the dataset name
    (the collection's `tabarena_task_name` — the runner's results `dataset` key).
    Items from datasets not listed here use `bundle_size`. Items belonging to
    different effective bundle sizes are not mixed in the same bundle.
    An explicit entry here takes precedence over the large-dataset auto-rule
    below (so it doubles as an escape hatch for large datasets)."""

    large_dataset_n_features: int | None = 5_000
    """Wide datasets (`n_features` greater than this) are bundled one item per
    array task (effective `bundle_size=1`), since each fit is expensive enough
    that batching hurts scheduling. Set to `None` to disable the feature check."""

    large_dataset_n_samples: int | None = 100_000
    """Large datasets (`n_samples_train_per_fold` greater than this) are bundled
    one item per array task (effective `bundle_size=1`). Set to `None` to
    disable the sample-count check."""

    def get_run_commands(
        self,
        *,
        jobs_dict: dict,
        path_setup: PathSetup,
        benchmark_name: str,
        parallel_safe_benchmark_name: str,
        resources_setup: ResourcesSetup,
        print_summary: bool = True,
    ) -> list[str] | None:
        """Persist `jobs_dict` and return the run commands a user should invoke.

        Returns `None` when there is no work to launch (so the caller can
        clean up any stale per-run artifacts). When `print_summary` is False the
        scheduler stays quiet (no run-command block / "no jobs" notice) so a
        caller like `TabArenaBenchmarkPlan` can print one consolidated summary.

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

    def bundle_items(self, jobs: list[Job], task_metadata: TaskMetadataCollection) -> tuple[list[dict], int]:
        """Group approved jobs into array-task bundles.

        Jobs are partitioned by their effective bundle size (see
        `_effective_bundle_size`; each job's dataset shape is looked up in
        `task_metadata`) so jobs with different effective sizes never share a
        task. Returns `(array_tasks, max_configs_per_job)` where each array task
        is `{"bundle_size": <target size>, "items": [...]}`: `items` holds the
        self-describing `(experiment, dataset, fold, repeat)` coordinates the
        runner resolves against the shipped `JobBatch`, and `bundle_size` is a
        scheduling-only hint (the target size this task was packed for) that the
        scheduler groups on; it is stripped before the per-array JSON is written.
        `max_configs_per_job` is the largest bundle observed (used by schedulers
        to budget the per-task time limit).
        """
        split_index = {
            (ttm.tabarena_task_name, split.fold, split.repeat): split
            for ttm in task_metadata
            for split in ttm.splits_metadata.values()
        }
        by_size: dict[int, list[Job]] = {}
        for job in jobs:
            size = self._effective_bundle_size(job, split=split_index[job.task.as_triple()])
            by_size.setdefault(size, []).append(job)

        array_tasks: list[dict] = []
        max_configs_per_job = 1
        # Sort sizes for deterministic job ordering across runs.
        for size in sorted(by_size):
            for bundle in to_batch_list(by_size[size], size):
                max_configs_per_job = max(max_configs_per_job, len(bundle))
                array_tasks.append(
                    {
                        # Scheduling-only hint: the target bundle size this task
                        # was packed for. The last bundle of a size group is a
                        # remainder (fewer items than `size`); tagging it lets the
                        # scheduler keep it in its size group's array rather than
                        # emitting a separate command for the odd length. Stripped
                        # before the per-array JSON is written (see
                        # `_write_job_batches_and_build_commands`).
                        "bundle_size": size,
                        "items": [
                            {
                                "experiment": job.experiment.name,
                                "dataset": job.task.dataset,
                                "fold": job.task.fold,
                                "repeat": job.task.repeat,
                            }
                            for job in bundle
                        ],
                    },
                )
        return array_tasks, max_configs_per_job

    def _effective_bundle_size(self, job: Job, *, split: SplitMetadata) -> int:
        """Effective bundle size for a single job (with its split's shape metadata).

        Precedence: an explicit `bundle_size_per_dataset` entry wins; otherwise
        large datasets (see `_is_large_dataset`) collapse to 1; otherwise the
        default `bundle_size` applies.
        """
        overrides = self.bundle_size_per_dataset or {}
        if job.task.dataset in overrides:
            return overrides[job.task.dataset]
        if self._is_large_dataset(split):
            return 1
        return self.bundle_size

    def _is_large_dataset(self, split: SplitMetadata) -> bool:
        """Whether a split's dataset shape exceeds either large-dataset threshold."""
        return (
            self.large_dataset_n_features is not None and split.num_features_train > self.large_dataset_n_features
        ) or (self.large_dataset_n_samples is not None and split.num_instances_train > self.large_dataset_n_samples)


@dataclass(kw_only=True)
class LocalSequentialSetup(SchedulerSetup):
    """Run the benchmark on the local machine, one item at a time.

    The non-cluster counterpart to `SlurmSetup`: instead of emitting `sbatch`
    array commands, it writes the same job JSON and returns a single command that
    invokes the bundled local runner (`run_local.py`). That runner flattens every
    bundled `(task, fold, repeat, config)` item and executes
    `run_tabarena_experiment.py` once per item, sequentially. By default each item
    runs in its own subprocess, exactly like a single SLURM array task, so model
    fits stay isolated from one another; set `execution_mode="in_process"` to run
    them all in one Python process instead.

    It subclasses the base `SchedulerSetup` (not `SlurmSetup`), so it has no
    partitions, gres, or time budget — those concepts don't apply locally. The
    inherited `bundle_items` still applies, but since everything runs
    sequentially the bundling is purely cosmetic (the runner iterates items
    regardless of how they are grouped).
    """

    continue_on_error: bool = False
    """If True, a failing item is logged and the runner keeps going; otherwise
    the runner stops at the first failure. Either way the runner exits non-zero
    if any item failed."""

    execution_mode: Literal["subprocess", "in_process"] = "subprocess"
    """How the local runner executes each item:
        - "subprocess" (default): one fresh subprocess per item — isolates each
          fit's memory / Ray / GPU state, exactly like a SLURM array task. Robust
          for long or heavy sweeps.
        - "in_process": run every item in the runner's own Python process. Faster
          (no per-item interpreter startup / library re-import) and debugger
          friendly, but fits share global state and a hard crash (segfault / OOM
          kill) aborts the whole run instead of a single item."""

    def get_extra_default_args(self) -> dict:
        """Pin the SLURM shared-resources Ray hint off for local runs.

        Keeps the per-job `defaults` dict the same shape as the SLURM path while
        ensuring the runner never performs the shared-filesystem Ray temp-dir
        setup (`setup_slurm_job` skips it when this is False).
        """
        return {"setup_ray_for_slurm_shared_resources_environment": False}

    def get_run_commands(
        self,
        *,
        jobs_dict: dict,
        path_setup: PathSetup,
        benchmark_name: str,
        parallel_safe_benchmark_name: str,
        resources_setup: ResourcesSetup,
        print_summary: bool = True,
    ) -> list[str] | None:
        """Persist `jobs_dict` to one JSON file and return the local-runner command.

        `resources_setup` is unused (no partition/time budget locally). Returns a
        single-element command list invoking `python -m tabflow_slurm.run_local
        <json>`, or `None` when there are no jobs (in which case the JSON is also
        removed, mirroring `SlurmSetup`).
        """
        # Reuse the SLURM job-JSON path; the `slurm_run_data_*` filename is
        # cosmetic — the file just holds `{"defaults": ..., "jobs": ...}`.
        json_path = path_setup.get_slurm_job_json_path(
            benchmark_name=benchmark_name,
            safe_benchmark_name=parallel_safe_benchmark_name,
        )

        all_jobs = jobs_dict["jobs"]
        if not all_jobs:
            if print_summary:
                print("No jobs to run.")
            Path(json_path).unlink(missing_ok=True)
            return None

        with Path(json_path).open("w") as f:
            json.dump({"defaults": jobs_dict["defaults"], "jobs": all_jobs}, f)

        command = f"{jobs_dict['defaults']['python']} -m tabflow_slurm.run_local {json_path}"
        if self.continue_on_error:
            command += " --continue_on_error True"
        if self.execution_mode != "subprocess":
            command += f" --execution_mode {self.execution_mode}"
        run_commands = [command]

        if print_summary:
            print(
                "##### Setup Jobs"
                "\nRun the following command(s) to start the jobs locally:"
                "\n" + "\n".join(run_commands) + "\n",
            )
        return run_commands


@dataclass(kw_only=True)
class SlurmSetup(SchedulerSetup):
    """Setup for SLURM jobs. Adjust as needed for your cluster setup."""

    gpu_partition: str
    """SLURM partition to use for GPU jobs."""
    cpu_partition: str
    """SLURM partition to use for CPU jobs."""
    extra_gres: str | None
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
        print_summary: bool = True,
    ) -> list[str] | None:
        """Persist `jobs_dict` to one or more JSON files and return matching sbatch commands.

        Array tasks are first grouped by their bundle size (number of configs
        per task) so each SLURM array gets a `--time` budgeted for the work its
        tasks actually do — singleton large-dataset tasks must not inherit the
        time limit of the larger bundles they happen to be enumerated alongside.
        Within a size group, jobs exceeding `max_array_size` are further split
        into multiple array-job batches. Each emitted array gets its own JSON
        file (suffixed with its size and/or batch index when more than one) and
        its own sbatch command. All SLURM-specific paths (job JSON, log dir,
        submit script) are derived from `path_setup` + the two benchmark name
        flavors.

        Returns the list of sbatch commands to run, or `None` if there are
        no jobs (in which case the base JSON file is also removed if it exists).
        """
        base_json_path = path_setup.get_slurm_job_json_path(
            benchmark_name=benchmark_name,
            safe_benchmark_name=parallel_safe_benchmark_name,
        )

        all_jobs = jobs_dict["jobs"]
        if not all_jobs:
            if print_summary:
                print("No jobs to run.")
            Path(base_json_path).unlink(missing_ok=True)
            return None

        run_commands = self._write_job_batches_and_build_commands(
            all_jobs=all_jobs,
            defaults=jobs_dict["defaults"],
            base_json_path=base_json_path,
            resources_setup=resources_setup,
            slurm_log_output=path_setup.get_slurm_log_output_path(benchmark_name),
            slurm_script_path=path_setup.submit_script_path,
        )

        if print_summary:
            n_batches = len(run_commands)
            batch_info = (
                f"\nSplit into {n_batches} SLURM arrays (one per bundle size, "
                f"each capped at {self.max_array_size} tasks)."
                if n_batches > 1
                else ""
            )
            print(
                "##### Setup Jobs"
                f"{batch_info}"
                "\nRun the following command(s) to start the jobs:"
                "\n" + "\n".join(run_commands) + "\n",
            )
        return run_commands

    def _write_job_batches_and_build_commands(
        self,
        *,
        all_jobs: list,
        defaults: dict,
        base_json_path: str,
        resources_setup: ResourcesSetup,
        slurm_log_output: str,
        slurm_script_path: str,
    ) -> list[str]:
        """Group `all_jobs` by bundle size, split each group into `max_array_size`
        batches, persist each batch as JSON, and return the matching
        `sbatch --array=...` commands.

        Each size group gets its own sbatch prefix whose `--time` is budgeted for
        that group's number of configs per task (so singleton large-dataset
        arrays don't inherit a larger bundle's time limit). With a single array
        the JSON is written to `base_json_path`; otherwise the path is suffixed
        with `_size{n}` and/or `_batch{i}` so each array has its own file.
        """
        # Group on the target bundle size, not the actual item count, so a size
        # group's remainder bundle (fewer items than the target) stays in its own
        # group instead of spawning a separate, near-empty array. Genuinely
        # different target sizes (e.g. large datasets collapsed to 1) still split.
        jobs_by_size: dict[int, list] = {}
        for job in all_jobs:
            jobs_by_size.setdefault(job["bundle_size"], []).append(job)
        multi_size = len(jobs_by_size) > 1

        run_commands: list[str] = []
        for size in sorted(jobs_by_size):
            base_command = self._build_sbatch_prefix(
                resources_setup=resources_setup,
                configs_per_job=size,
                slurm_log_output=slurm_log_output,
                slurm_script_path=slurm_script_path,
            )
            job_batches = [list(b) for b in to_batch_list(jobs_by_size[size], self.max_array_size)]
            multi_batch = len(job_batches) > 1

            for batch_idx, batch_jobs in enumerate(job_batches):
                suffix = f"{f'_size{size}' if multi_size else ''}{f'_batch{batch_idx}' if multi_batch else ''}"
                json_path = base_json_path.replace(".json", f"{suffix}.json") if suffix else base_json_path
                # Drop the scheduling-only `bundle_size` hint so the shipped JSON
                # keeps the `{"items": [...]}` shape the node runner expects.
                shipped_jobs = [{"items": job["items"]} for job in batch_jobs]
                with Path(json_path).open("w") as f:
                    json.dump({"defaults": defaults, "jobs": shipped_jobs}, f)

                run_commands.append(
                    f"sbatch --array=0-{len(batch_jobs) - 1}%{self.array_job_limit} {base_command} {json_path}",
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

        time_in_h = resources_setup.time_limit_per_config // 3600 * configs_per_job + self.time_limit_overhead

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


@dataclass(kw_only=True)
class GCPSlurmSetup(SlurmSetup):
    """Default Slurm setup for a GCP Slurm cluster used for BeyondArena."""

    gpu_partition: str = "gpua100highmemoryspotmt"
    cpu_partition: str = "cpuhighmem16mtspot"
    extra_gres: None = None
    exclusive_node: bool = True
