"""Scheduler abstractions: generic batching plus the SLURM implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tabarena.utils.ray_utils import to_batch_list

if TYPE_CHECKING:
    from tabflow_slurm.setup.candidates import JobCandidate
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
    """Number of `(task_id, fold, repeat, config_index)` items batched into a
    single array task. Items may span different `(task_id, fold, repeat)`
    groups.

    Mirrors the design used by `tabflow` (the SageMaker side) — see
    `tabflow/cli/launch_jobs.py::get_tasks_batched`."""

    bundle_size_per_dataset: dict[str, int] | None = None
    """Optional per-dataset override of `bundle_size`, keyed by `dataset_name`.
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

    def bundle_items(self, approved: list[JobCandidate]) -> tuple[list[dict], int]:
        """Group approved candidates into array-task bundles.

        Candidates are partitioned by their effective bundle size (see
        `_effective_bundle_size`) so candidates with different effective sizes
        never share a task. Returns `(jobs, max_configs_per_job)` where each job
        is `{"items": [...]}` (JSON-serializable, only the identifying-tuple
        fields), and `max_configs_per_job` is the largest bundle observed
        (used by schedulers to budget the per-task time limit).
        """
        by_size: dict[int, list[JobCandidate]] = {}
        for c in approved:
            by_size.setdefault(self._effective_bundle_size(c), []).append(c)

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

    def _effective_bundle_size(self, c: JobCandidate) -> int:
        """Effective bundle size for a single candidate.

        Precedence: an explicit `bundle_size_per_dataset` entry wins; otherwise
        large datasets (see `_is_large_dataset`) collapse to 1; otherwise the
        default `bundle_size` applies.
        """
        overrides = self.bundle_size_per_dataset or {}
        if c.dataset_name in overrides:
            return overrides[c.dataset_name]
        if self._is_large_dataset(c):
            return 1
        return self.bundle_size

    def _is_large_dataset(self, c: JobCandidate) -> bool:
        """Whether a candidate's dataset exceeds either large-dataset threshold."""
        return (self.large_dataset_n_features is not None and c.n_features > self.large_dataset_n_features) or (
            self.large_dataset_n_samples is not None and c.n_samples_train_per_fold > self.large_dataset_n_samples
        )


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

        If `jobs_dict["jobs"]` exceeds `max_array_size`, the jobs are split across
        multiple array-job batches; each batch gets its own `_batch{i}.json`
        file and its own sbatch command (the single-batch case keeps the base
        path). All SLURM-specific paths (job JSON, log dir, submit script) are
        derived from `path_setup` + the two benchmark name flavors.

        The time-budget computation needs to know the worst-case number of
        configs handled by a single array task; this is read from
        `jobs_dict["max_configs_per_job"]` (populated by the caller).

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

        base_command = self._build_sbatch_prefix(
            resources_setup=resources_setup,
            configs_per_job=jobs_dict["max_configs_per_job"],
            slurm_log_output=path_setup.get_slurm_log_output_path(benchmark_name),
            slurm_script_path=path_setup.submit_script_path,
        )

        run_commands = self._write_job_batches_and_build_commands(
            all_jobs=all_jobs,
            defaults=jobs_dict["defaults"],
            base_json_path=base_json_path,
            base_command=base_command,
        )

        if print_summary:
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
                "\n" + "\n".join(run_commands) + "\n",
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
            json_path = base_json_path.replace(".json", f"_batch{batch_idx}.json") if multi_batch else base_json_path
            with Path(json_path).open("w") as f:
                json.dump({"defaults": defaults, "jobs": batch_jobs}, f)

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
