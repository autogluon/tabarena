"""Top-level benchmark orchestrator tying the setup pieces together."""

from __future__ import annotations

import os
import shutil
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import ray

from tabarena.benchmark.experiment import JobBatch, build_jobs, job_cache_exists_batch
from tabarena.utils.ray_utils import ray_map_list, to_batch_list

if TYPE_CHECKING:
    from tabarena.benchmark.experiment import Job, TabArenaExperimentBundle
    from tabarena.benchmark.task.metadata import TaskMetadataCollection
    from tabflow_slurm.setup.paths import PathSetup
    from tabflow_slurm.setup.resources import ResourcesSetup
    from tabflow_slurm.setup.scheduler import SchedulerSetup


@dataclass
class TabArenaBenchmarkSetup:
    """A single homogeneous benchmark run (one resources/scheduler/tasks/experiment).

    Internal engine — not part of the public API. Construct and drive it via
    `TabArenaBenchmarkPlan` (see `tabflow_slurm.setup.plan`), which resolves
    per-model overrides, groups jobs, and builds one of these per group.
    """

    benchmark_name: str
    """Unique name of the benchmark; determines where output artifacts are stored."""

    tasks: TaskMetadataCollection
    """The tasks (dataset x split) to run in the benchmark — already filtered
    (see `TaskMetadataCollection.subset_tasks`) but not necessarily materialized;
    `get_jobs_to_run` materializes it (downloads only this collection's tasks)."""
    experiment_bundle: TabArenaExperimentBundle
    """Defines which models / experiments to run in the benchmark and builds them
    (models, per-model config counts, preprocessing pipelines, fold fitting,
    per-model constraints, and the dynamic validation protocol)."""

    path_setup: PathSetup
    """Contains all paths related to the benchmark. Requires a `workspace`
    directory and the `python_path`; `run_script` and `submit_script` default
    to the scripts bundled with this package."""
    scheduler_setup: SchedulerSetup
    """Scheduler-specific config for the benchmark (defaults to SLURM)."""
    resources_setup: ResourcesSetup
    """Compute and time-budget resources for the benchmark jobs."""

    # Misc Settings
    # -------------
    parallel_safe_benchmark_name: str | None = None
    """Per-run name for the job-batch/job-JSON setup artifacts, so parallel runs
    of the same `benchmark_name` don't overwrite each other (SLURM output and
    TabArena output still share the `benchmark_name` folders). Set by
    `TabArenaBenchmarkPlan` (one per group); falls back to `benchmark_name` for a
    single, non-parallel run."""
    ignore_cache: bool = False
    """If True, will overwrite the cache and run all jobs again."""
    num_ray_cpus: int | Literal["auto"] = "auto"
    """Number of CPUs to use for checking the cache and generating the jobs.
    This should be set to the number of CPUs available to the python script.
    If "auto", we use all available CPUs."""

    @property
    def _safe_benchmark_name(self) -> str:
        """Per-run name for setup artifacts; falls back to `benchmark_name`."""
        return self.parallel_safe_benchmark_name or self.benchmark_name

    @property
    def _job_batch_dir(self) -> str:
        """Directory of this run's `JobBatch` artifact (per parallel run)."""
        return self.path_setup.get_job_batch_dir(
            benchmark_name=self.benchmark_name,
            safe_benchmark_name=self._safe_benchmark_name,
        )

    def get_jobs_to_run(self) -> tuple[list[dict], int]:
        """Resolve the work to run for this benchmark.

        Pipeline:
            1. Create the output / log / setup / OpenML-cache directories if missing.
            2. Materialize the task collection (download only its tasks) and build
               the experiments from the bundle (which attaches each experiment's
               `ModelConstraints`).
            3. Enumerate the sweep via the core `build_jobs` (experiments x splits;
               constraint-violating pairs are dropped during enumeration).
            4. Drop cache hits in parallel via Ray (the core `job_cache_exists_batch`),
               unless `ignore_cache`.
            5. Persist the surviving jobs as a self-contained `JobBatch` artifact
               (experiments + task metadata + job coordinates) for the compute nodes.
            6. Bundle the survivors into array tasks via `scheduler_setup.bundle_items`.

        Returns `(jobs, max_configs_per_job)`: `jobs` is a list of
        `{"items": [...]}` per array task; `max_configs_per_job` is the
        largest bundle observed and is used to budget the per-task time limit.
        """
        self.path_setup.ensure_runtime_dirs(self.benchmark_name)
        collection = self.tasks.materialize()
        experiments = self.experiment_bundle.build_experiments(
            time_limit=self.resources_setup.time_limit,
            num_cpus=self.resources_setup.num_cpus,
            num_gpus=self.resources_setup.effective_num_gpus_model,
            memory_limit=self.resources_setup.effective_memory_limit,
            time_limit_with_preprocessing=self.resources_setup.time_limit_with_model_agnostic_preprocessing,
        )

        # Constraint-violating (experiment, split) pairs are dropped by build_jobs itself
        # (the bundle attached each experiment's ModelConstraints at build time).
        runnable_jobs = build_jobs(experiments, collection)
        if not self.ignore_cache:
            runnable_jobs = self._drop_cache_hits_via_ray(runnable_jobs, collection)

        self._save_job_batch(runnable_jobs, collection)
        jobs, max_configs_per_job = self.scheduler_setup.bundle_items(runnable_jobs, collection)

        print(
            f"Approved {len(runnable_jobs)} (experiment, dataset, fold, repeat) items"
            f" -> {len(jobs)} array tasks (max {max_configs_per_job} items/task).",
        )
        return jobs, max_configs_per_job

    def _drop_cache_hits_via_ray(self, jobs: list[Job], collection: TaskMetadataCollection) -> list[Job]:
        """Fan out the cache check across Ray workers; return the not-yet-cached subset.

        Each job is projected to a plain ``(method_name, task_id_str, fold, repeat)``
        tuple — no live experiments are pickled to workers — and checked through the
        core, writer-aligned ``job_cache_exists_batch``.
        """
        task_id_by_dataset = {ttm.tabarena_task_name: ttm.task_id_str for ttm in collection}
        items = [
            (job.experiment.name, task_id_by_dataset[job.task.dataset], job.task.fold, job.task.repeat) for job in jobs
        ]

        num_ray_cpus = len(os.sched_getaffinity(0)) if self.num_ray_cpus == "auto" else self.num_ray_cpus
        if ray.is_initialized():
            ray.shutdown()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            ray.init(num_cpus=num_ray_cpus)

        batched = ray_map_list(
            list_to_map=list(to_batch_list(items, 10_000)),
            func=job_cache_exists_batch,
            func_element_key_string="items",
            num_workers=num_ray_cpus,
            num_cpus_per_worker=1,
            func_kwargs={"output_dir": self.path_setup.get_output_path(self.benchmark_name)},
            track_progress=True,
            tqdm_kwargs={"desc": "Checking Cache"},
        )
        cached_flags = [b for batch in batched for b in batch]
        return [job for cached, job in zip(cached_flags, jobs, strict=True) if not cached]

    def _save_job_batch(self, jobs: list[Job], collection: TaskMetadataCollection) -> None:
        """Persist the surviving jobs as the run's `JobBatch` artifact (or clean it up).

        With no jobs there is nothing to ship; remove any stale artifact from a
        previous invocation so the setup output always reflects this run.
        """
        if jobs:
            JobBatch(jobs=jobs, task_metadata=collection).save(self._job_batch_dir)
        else:
            shutil.rmtree(self._job_batch_dir, ignore_errors=True)

    def get_jobs_dict(self) -> dict:
        """Build the dict consumed by `scheduler_setup.get_run_commands`.

        Contains three pieces of state:
            - `defaults`: per-job runtime arguments shared across all array tasks.
            - `jobs`: list of `{"items": [...]}` array-task bundles.
            - `max_configs_per_job`: worst-case bundle size (informational; the
              scheduler now budgets each array's time limit per its own bundle
              size, see `SlurmSetup._write_job_batches_and_build_commands`).
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
            "job_batch_dir": self._job_batch_dir,
            "output_dir": self.path_setup.get_output_path(self.benchmark_name),
            "num_cpus": self.resources_setup.num_cpus,
            "num_gpus": self.resources_setup.num_gpus,
            "memory_limit": self.resources_setup.effective_memory_limit,
            "ignore_cache": self.ignore_cache,
            **self.scheduler_setup.get_extra_default_args(),
        }

    def setup_jobs(self, *, print_run_commands: bool = True) -> list[str] | None:
        """Generate the scheduler job file(s) and return the run commands.

        Delegates persistence and command construction to
        `scheduler_setup.get_run_commands`. Returns `None` when there are no
        jobs to run; in that case the `JobBatch` artifact for this parallel run
        was already removed (see `_save_job_batch`) so it can be re-prepared
        cleanly on the next invocation.

        `print_run_commands` controls whether the scheduler prints its own
        run-command summary; `TabArenaBenchmarkPlan` sets this to False so it can
        print one consolidated summary across all runs instead.
        """
        return self.scheduler_setup.get_run_commands(
            jobs_dict=self.get_jobs_dict(),
            path_setup=self.path_setup,
            benchmark_name=self.benchmark_name,
            parallel_safe_benchmark_name=self._safe_benchmark_name,
            resources_setup=self.resources_setup,
            print_summary=print_run_commands,
        )
