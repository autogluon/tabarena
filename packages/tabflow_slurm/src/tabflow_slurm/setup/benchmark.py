"""Top-level benchmark orchestrator tying the setup pieces together."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import ray

from tabarena.utils.ray_utils import ray_map_list, to_batch_list
from tabflow_slurm.setup.candidates import JobCandidate, should_run_job_batch

if TYPE_CHECKING:
    from tabarena.benchmark.experiment import TabArenaExperimentBundle
    from tabarena.benchmark.task.metadata import TabArenaMetadataBundle, TabArenaTaskMetadata
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

    tasks_to_run_setup: TabArenaMetadataBundle
    """Defines which tasks to run in the benchmark, including the source of
    task metadata and any filters applied on top of it."""
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
    """Per-run name for the config/job `.yaml`/`.json` setup artifacts, so
    parallel runs of the same `benchmark_name` don't overwrite each other (SLURM
    output and TabArena output still share the `benchmark_name` folders). Set by
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

    def get_jobs_to_run(self) -> tuple[list[dict], int]:
        """Resolve the work to run for this benchmark.

        Pipeline:
            1. Create the output / log / setup / OpenML-cache directories if missing.
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
        self.path_setup.ensure_runtime_dirs(self.benchmark_name)
        task_metadata_list = self.tasks_to_run_setup.load_task_metadata()
        configs = self.experiment_bundle.generate_configs_yaml(
            configs_path=self.path_setup.get_configs_path(
                benchmark_name=self.benchmark_name,
                safe_benchmark_name=self._safe_benchmark_name,
            ),
            time_limit=self.resources_setup.time_limit,
            num_cpus=self.resources_setup.num_cpus,
            num_gpus=self.resources_setup.effective_num_gpus_model,
            memory_limit=self.resources_setup.effective_memory_limit,
            time_limit_with_preprocessing=self.resources_setup.time_limit_with_model_agnostic_preprocessing,
        )

        candidates = self._enumerate_candidates(task_metadata_list, configs)
        approved = self._filter_via_ray(candidates)

        jobs, max_configs_per_job = self.scheduler_setup.bundle_items(approved)

        print(
            f"Approved {len(approved)} (task, fold, repeat, config) items"
            f" -> {len(jobs)} array tasks (max {max_configs_per_job} items/task).",
        )
        return jobs, max_configs_per_job

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
                    ),
                )
        return candidates

    def _filter_via_ray(self, candidates: list[JobCandidate]) -> list[JobCandidate]:
        """Fan out `should_run_job` across Ray workers; return the approved subset."""
        num_ray_cpus = len(os.sched_getaffinity(0)) if self.num_ray_cpus == "auto" else self.num_ray_cpus
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
                "model_constraints": self.experiment_bundle.model_constraints,
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
            "configs_yaml_file": self.path_setup.get_configs_path(
                benchmark_name=self.benchmark_name,
                safe_benchmark_name=self._safe_benchmark_name,
            ),
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
        jobs to run; in that case the configs YAML for this parallel run is
        also removed so it can be re-prepared cleanly on the next invocation.

        `print_run_commands` controls whether the scheduler prints its own
        run-command summary; `TabArenaBenchmarkPlan` sets this to False so it can
        print one consolidated summary across all runs instead.
        """
        run_commands = self.scheduler_setup.get_run_commands(
            jobs_dict=self.get_jobs_dict(),
            path_setup=self.path_setup,
            benchmark_name=self.benchmark_name,
            parallel_safe_benchmark_name=self._safe_benchmark_name,
            resources_setup=self.resources_setup,
            print_summary=print_run_commands,
        )
        if run_commands is None:
            Path(
                self.path_setup.get_configs_path(
                    benchmark_name=self.benchmark_name,
                    safe_benchmark_name=self._safe_benchmark_name,
                ),
            ).unlink(missing_ok=True)
        return run_commands
