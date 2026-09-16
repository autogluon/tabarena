"""Run a benchmark plan on SkyPilot managed jobs that drain a GCS claim queue.

The SLURM scheduler relies on a filesystem the head node shares with every compute node: the
``JobBatch`` and the job JSON are read from it, results are written into it. A SkyPilot worker is
a plain GCE VM with nothing in common with the head node, so :class:`SkyPilotSetup` moves that
exchange into a bucket and ships the environment along:

* ``setup`` freezes the head venv and archives every local checkout it installs
  (:mod:`tabflow_slurm.setup.sky_env`), copies the ``JobBatch`` and one task JSON per bundle to
  ``<bucket>/<prefix>/runs/<benchmark>/queue/<launch id>/``, and renders the SkyPilot YAML.
* The printed command is one ``sky jobs launch --num-jobs N``. Each managed job (or each job on a
  pool, see :attr:`SkyPilotSetup.use_pool`) builds the venv from the staged manifest, then runs
  :mod:`tabflow_slurm.sky_worker`, which claims bundles with an exclusive create, fits their items
  with the bundled runner (``--cache_root`` and ``--materialize_tasks`` make the VM self-sufficient)
  and copies each result into ``runs/<benchmark>/output/data/``.
* Before the head node's cache check and before an evaluation, :meth:`SkyPilotSetup.sync_results_to_local`
  mirrors that ``output/data`` prefix into the workspace, so relaunches skip finished items and
  ``eval`` reads the same paths it reads after a SLURM run.

Everything goes through the cluster's ``sky`` CLI and its shared API server, which resolves the
accelerator names of this cluster (``RTXPRO6000`` is a spot ``g4-standard-48``) and whose admin
policy expands a request across every region the VPC reaches. The YAML therefore names no cloud,
region or zone and, by default, no ``infra`` either; :attr:`SkyPilotSetup.infra` exists for a
server without that policy (a local upstream API server).
"""

from __future__ import annotations

import copy
import getpass
import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import yaml

from tabflow_slurm.setup.scheduler import SchedulerSetup
from tabflow_slurm.setup.sky_env import render_env_setup_script, stage_environment
from tabflow_slurm.setup.sky_storage import GcsStorage

if TYPE_CHECKING:
    from tabflow_slurm.setup.paths import PathSetup
    from tabflow_slurm.setup.resources import ResourcesSetup
    from tabflow_slurm.setup.sky_env import EnvSpec
    from tabflow_slurm.setup.sky_storage import Storage

#: The worker's cache directory on the VM (OpenML, HuggingFace, data-foundry and TabArena caches).
WORKER_CACHE_ROOT = "$HOME/tabarena_sky/cache"
#: The venv the job's setup builds on the VM (see ``sky_env.render_env_setup_script``).
WORKER_PYTHON = "$HOME/tabarena_sky/venv/bin/python"


class _LiteralDumper(yaml.SafeDumper):
    """``yaml.safe_dump`` with multi-line strings in literal block style (readable ``setup: |`` sections)."""


def _represent_multiline_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_LiteralDumper.add_representer(str, _represent_multiline_str)


def dump_yaml(data: dict) -> str:
    """Serialize a SkyPilot task or pool spec (key order kept, multi-line scripts as blocks)."""
    return yaml.dump(data, Dumper=_LiteralDumper, sort_keys=False, default_flow_style=False, width=1000)


@dataclass(frozen=True)
class SkyRunLayout:
    """Where one benchmark's artifacts live in the bucket."""

    run_uri: str
    """``<bucket>/<prefix>/runs/<benchmark_name>``."""

    @property
    def output_data_uri(self) -> str:
        """The results tree the workers copy into, mirrored to ``<workspace>/output/<benchmark>/data``."""
        return f"{self.run_uri}/output/data"

    def queue_uri(self, launch_id: str) -> str:
        return f"{self.run_uri}/queue/{launch_id}"

    def logs_uri(self, launch_id: str) -> str:
        return f"{self.run_uri}/logs/{launch_id}"


@dataclass(kw_only=True)
class SkyPilotSetup(SchedulerSetup):
    """Scheduler that runs the bundles as SkyPilot managed jobs draining a GCS claim queue.

    Compose it exactly like a ``SlurmSetup`` (``TabArenaBenchmarkPlan(scheduler_setup=SkyPilotSetup(...))``);
    ``PathSetup`` keeps naming the head node's workspace and venv. What differs is where the work
    runs and how results come back, see the module docstring. Two execution modes share one worker:

    * ``use_pool=False`` (default): ``sky jobs launch --num-jobs <workers>`` starts ``workers``
      managed jobs, each on its own spot VM that builds the venv in ``setup:`` and drains the queue
      until nothing is claimable. Recovery after a preemption reruns setup and resumes the job's
      own claims. No standing infrastructure besides SkyPilot's jobs controller.
    * ``use_pool=True``: ``sky jobs pool apply`` builds the venv once per pool worker; jobs only
      run the worker and reuse the workers' caches (datasets and weights persist across jobs),
      which suits many short bundles and BeyondArena. A pool bills while idle; the printed block
      ends with ``pool down``.
    """

    bucket: str = "gs://p2or-sky-cache-eu-dev"
    """The bucket every artifact of a run lives in (the launching account and SkyPilot's VM service
    account must both be able to write it)."""
    prefix: str | None = None
    """Path under the bucket; ``None`` means ``<login user>/tabarena``."""
    infra: str | None = None
    """SkyPilot ``infra`` of the workers. ``None`` (default) leaves the choice to the shared API
    server, whose admin policy expands a request across every region the VPC reaches and rejects an
    explicit one. Set it (e.g. ``"gcp/europe-west4"``) only against a server without that policy."""
    workers: int = 8
    """Worker jobs per launch (``--num-jobs``, capped at the number of bundles) and, in pool mode,
    the pool size. The analogue of SLURM's ``%N`` concurrency cap."""
    use_pool: bool = False
    """Run on a job pool instead of one VM per managed job (see the class docstring)."""
    pool_name: str | None = None
    """Pool name in pool mode; ``None`` means ``tabarena-gpu`` or ``tabarena-cpu`` by the run's resources."""
    gpu_accelerator: str = "RTXPRO6000:1"
    """SkyPilot accelerator spec for GPU runs. The default is the card of the SLURM GPU partition (a
    ``g4-standard-48``: 96 GB VRAM, 48 vCPU, 180 GB RAM, spot); ``sky gpus list`` names the others.
    Pair it with ``fake_memory_for_estimates`` set to that card's VRAM (96 for the default)."""
    cpu_cpus: str = "16+"
    """``resources.cpus`` for CPU runs (the SLURM CPU partition's 16 vCPU / 64 GB shape by default)."""
    cpu_memory: str = "64+"
    """``resources.memory`` (GB) for CPU runs."""
    cpu_instance_type: str | None = None
    """Pin a machine type for CPU runs instead of ``cpus``/``memory`` (comparable timings)."""
    use_spot: bool = True
    """Spot VMs; managed jobs recover from preemption."""
    disk_size: int = 128
    """Boot disk in GB: the venv with CUDA torch is about 15 GB, plus dataset caches."""
    item_time_limit_overhead: int = 3600
    """Seconds added to ``ResourcesSetup.time_limit_per_config`` for one item's wall-clock budget
    (the worker kills an item that exceeds it; SLURM's ``time_limit_overhead`` is hours per task)."""
    python_version: str = "3.12"
    """Python the worker venv is created with (match the head venv)."""
    secrets: tuple[str, ...] = ()
    """Environment variable names forwarded to the jobs as SkyPilot ``secrets`` (e.g. ``HF_TOKEN``
    for gated weights); their values are read from the launching shell by ``sky``."""
    requirements_extra_lines: tuple[str, ...] = ()
    """Lines appended to the staged ``requirements.txt`` (e.g. an index URL for another torch build)."""
    sky_binary: str | None = None
    """The ``sky`` executable for the printed commands; ``None`` means the cluster's ``sky`` on ``PATH``,
    else the one next to ``PathSetup.python_path`` (the ``skypilot`` extra)."""
    api_server_endpoint: str | None = None
    """The shared API server (``http://skypilot-api:46580`` on this cluster). When set, the printed
    block exports ``SKYPILOT_API_SERVER_ENDPOINT``; when ``None`` it requires the shell to provide it
    (login nodes preset it) and fails fast otherwise, so a launch never lands on a local server."""
    storage: Storage = field(default_factory=GcsStorage, compare=False, repr=False)
    """Object-store client; tests inject a directory backed one."""

    prefetches_weights_on_head: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError(f"workers must be at least 1, got {self.workers}.")
        self._synced: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------ naming
    @property
    def resolved_prefix(self) -> str:
        return self.prefix if self.prefix is not None else f"{getpass.getuser()}/tabarena"

    @property
    def root_uri(self) -> str:
        return f"{self.bucket.rstrip('/')}/{self.resolved_prefix.strip('/')}"

    def layout(self, benchmark_name: str) -> SkyRunLayout:
        return SkyRunLayout(run_uri=f"{self.root_uri}/runs/{benchmark_name}")

    def pool_for(self, resources: ResourcesSetup) -> str:
        if self.pool_name is not None:
            return self.pool_name
        return "tabarena-gpu" if resources.num_gpus > 0 else "tabarena-cpu"

    def sky_command(self, path_setup: PathSetup) -> str:
        """The ``sky`` executable: ``sky_binary``, else the cluster's ``sky`` on ``PATH``, else the run venv's."""
        if self.sky_binary is not None:
            return self.sky_binary
        return shutil.which("sky") or str(Path(path_setup.python_path).parent / "sky")

    def item_timeout_seconds(self, resources: ResourcesSetup) -> int:
        return int(resources.time_limit_per_config) + int(self.item_time_limit_overhead)

    def resources_block(self, resources: ResourcesSetup) -> dict:
        """The ``resources`` fields that pick a worker's hardware (without spot / disk)."""
        block: dict = {} if self.infra is None else {"infra": self.infra}
        if resources.num_gpus > 0:
            name, _, count = self.gpu_accelerator.partition(":")
            block["accelerators"] = {name: int(count or 1)}
        elif self.cpu_instance_type is not None:
            block["instance_type"] = self.cpu_instance_type
        else:
            block.update({"cpus": self.cpu_cpus, "memory": self.cpu_memory})
        return block

    def describe_target(self, resources: ResourcesSetup) -> str:
        block = self.resources_block(resources)
        hardware = ", ".join(f"{k}={v}" for k, v in block.items() if k != "infra")
        mode = f"pool {self.pool_for(resources)}" if self.use_pool else f"{self.workers} worker job(s)"
        return f"sky {self.infra or 'gcp'} ({hardware}, {'spot' if self.use_spot else 'on-demand'}, {mode})"

    def get_extra_default_args(self) -> dict:
        """No shared filesystem, so no shared-resources Ray setup (each VM is its own node)."""
        return {"setup_ray_for_slurm_shared_resources_environment": False}

    # ------------------------------------------------------------------ results sync
    def sync_results_to_local(self, *, path_setup: PathSetup, benchmark_name: str, force: bool = False) -> None:
        """Mirror the bucket's ``output/data`` of this benchmark into the workspace's output dir.

        Nothing is deleted locally, so results of a SLURM run under the same ``benchmark_name`` are
        kept and merged. A prefix that does not exist yet (first launch) is a no-op. Runs once per
        process for a given (remote, local) pair unless ``force``.
        """
        remote = self.layout(benchmark_name).output_data_uri
        local = Path(path_setup.get_output_path(benchmark_name)) / "data"
        key = (remote, str(local))
        if key in self._synced and not force:
            return
        self._synced.add(key)
        if not self.storage.exists_prefix(remote):
            print(f"No results under {remote} yet; nothing to sync.")
            return
        before = sum(1 for _ in local.rglob("results.pkl")) if local.exists() else 0
        self.storage.download_dir(remote, local)
        after = sum(1 for _ in local.rglob("results.pkl"))
        print(f"Synced {remote} into {local}: {before} -> {after} results.pkl")

    # ------------------------------------------------------------------ launch
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
        """Stage the environment, the batch and the task queue; return the ``sky`` command block.

        Everything is written under ``<setup_out>/<benchmark>/sky/<safe name>/`` first and then
        copied to the bucket. Because the worker budgets time per item, every bundle size shares one
        queue and one launch (SLURM needs one array per size only because ``--time`` is per task).
        Returns ``None`` when there is nothing to run.
        """
        all_jobs = jobs_dict["jobs"]
        if not all_jobs:
            if print_summary:
                print("No jobs to run.")
            return None

        env = stage_environment(
            python_path=path_setup.python_path,
            storage=self.storage,
            env_prefix=f"{self.root_uri}/env",
            repo_prefix=f"{self.root_uri}/repo",
            python_version=self.python_version,
            extra_requirement_lines=self.requirements_extra_lines,
        )
        layout = self.layout(benchmark_name)
        launch_id = f"{parallel_safe_benchmark_name}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
        sky_dir = path_setup.get_setup_out_path(benchmark_name) / "sky" / parallel_safe_benchmark_name
        queue_dir = sky_dir / "queue" / launch_id
        self._write_queue(
            queue_dir,
            jobs=all_jobs,
            defaults={**jobs_dict["defaults"], "item_timeout_seconds": self.item_timeout_seconds(resources_setup)},
            job_batch_dir=Path(jobs_dict["defaults"]["job_batch_dir"]),
        )
        queue_uri = layout.queue_uri(launch_id)
        self.storage.upload_dir(queue_dir, queue_uri)

        model_names = list(jobs_dict.get("model_names", []))
        job_yaml = sky_dir / "job.yaml"
        job_yaml.write_text(
            dump_yaml(
                self.render_job_spec(
                    resources=resources_setup,
                    env=env,
                    queue_uri=queue_uri,
                    run_uri=layout.run_uri,
                    launch_id=launch_id,
                    model_names=model_names,
                )
            )
        )
        pool_yaml = None
        if self.use_pool:
            pool_yaml = (
                path_setup.get_setup_out_path(benchmark_name) / "sky" / f"pool_{self.pool_for(resources_setup)}.yaml"
            )
            pool_yaml.write_text(dump_yaml(self.render_pool_spec(resources=resources_setup, env=env)))

        n_tasks = len(all_jobs)
        commands = self._command_block(
            path_setup=path_setup,
            resources=resources_setup,
            launch_id=launch_id,
            n_tasks=n_tasks,
            job_yaml=job_yaml,
            pool_yaml=pool_yaml,
            queue_uri=queue_uri,
        )
        (sky_dir / "launch.json").write_text(
            json.dumps(
                {
                    "launch_id": launch_id,
                    "queue_uri": queue_uri,
                    "run_uri": layout.run_uri,
                    "env_manifest": env.manifest_uri,
                    "n_tasks": n_tasks,
                    "n_items": sum(len(job["items"]) for job in all_jobs),
                    "commands": commands,
                },
                indent=2,
            )
        )
        if print_summary:
            print("##### Setup Jobs\nRun the following command(s) to start the jobs:\n" + "\n".join(commands) + "\n")
        return commands

    @staticmethod
    def _write_queue(queue_dir: Path, *, jobs: list[dict], defaults: dict, job_batch_dir: Path) -> None:
        """One self-contained task JSON per bundle plus a copy of the ``JobBatch`` directory."""
        tasks_dir = queue_dir / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        for idx, job in enumerate(jobs):
            (tasks_dir / f"{idx:06d}.json").write_text(json.dumps({"defaults": defaults, "items": job["items"]}))
        shutil.copytree(job_batch_dir, queue_dir / "job_batch", dirs_exist_ok=True)

    def render_job_spec(
        self,
        *,
        resources: ResourcesSetup,
        env: EnvSpec,
        queue_uri: str,
        run_uri: str,
        launch_id: str,
        model_names: list[str],
    ) -> dict:
        """The managed-job YAML (as a dict): resources, envs, secrets, setup (per-job mode) and run."""
        block = self.resources_block(resources)
        if not self.use_pool:
            block = {**block, "use_spot": self.use_spot, "disk_size": self.disk_size}
        spec: dict = {
            "name": launch_id,
            "resources": block,
            "envs": {
                "ENV_SPEC": env.manifest_uri,
                "CACHE_ROOT": WORKER_CACHE_ROOT,
                "QUEUE_URI": queue_uri,
                "RUN_URI": run_uri,
                "LAUNCH_ID": launch_id,
                "MODELS": ",".join(model_names),
            },
        }
        if self.secrets:
            spec["secrets"] = dict.fromkeys(self.secrets, "")
        if not self.use_pool:
            spec["setup"] = render_env_setup_script()
        spec["run"] = self.render_run_script()
        return spec

    def render_pool_spec(self, *, resources: ResourcesSetup, env: EnvSpec) -> dict:
        """The pool YAML (as a dict): the same hardware, ``workers`` and the venv build as ``setup``."""
        return {
            "name": self.pool_for(resources),
            "resources": {**self.resources_block(resources), "use_spot": self.use_spot, "disk_size": self.disk_size},
            "pool": {"workers": self.workers},
            "envs": {"ENV_SPEC": env.manifest_uri, "CACHE_ROOT": WORKER_CACHE_ROOT},
            "setup": render_env_setup_script(),
        }

    @staticmethod
    def render_run_script() -> str:
        """The ``run:`` section: start the worker from the venv the setup built."""
        return f'exec "{WORKER_PYTHON}" -P -m tabflow_slurm.sky_worker\n'

    def _command_block(
        self,
        *,
        path_setup: PathSetup,
        resources: ResourcesSetup,
        launch_id: str,
        n_tasks: int,
        job_yaml: Path,
        pool_yaml: Path | None,
        queue_uri: str,
    ) -> list[str]:
        sky = self.sky_command(path_setup)
        num_jobs = min(self.workers, n_tasks)
        lines = [f"# --- {launch_id}: {self.describe_target(resources)}, {n_tasks} bundle(s) ---"]
        if self.api_server_endpoint is not None:
            lines.append(f"export SKYPILOT_API_SERVER_ENDPOINT={self.api_server_endpoint}")
        else:
            lines.append(
                ': "${SKYPILOT_API_SERVER_ENDPOINT:?set it to the shared SkyPilot API server, '
                'e.g. http://skypilot-api:46580 (login nodes preset it)}"'
            )
        lines.append(f"{sky} check gcp    # once per machine; GCP must be enabled on the server")
        if pool_yaml is not None:
            pool = self.pool_for(resources)
            lines.append(
                f"{sky} jobs pool apply -y -p {pool} --workers {self.workers} {pool_yaml}    # idempotent; a new env rolls the workers"
            )
            lines.append(f"{sky} jobs pool status --all {pool}    # wait until every worker is READY")
            lines.append(f"{sky} jobs launch -y -d --pool {pool} -n {launch_id} --num-jobs {num_jobs} {job_yaml}")
        else:
            lines.append(f"{sky} jobs launch -y -d -n {launch_id} --num-jobs {num_jobs} {job_yaml}")
        lines.append(
            f"# progress: sky_progress.sh {queue_uri} {n_tasks}    (or: {sky} jobs queue; {sky} jobs logs <id>)"
        )
        if pool_yaml is not None:
            lines.append(
                f"# when done: run the eval (it syncs the results), then: {sky} jobs pool down -y {self.pool_for(resources)}    (a pool bills while idle)"
            )
        else:
            lines.append(
                f"# when done: run the eval (it syncs the results); stop early: {sky} jobs cancel -n {launch_id} -y"
            )
        return ["\n".join(lines)]


def load_task(path: Path) -> dict:
    """Read one task JSON written by :meth:`SkyPilotSetup._write_queue` (a deep copy, safe to edit)."""
    return copy.deepcopy(json.loads(Path(path).read_text()))
