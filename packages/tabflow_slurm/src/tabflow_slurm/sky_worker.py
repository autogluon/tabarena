"""Drain a GCS claim queue of benchmark bundles on a SkyPilot worker VM.

The queue is what :class:`tabflow_slurm.setup.skypilot.SkyPilotSetup` staged: one task JSON per
bundle under ``<QUEUE_URI>/tasks/``, the ``JobBatch`` under ``<QUEUE_URI>/job_batch/``. Every
worker job of a launch runs this module and shares the queue through three marker prefixes:

* ``claims/<idx>`` holds the id of the job that owns bundle ``idx``. A worker claims with an
  exclusive create, so two workers never fit the same bundle.
* ``done/<idx>.<k>`` records item ``k`` of bundle ``idx`` (``<status> <rc> <seconds> <coords>``),
  ``done/<idx>`` the finished bundle. ``failed/<idx>.<k>`` duplicates the non-ok item records so
  progress needs only listings.
* A recovered job first resumes the bundles it owns, skipping items that already have a marker,
  then claims new ones. Its ``SKYPILOT_TASK_ID`` gets a new launch timestamp on every recovery
  while the ``<job name>_<job id>-<task id>`` tail stays, so a claim is matched on that tail
  (:func:`same_job`). When nothing is claimable the worker exits; orphaned claims of a cancelled
  job are re-enumerated by the next ``setup``.

Before an item runs, the dataset's cache entries listed in the queue's ``cache_manifest.json`` (seeded
by the setup, see :mod:`tabflow_slurm.setup.sky_cache`) are pulled into ``CACHE_ROOT``, so the runner
finds the task cached and contacts neither OpenML nor the Hub; a dataset without entries is
downloaded by the runner as a fallback. Each item runs the bundled runner in its own process, built
with the same argument list the local runner uses (``run_local._build_item_command``) plus ``--cache_root`` and ``--materialize_tasks``,
under a per-item wall-clock budget, in a fresh scratch directory, with the caches under
``CACHE_ROOT``. A successful item's ``data/`` tree is copied into ``<RUN_URI>/output/data``; a
failed or timed-out item uploads nothing but its log. Configuration comes from the environment the
job YAML sets (see :class:`WorkerConfig.from_env`).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from tabflow_slurm.run_local import _build_item_command
from tabflow_slurm.setup.paths import get_run_script_path
from tabflow_slurm.setup.sky_cache import weight_cache_env
from tabflow_slurm.setup.sky_storage import GcsStorage

if TYPE_CHECKING:
    from tabflow_slurm.setup.sky_storage import Storage

#: Thread-pool variables a login shell may carry; the models size their pools themselves.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
)
#: Exit code recorded for an item the worker killed at its wall-clock budget.
TIMEOUT_EXIT_CODE = 124
#: Failed claims in a row before the done/claims listings are refreshed.
CLAIM_REFRESH_EVERY = 5


def same_job(owner: str, worker_id: str) -> bool:
    """Whether the claim written by ``owner`` belongs to the managed job running as ``worker_id``.

    A managed job's ``SKYPILOT_TASK_ID`` reads ``sky-managed-<timestamp>_<job name>_<job id>-<task id>``.
    SkyPilot stamps a new timestamp when it recovers the job after a preemption, so the id of the
    recovered worker differs from the one it wrote into its claims before; everything after the first
    underscore is the same job. Ids without that prefix (tests, other schedulers) compare whole.
    """
    if owner == worker_id:
        return True
    return "_" in owner and "_" in worker_id and owner.split("_", 1)[1] == worker_id.split("_", 1)[1]


@dataclass
class WorkerConfig:
    """What one worker job needs to know; :meth:`from_env` reads it from the job's environment."""

    queue_uri: str
    run_uri: str
    launch_id: str
    cache_root: Path
    worker_id: str
    """The managed job's ``SKYPILOT_TASK_ID``, written into claims; a recovery changes its timestamp
    prefix, so claims are compared with :func:`same_job`."""
    rank: int = 0
    num_jobs: int = 1
    models: tuple[str, ...] = ()
    """Registry model names whose weights are prefetched once at start (``MODELS`` env)."""
    python: str = sys.executable
    run_script: str = field(default_factory=lambda: str(get_run_script_path()))
    work_dir: Path = field(default_factory=lambda: Path.home() / "tabarena_sky" / "work")
    scratch_root: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "tabarena_sky")
    """Parent of the per-item scratch directories (``<scratch_root>/<bundle>_<item>/{tmp,ag}``), kept
    apart from ``work_dir`` and short on purpose: an item's ``TMPDIR`` hosts the Unix sockets of
    multiprocessing managers (EBM's fit starts one) whose paths are limited to 108 bytes, and the
    launch directory under ``work_dir`` carries the launch id."""
    stagger_seconds: float = 3.0
    """Delay of ``rank * stagger_seconds`` before the first download, so fresh workers do not hit
    OpenML or the Hub at the same instant."""
    stop_ray_on_failure: bool = True
    """Run ``ray stop --force`` after a failed or killed item (a SIGKILLed runner can leave raylets
    behind on the VM). Off in tests, which share a machine with real Ray sessions."""
    storage: Storage = field(default_factory=GcsStorage, repr=False)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, *, storage: Storage | None = None) -> WorkerConfig:
        """Read ``QUEUE_URI``, ``RUN_URI``, ``LAUNCH_ID``, ``CACHE_ROOT``, ``SKYPILOT_TASK_ID`` (required),
        ``SKYPILOT_JOB_RANK`` / ``SKYPILOT_NUM_JOBS`` / ``MODELS`` (optional) and the test overrides
        ``SKY_WORKER_PYTHON``, ``SKY_WORKER_RUN_SCRIPT``, ``SKY_WORKER_HOME``, ``SKY_WORKER_SCRATCH_ROOT``,
        ``SKY_WORKER_STAGGER_SECONDS``.
        """
        env = os.environ if environ is None else environ
        missing = [
            name
            for name in ("QUEUE_URI", "RUN_URI", "LAUNCH_ID", "CACHE_ROOT", "SKYPILOT_TASK_ID")
            if not env.get(name)
        ]
        if missing:
            raise RuntimeError(f"sky_worker needs the environment variables {missing}.")
        kwargs: dict = {
            "queue_uri": env["QUEUE_URI"].rstrip("/"),
            "run_uri": env["RUN_URI"].rstrip("/"),
            "launch_id": env["LAUNCH_ID"],
            "cache_root": Path(os.path.expandvars(env["CACHE_ROOT"])).expanduser(),
            "worker_id": env["SKYPILOT_TASK_ID"],
            "rank": int(env.get("SKYPILOT_JOB_RANK", "0") or 0),
            "num_jobs": max(1, int(env.get("SKYPILOT_NUM_JOBS", "1") or 1)),
            "models": tuple(m for m in env.get("MODELS", "").split(",") if m),
        }
        if env.get("SKY_WORKER_PYTHON"):
            kwargs["python"] = env["SKY_WORKER_PYTHON"]
        if env.get("SKY_WORKER_RUN_SCRIPT"):
            kwargs["run_script"] = env["SKY_WORKER_RUN_SCRIPT"]
        if env.get("SKY_WORKER_HOME"):
            kwargs["work_dir"] = Path(env["SKY_WORKER_HOME"])
        if env.get("SKY_WORKER_SCRATCH_ROOT"):
            kwargs["scratch_root"] = Path(env["SKY_WORKER_SCRATCH_ROOT"])
        if env.get("SKY_WORKER_STAGGER_SECONDS"):
            kwargs["stagger_seconds"] = float(env["SKY_WORKER_STAGGER_SECONDS"])
        if env.get("SKY_WORKER_STOP_RAY_ON_FAILURE"):
            kwargs["stop_ray_on_failure"] = env["SKY_WORKER_STOP_RAY_ON_FAILURE"].lower() in ("1", "true", "yes")
        if storage is not None:
            kwargs["storage"] = storage
        return cls(**kwargs)


def _log(message: str) -> None:
    print(f"[sky_worker {time.strftime('%H:%M:%S')}] {message}", flush=True)


class Worker:
    """One worker job: prepare the VM, resume own claims, then claim and run bundles until none is left."""

    def __init__(self, config: WorkerConfig):
        self.cfg = config
        self.storage = config.storage
        self.launch_dir = config.work_dir / config.launch_id
        self.job_batch_dir = self.launch_dir / "job_batch"
        self.claims_uri = f"{config.queue_uri}/claims"
        self.done_uri = f"{config.queue_uri}/done"
        self.failed_uri = f"{config.queue_uri}/failed"
        self.tasks_uri = f"{config.queue_uri}/tasks"
        self.cache_manifest: dict | None = None
        self._pulled: set[str] = set()

    # ------------------------------------------------------------------ queue views
    def list_tasks(self) -> list[str]:
        names = [n[: -len(".json")] for n in self.storage.list_names(self.tasks_uri) if n.endswith(".json")]
        if not names:
            raise RuntimeError(f"No tasks under {self.tasks_uri}.")
        return sorted(names)

    def list_done_bundles(self) -> set[str]:
        return {n for n in self.storage.list_names(self.done_uri) if "." not in n}

    def list_done_items(self, idx: str) -> set[str]:
        return {n for n in self.storage.list_names(self.done_uri) if n.startswith(f"{idx}.")}

    def list_claims(self) -> set[str]:
        return set(self.storage.list_names(self.claims_uri))

    # ------------------------------------------------------------------ lifecycle
    def run(self) -> int:
        """The worker's main sequence; returns the process exit code (0, the queue is shared work)."""
        cfg = self.cfg
        _log(f"worker {cfg.worker_id} rank {cfg.rank}/{cfg.num_jobs} on queue {cfg.queue_uri}")
        if cfg.rank and cfg.stagger_seconds:
            time.sleep(cfg.rank * cfg.stagger_seconds)
        self.prepare()
        self.recover_own_claims()
        self.claim_loop()
        return 0

    def prepare(self) -> None:
        """Create the work and cache directories, download the batch, prefetch the models' weights."""
        for sub in ("huggingface", "tabarena", "data_foundry", "openml", "jit"):
            (self.cfg.cache_root / sub).mkdir(parents=True, exist_ok=True)
        self.launch_dir.mkdir(parents=True, exist_ok=True)
        if not self.job_batch_dir.exists():
            self.storage.download_dir(f"{self.cfg.queue_uri}/job_batch", self.job_batch_dir)
        manifest_uri = f"{self.cfg.queue_uri}/cache_manifest.json"
        if self.storage.exists(manifest_uri):
            self.cache_manifest = json.loads(self.storage.read_text(manifest_uri))
            n_datasets = len(self.cache_manifest.get("datasets", {}))
            _log(f"dataset cache {self.cache_manifest.get('cache_uri')} covers {n_datasets} dataset(s)")
        weights = (self.cache_manifest or {}).get("weights", {})
        if weights:
            self.pull_entries([rel for rels in weights.values() for rel in rels])
        # Models whose weights are not in the seeded cache are prefetched from the Hub as a fallback.
        unseeded = [m for m in self.cfg.models if not weights.get(m)]
        if unseeded and not (self.cache_manifest or {}).get("offline_weights"):
            self.prefetch_weights(unseeded)

    def prefetch_weights(self, models: list[str] | None = None) -> None:
        """Warm the given models' weights into the worker's cache from the Hub (best effort, once per job start)."""
        models = list(self.cfg.models if models is None else models)
        code = (
            f"from tabarena.models.prefetch import prefetch_weights; prefetch_weights({models!r}, raise_on_error=False)"
        )
        _log(f"prefetching weights for {', '.join(models)}")
        result = subprocess.run(  # noqa: S603
            [self.cfg.python, "-P", "-c", code], env=self.item_env(self.launch_dir / "prefetch"), check=False
        )
        if result.returncode != 0:
            _log(f"WARNING: weight prefetch exited {result.returncode}; the fits download what they need")

    def recover_own_claims(self) -> None:
        """Finish the bundles this job claimed before a recovery (their items may be half done)."""
        done = self.list_done_bundles()
        for idx in sorted(self.list_claims() - done):
            try:
                owner = self.storage.read_text(f"{self.claims_uri}/{idx}").strip()
            except Exception as exc:
                _log(f"could not read claim {idx}: {exc!r}")
                continue
            if same_job(owner, self.cfg.worker_id):
                _log(f"resuming own claim {idx} (claimed as {owner})")
                self.run_bundle(idx)

    def claim_loop(self) -> None:
        """Claim unowned bundles (rotated by rank so simultaneous starters spread out) until none is left."""
        tasks = self.list_tasks()
        start = (self.cfg.rank * math.ceil(len(tasks) / self.cfg.num_jobs)) % len(tasks)
        order = tasks[start:] + tasks[:start]
        done = self.list_done_bundles()
        claimed = self.list_claims()
        lost = 0
        for idx in order:
            if idx in done or idx in claimed:
                continue
            if self.storage.create_exclusive(f"{self.claims_uri}/{idx}", self.cfg.worker_id):
                lost = 0
                self.run_bundle(idx)
                continue
            lost += 1
            if lost >= CLAIM_REFRESH_EVERY:
                done = self.list_done_bundles()
                claimed = self.list_claims()
                lost = 0
        remaining = len(set(tasks) - self.list_done_bundles())
        _log(f"nothing left to claim: remaining={remaining} claimable=0; exiting")

    # ------------------------------------------------------------------ bundles and items
    def run_bundle(self, idx: str) -> None:
        task = json.loads(self.storage.read_text(f"{self.tasks_uri}/{idx}.json"))
        already = self.list_done_items(idx)
        records: list[str] = []
        for k, item in enumerate(task["items"]):
            marker = f"{idx}.{k}"
            if marker in already:
                records.append(f"{marker} skipped (already done)")
                continue
            status, rc, seconds = self.run_item(idx, k, task["defaults"], item)
            record = f"{status} {rc} {seconds} {item['experiment']} {item['dataset']} {item['fold']} {item['repeat']}"
            self.storage.write_text(f"{self.done_uri}/{marker}", record + "\n")
            if status != "ok":
                self.storage.write_text(f"{self.failed_uri}/{marker}", record + "\n")
            records.append(f"{marker} {record}")
        self.storage.write_text(f"{self.done_uri}/{idx}", "\n".join(records) + "\n")
        _log(f"bundle {idx} done ({len(task['items'])} item(s))")

    def pull_dataset_cache(self, dataset: str) -> None:
        """Copy the dataset's seeded cache entries into ``CACHE_ROOT`` (once per worker process)."""
        if not self.cache_manifest:
            return
        self.pull_entries(self.cache_manifest.get("datasets", {}).get(dataset, []))

    def pull_entries(self, rels: list[str]) -> None:
        """Copy the given cache entries (relative to the cache layout) from the bucket into ``CACHE_ROOT``."""
        base = str((self.cache_manifest or {}).get("cache_uri", "")).rstrip("/")
        for rel in rels:
            if rel in self._pulled:
                continue
            local = self.cfg.cache_root / rel
            try:
                if Path(rel).suffix and not rel.endswith(("/snapshots", "/refs")):
                    if not local.exists():
                        self.storage.download_file(f"{base}/{rel}", local)
                else:
                    self.storage.download_dir(f"{base}/{rel}", local)
            except Exception as exc:
                _log(f"WARNING: could not pull {rel} from the cache ({exc!r}); the runner downloads it")
                continue
            self._pulled.add(rel)

    def item_env(self, scratch: Path) -> dict[str, str]:
        """The environment of one item process: local caches, per-item scratch, no inherited thread pools."""
        cache = self.cfg.cache_root
        env = {k: v for k, v in os.environ.items() if k not in THREAD_ENV_VARS}
        env.update(weight_cache_env(cache))  # HF_HOME, XDG_CACHE_HOME (tabpfn), TORCH_HOME under CACHE_ROOT
        if (self.cache_manifest or {}).get("offline_weights"):
            env["HF_HUB_OFFLINE"] = "1"  # every model's weights were pulled from the seeded cache
        env.update(
            {
                "TMPDIR": str(scratch / "tmp"),
                "TABARENA_MODEL_ARTIFACTS_BASE_PATH": str(scratch / "ag"),
                "TABARENA_CACHE": str(cache / "tabarena"),
                "DATA_FOUNDRY_CACHE": str(cache / "data_foundry"),
                "NUMBA_CACHE_DIR": str(cache / "jit" / "numba"),
                "TRITON_CACHE_DIR": str(cache / "jit" / "triton"),
                "CUDA_CACHE_PATH": str(cache / "jit" / "nv"),
                "TABPFN_DISABLE_TELEMETRY": "1",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        env.pop("HF_HUB_CACHE", None)
        env.pop("HUGGINGFACE_HUB_CACHE", None)
        for sub in ("tmp", "ag"):
            (scratch / sub).mkdir(parents=True, exist_ok=True)
        return env

    def item_command(self, defaults: dict, item: dict, output_dir: Path) -> list[str]:
        """The runner argv: the local runner's list with this VM's paths, plus the self-sufficiency flags."""
        local_defaults = {
            **defaults,
            "python": self.cfg.python,
            "run_script": self.cfg.run_script,
            "job_batch_dir": str(self.job_batch_dir),
            "output_dir": str(output_dir),
        }
        return [
            *_build_item_command(local_defaults, item),
            "--cache_root",
            str(self.cfg.cache_root),
            "--materialize_tasks",
            "True",
        ]

    def run_item(self, idx: str, k: int, defaults: dict, item: dict) -> tuple[str, int, int]:
        """Fit one item under its budget; upload its results (ok) or only its log; return ``(status, rc, seconds)``."""
        item_dir = self.launch_dir / "items" / f"{idx}_{k}"
        shutil.rmtree(item_dir, ignore_errors=True)
        output_dir = item_dir / "out"
        scratch = self.cfg.scratch_root / f"{idx}_{k}"  # short: see WorkerConfig.scratch_root
        shutil.rmtree(scratch, ignore_errors=True)
        ray_logs = item_dir / "ray_logs"
        output_dir.mkdir(parents=True)
        env = self.item_env(scratch)
        env["TABARENA_RAY_LOG_DIR"] = str(ray_logs)
        log_path = item_dir / "item.log"
        timeout = int(defaults.get("item_timeout_seconds", 0)) or None
        coords = (
            f"experiment={item['experiment']} dataset={item['dataset']} fold={item['fold']} repeat={item['repeat']}"
        )
        _log(f"item {idx}.{k} start: {coords} (budget {timeout}s)")
        self.pull_dataset_cache(item["dataset"])
        started = time.monotonic()
        rc = self._run_process(
            self.item_command(defaults, item, output_dir), env=env, log_path=log_path, timeout=timeout
        )
        seconds = int(time.monotonic() - started)
        status = "ok" if rc == 0 else ("timeout" if rc == TIMEOUT_EXIT_CODE else f"fail:{rc}")
        if rc == 0:
            self._upload_with_retries(output_dir / "data", f"{self.cfg.run_uri}/output/data")
        else:
            shutil.rmtree(output_dir, ignore_errors=True)  # never ship a possibly truncated results.pkl
            if self.cfg.stop_ray_on_failure:
                subprocess.run(  # noqa: S603
                    [self.cfg.python, "-m", "ray", "stop", "--force"], check=False, capture_output=True
                )
        logs_uri = f"{self.cfg.run_uri}/logs/{self.cfg.launch_id}"
        try:
            self.storage.upload_file(log_path, f"{logs_uri}/{idx}_{k}.log")
            if ray_logs.is_dir() and any(ray_logs.rglob("*")):
                self.storage.copy_tree(ray_logs, f"{logs_uri}/{idx}_{k}_ray")
        except Exception as exc:
            _log(f"WARNING: could not upload the log of item {idx}.{k}: {exc!r}")
        _log(f"item {idx}.{k} {status} in {seconds}s")
        shutil.rmtree(scratch, ignore_errors=True)
        return status, rc, seconds

    @staticmethod
    def _run_process(argv: list[str], *, env: dict[str, str], log_path: Path, timeout: int | None) -> int:
        """Run ``argv`` in its own process group, tee-less (stdout and stderr into ``log_path``); kill at ``timeout``."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("wb") as log:
            log.write((" ".join(argv) + "\n\n").encode())
            log.flush()
            proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)  # noqa: S603
            try:
                return proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                log.write(f"\n##### killed by sky_worker after {timeout}s\n".encode())
                for sig, grace in ((signal.SIGTERM, 60), (signal.SIGKILL, None)):
                    try:
                        os.killpg(proc.pid, sig)
                        proc.wait(timeout=grace)
                        break
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        continue
                return TIMEOUT_EXIT_CODE

    def _upload_with_retries(self, local: Path, uri: str, attempts: int = 3) -> None:
        if not local.is_dir():
            _log(f"WARNING: {local} has no results to upload")
            return
        for attempt in range(1, attempts + 1):
            try:
                self.storage.copy_tree(local, uri)
                return
            except Exception as exc:
                _log(f"upload attempt {attempt}/{attempts} failed: {exc!r}")
                if attempt == attempts:
                    raise
                time.sleep(10 * attempt)


def main() -> int:
    """Entry point of ``python -P -m tabflow_slurm.sky_worker``."""
    return Worker(WorkerConfig.from_env()).run()


if __name__ == "__main__":
    sys.exit(main())
