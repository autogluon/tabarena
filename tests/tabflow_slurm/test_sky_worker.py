"""Tests for the SkyPilot worker (`tabflow_slurm.sky_worker`) against a directory-backed queue and a stub runner."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("tabflow_slurm.setup", reason="tabflow_slurm is not installed")

from tabflow_slurm.setup.skypilot import SkyPilotSetup
from tabflow_slurm.sky_worker import TIMEOUT_EXIT_CODE, Worker, WorkerConfig

_STUB_RUNNER = """
import argparse, json, os, sys, time
from pathlib import Path

parser = argparse.ArgumentParser()
for name in ("--job_batch_dir", "--experiment", "--dataset", "--fold", "--repeat", "--output_dir", "--num_cpus",
             "--num_gpus", "--memory_limit", "--setup_ray_for_slurm_shared_resources_environment", "--ignore_cache",
             "--cache_root", "--materialize_tasks", "--offline_weights", "--cpu_budget_check", "--require_warmup"):
    parser.add_argument(name)
args = parser.parse_args()
record_dir = Path(os.environ["STUB_RECORD_DIR"])
record_dir.mkdir(parents=True, exist_ok=True)
(record_dir / f"{args.experiment}.json").write_text(json.dumps({
    "argv": sys.argv[1:],
    "env": {k: os.environ.get(k) for k in ("TMPDIR", "HF_HOME", "TABARENA_CACHE", "OMP_NUM_THREADS", "TABARENA_RAY_LOG_DIR")},
}))
print("stub runner", args.experiment, flush=True)
if "slow" in args.experiment:
    time.sleep(30)
if "fail" in args.experiment:
    sys.exit(1)
out = Path(args.output_dir) / "data" / args.experiment / args.dataset / f"{args.repeat}_{args.fold}"
out.mkdir(parents=True)
(out / "results.pkl").write_bytes(b"result")
"""


def _defaults(item_timeout: int = 60) -> dict:
    return {
        "python": "/head/python",
        "run_script": "/head/run.py",
        "job_batch_dir": "/head/batch",
        "output_dir": "/head/out",
        "num_cpus": None,
        "num_gpus": 0,
        "memory_limit": None,
        "ignore_cache": False,
        "setup_ray_for_slurm_shared_resources_environment": False,
        "item_timeout_seconds": item_timeout,
    }


def _item(experiment: str, fold: int = 0) -> dict:
    return {"experiment": experiment, "dataset": "d", "fold": fold, "repeat": 0}


@pytest.fixture
def stub_runner(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "stub_runner.py"
    path.write_text(_STUB_RUNNER)
    monkeypatch.setenv("STUB_RECORD_DIR", str(tmp_path / "records"))
    return path


def _stage_queue(local_storage, tmp_path, bundles: list[list[dict]], *, item_timeout: int = 60) -> str:
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir(exist_ok=True)
    (batch_dir / "task_source.json").write_text('{"preset": "TabArena-v0.1"}')
    queue_dir = tmp_path / "queue"
    SkyPilotSetup._write_queue(
        queue_dir,
        jobs=[{"items": items} for items in bundles],
        defaults=_defaults(item_timeout),
        job_batch_dir=batch_dir,
    )
    queue_uri = "gs://b/me/runs/bench/queue/L"
    local_storage.upload_dir(queue_dir, queue_uri)
    return queue_uri


def _config(local_storage, tmp_path, stub_runner, queue_uri, *, worker_id="job-1", rank=0, num_jobs=1) -> WorkerConfig:
    return WorkerConfig(
        queue_uri=queue_uri,
        run_uri="gs://b/me/runs/bench",
        launch_id="L",
        cache_root=tmp_path / "cache",
        worker_id=worker_id,
        rank=rank,
        num_jobs=num_jobs,
        python=sys.executable,
        run_script=str(stub_runner),
        work_dir=tmp_path / "work",
        stagger_seconds=0,
        stop_ray_on_failure=False,
        storage=local_storage,
    )


def _records(tmp_path) -> dict[str, dict]:
    record_dir = tmp_path / "records"
    if not record_dir.exists():
        return {}
    return {p.stem: json.loads(p.read_text()) for p in record_dir.glob("*.json")}


class TestFromEnv:
    def test_reads_the_job_environment(self, local_storage, tmp_path):
        env = {
            "QUEUE_URI": "gs://b/q/",
            "RUN_URI": "gs://b/r",
            "LAUNCH_ID": "L",
            "CACHE_ROOT": str(tmp_path / "c"),
            "SKYPILOT_TASK_ID": "sky-2026-1",
            "SKYPILOT_JOB_RANK": "2",
            "SKYPILOT_NUM_JOBS": "8",
            "MODELS": "TabPFN-3,Linear",
            "SKY_WORKER_PYTHON": "/venv/bin/python",
            "SKY_WORKER_HOME": str(tmp_path / "w"),
            "SKY_WORKER_STAGGER_SECONDS": "0",
            "SKY_WORKER_STOP_RAY_ON_FAILURE": "false",
        }
        cfg = WorkerConfig.from_env(env, storage=local_storage)
        assert (cfg.queue_uri, cfg.run_uri, cfg.launch_id) == ("gs://b/q", "gs://b/r", "L")
        assert cfg.cache_root == tmp_path / "c"
        assert (cfg.worker_id, cfg.rank, cfg.num_jobs) == ("sky-2026-1", 2, 8)
        assert cfg.models == ("TabPFN-3", "Linear")
        assert cfg.python == "/venv/bin/python"
        assert cfg.work_dir == tmp_path / "w"
        assert cfg.stagger_seconds == 0
        assert cfg.stop_ray_on_failure is False
        assert cfg.run_script.endswith("run_tabarena_experiment.py")

    def test_missing_required_variables_raise(self):
        with pytest.raises(RuntimeError, match="SKYPILOT_TASK_ID"):
            WorkerConfig.from_env(
                {"QUEUE_URI": "gs://b/q", "RUN_URI": "gs://b/r", "LAUNCH_ID": "L", "CACHE_ROOT": "/c"}
            )


class TestWorkerRun:
    def test_claims_runs_uploads_and_marks(self, local_storage, tmp_path, stub_runner, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        queue_uri = _stage_queue(local_storage, tmp_path, [[_item("cfg_0")], [_item("cfg_1")]])
        cfg = _config(local_storage, tmp_path, stub_runner, queue_uri)
        assert Worker(cfg).run() == 0

        # Both bundles were claimed by this job and finished.
        assert local_storage.read_text(f"{queue_uri}/claims/000000") == "job-1"
        assert local_storage.read_text(f"{queue_uri}/claims/000001") == "job-1"
        assert sorted(local_storage.list_names(f"{queue_uri}/done")) == ["000000", "000000.0", "000001", "000001.0"]
        assert local_storage.read_text(f"{queue_uri}/done/000000.0").split()[:2] == ["ok", "0"]
        assert local_storage.list_names(f"{queue_uri}/failed") == []
        # Results landed under the run's output tree with the runner's own layout, logs next to them.
        assert local_storage.read_text("gs://b/me/runs/bench/output/data/cfg_0/d/0_0/results.pkl") == "result"
        assert local_storage.read_text("gs://b/me/runs/bench/output/data/cfg_1/d/0_0/results.pkl") == "result"
        assert "stub runner cfg_0" in local_storage.read_text("gs://b/me/runs/bench/logs/L/000000_0.log")

        # The runner saw this VM's paths, the self-sufficiency flags and a clean environment.
        record = _records(tmp_path)["cfg_0"]
        argv = record["argv"]
        assert argv[argv.index("--job_batch_dir") + 1] == str(tmp_path / "work" / "L" / "job_batch")
        assert argv[argv.index("--cache_root") + 1] == str(tmp_path / "cache")
        assert argv[argv.index("--materialize_tasks") + 1] == "True"
        assert argv[argv.index("--setup_ray_for_slurm_shared_resources_environment") + 1] == "False"
        assert argv[argv.index("--output_dir") + 1].startswith(str(tmp_path / "work" / "L" / "items"))
        assert record["env"]["HF_HOME"] == str(tmp_path / "cache" / "huggingface")
        assert record["env"]["TABARENA_CACHE"] == str(tmp_path / "cache" / "tabarena")
        assert record["env"]["TMPDIR"].startswith(str(tmp_path / "work" / "L" / "items"))
        assert record["env"]["OMP_NUM_THREADS"] is None
        # The batch copy was downloaded once.
        assert (tmp_path / "work" / "L" / "job_batch" / "task_source.json").exists()

    def test_done_bundles_and_foreign_claims_are_skipped(self, local_storage, tmp_path, stub_runner, capsys):
        queue_uri = _stage_queue(local_storage, tmp_path, [[_item("cfg_0")], [_item("cfg_1")]])
        local_storage.write_text(f"{queue_uri}/done/000000", "done elsewhere\n")
        local_storage.write_text(f"{queue_uri}/claims/000001", "job-9")
        Worker(_config(local_storage, tmp_path, stub_runner, queue_uri)).run()
        assert _records(tmp_path) == {}
        assert "remaining=1 claimable=0" in capsys.readouterr().out

    def test_recovery_resumes_an_own_claim_after_its_done_items(self, local_storage, tmp_path, stub_runner):
        queue_uri = _stage_queue(local_storage, tmp_path, [[_item("cfg_0", fold=0), _item("cfg_0", fold=1)]])
        local_storage.write_text(f"{queue_uri}/claims/000000", "job-1")
        local_storage.write_text(f"{queue_uri}/done/000000.0", "ok 0 1 cfg_0 d 0 0\n")
        Worker(_config(local_storage, tmp_path, stub_runner, queue_uri)).run()
        records = _records(tmp_path)
        # Only the second item ran (the first already had its marker) and the bundle was closed.
        assert list(records) == ["cfg_0"]
        assert records["cfg_0"]["argv"][records["cfg_0"]["argv"].index("--fold") + 1] == "1"
        assert "skipped (already done)" in local_storage.read_text(f"{queue_uri}/done/000000")
        assert local_storage.exists("gs://b/me/runs/bench/output/data/cfg_0/d/0_1/results.pkl")

    def test_timeout_marks_the_item_failed_and_uploads_no_results(self, local_storage, tmp_path, stub_runner):
        queue_uri = _stage_queue(local_storage, tmp_path, [[_item("slow_cfg")]], item_timeout=1)
        Worker(_config(local_storage, tmp_path, stub_runner, queue_uri)).run()
        marker = local_storage.read_text(f"{queue_uri}/done/000000.0").split()
        assert marker[:2] == ["timeout", str(TIMEOUT_EXIT_CODE)]
        assert local_storage.list_names(f"{queue_uri}/failed") == ["000000.0"]
        assert not local_storage.exists_prefix("gs://b/me/runs/bench/output/data")
        assert "killed by sky_worker" in local_storage.read_text("gs://b/me/runs/bench/logs/L/000000_0.log")

    def test_a_failing_item_does_not_stop_its_bundle_mates(self, local_storage, tmp_path, stub_runner):
        queue_uri = _stage_queue(local_storage, tmp_path, [[_item("fail_cfg"), _item("ok_cfg")]])
        Worker(_config(local_storage, tmp_path, stub_runner, queue_uri)).run()
        assert local_storage.read_text(f"{queue_uri}/done/000000.0").split()[:2] == ["fail:1", "1"]
        assert local_storage.read_text(f"{queue_uri}/done/000000.1").split()[:2] == ["ok", "0"]
        assert local_storage.list_names(f"{queue_uri}/failed") == ["000000.0"]
        assert not local_storage.exists("gs://b/me/runs/bench/output/data/fail_cfg/d/0_0/results.pkl")
        assert local_storage.exists("gs://b/me/runs/bench/output/data/ok_cfg/d/0_0/results.pkl")
        summary = local_storage.read_text(f"{queue_uri}/done/000000")
        assert "000000.0 fail:1" in summary and "000000.1 ok" in summary

    def test_rank_rotation_starts_a_worker_at_its_own_shard(self, local_storage, tmp_path, stub_runner):
        queue_uri = _stage_queue(local_storage, tmp_path, [[_item(f"cfg_{i}")] for i in range(4)])
        Worker(_config(local_storage, tmp_path, stub_runner, queue_uri, worker_id="job-2", rank=1, num_jobs=2)).run()
        claims = [u for u in local_storage.uploads if "/claims/" in u]
        assert claims[0].endswith("/claims/000002")
        assert len(claims) == 4  # it went on to take the rest once its shard was done

    def test_two_workers_never_fit_the_same_bundle(self, local_storage, tmp_path, stub_runner):
        queue_uri = _stage_queue(local_storage, tmp_path, [[_item(f"cfg_{i}")] for i in range(3)])
        Worker(_config(local_storage, tmp_path, stub_runner, queue_uri, worker_id="job-1", rank=0, num_jobs=2)).run()
        Worker(_config(local_storage, tmp_path, stub_runner, queue_uri, worker_id="job-2", rank=1, num_jobs=2)).run()
        owners = {idx: local_storage.read_text(f"{queue_uri}/claims/{idx}") for idx in ("000000", "000001", "000002")}
        assert set(owners.values()) == {"job-1"}  # the first worker drained everything; the second found nothing
        assert len(_records(tmp_path)) == 3


def test_stub_runner_env_isolation_leaves_the_parent_untouched(local_storage, tmp_path, stub_runner):
    queue_uri = _stage_queue(local_storage, tmp_path, [[_item("cfg_0")]])
    before = dict(os.environ)
    Worker(_config(local_storage, tmp_path, stub_runner, queue_uri)).run()
    assert {k: v for k, v in os.environ.items() if k != "STUB_RECORD_DIR"} == {
        k: v for k, v in before.items() if k != "STUB_RECORD_DIR"
    }
