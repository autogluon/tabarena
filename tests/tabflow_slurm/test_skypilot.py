"""Tests for the SkyPilot scheduler (`tabflow_slurm.setup.skypilot`), with the bucket replaced by a directory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("tabflow_slurm.setup", reason="tabflow_slurm is not installed")

from tabflow_slurm.setup import skypilot as sky_mod
from tabflow_slurm.setup.paths import PathSetup
from tabflow_slurm.setup.resources import ResourcesSetup
from tabflow_slurm.setup.sky_env import EnvSpec
from tabflow_slurm.setup.skypilot import SkyPilotSetup, dump_yaml


def _resources(**kw) -> ResourcesSetup:
    kw.setdefault("time_limit", 3600)
    kw.setdefault("num_cpus", 8)
    kw.setdefault("num_gpus", 0)
    kw.setdefault("memory_limit", 32)
    return ResourcesSetup(**kw)


def _env() -> EnvSpec:
    return EnvSpec(
        python_version="3.12",
        requirements_uri="gs://b/me/tabarena/env/abc/requirements.txt",
        repos=(),
        manifest_uri="gs://b/me/tabarena/env/abc/env.json",
        env_hash="abc",
    )


def _setup(local_storage, **kw) -> SkyPilotSetup:
    kw.setdefault("bucket", "gs://b")
    kw.setdefault("prefix", "me/tabarena")
    return SkyPilotSetup(storage=local_storage, **kw)


def _jobs_dict(batch_dir: Path, n_bundles: int = 3) -> dict:
    return {
        "defaults": {
            "python": "/head/venv/bin/python",
            "run_script": "/head/run.py",
            "job_batch_dir": str(batch_dir),
            "output_dir": "/head/ws/output/bench",
            "num_cpus": None,
            "num_gpus": 0,
            "memory_limit": None,
            "ignore_cache": False,
            "setup_ray_for_slurm_shared_resources_environment": False,
        },
        "jobs": [
            {"bundle_size": 2, "items": [{"experiment": f"cfg_{i}", "dataset": "d", "fold": 0, "repeat": 0}]}
            for i in range(n_bundles)
        ],
        "max_configs_per_job": 2,
        "model_names": ["TabPFN-3"],
    }


@pytest.fixture
def batch_dir(tmp_path) -> Path:
    path = tmp_path / "job_batch"
    path.mkdir()
    (path / "experiments.yaml").write_text("[]\n")
    (path / "task_source.json").write_text('{"preset": "TabArena-v0.1"}')
    return path


@pytest.fixture
def staged_env(monkeypatch):
    monkeypatch.setattr(sky_mod, "stage_environment", lambda **_: _env())


class TestConfiguration:
    def test_workers_must_be_positive(self, local_storage):
        with pytest.raises(ValueError, match="workers"):
            _setup(local_storage, workers=0)

    def test_layout_and_prefix_default_to_the_login_user(self, local_storage, monkeypatch):
        monkeypatch.setattr(sky_mod.getpass, "getuser", lambda: "me")
        setup = SkyPilotSetup(storage=local_storage, bucket="gs://b/")
        assert setup.root_uri == "gs://b/me/tabarena"
        layout = setup.layout("bench")
        assert layout.run_uri == "gs://b/me/tabarena/runs/bench"
        assert layout.output_data_uri == "gs://b/me/tabarena/runs/bench/output/data"
        assert layout.queue_uri("L") == "gs://b/me/tabarena/runs/bench/queue/L"

    def test_resources_block_for_gpu_cpu_and_pinned_instance(self, local_storage):
        setup = _setup(local_storage)
        assert setup.resources_block(_resources(num_gpus=1)) == {
            "infra": "gcp/europe-west4",
            "accelerators": {"A100-80GB": 1},
        }
        assert setup.resources_block(_resources()) == {"infra": "gcp/europe-west4", "cpus": "16+", "memory": "64+"}
        pinned = _setup(local_storage, cpu_instance_type="n2-standard-16", infra="gcp")
        assert pinned.resources_block(_resources()) == {"infra": "gcp", "instance_type": "n2-standard-16"}

    def test_describe_target_and_pool_names(self, local_storage):
        setup = _setup(local_storage, workers=4)
        assert setup.describe_target(_resources(num_gpus=1)) == (
            "sky gcp/europe-west4 (accelerators={'A100-80GB': 1}, spot, 4 worker job(s))"
        )
        pool = _setup(local_storage, use_pool=True)
        assert pool.pool_for(_resources(num_gpus=1)) == "tabarena-gpu"
        assert pool.pool_for(_resources()) == "tabarena-cpu"
        assert "pool tabarena-cpu" in pool.describe_target(_resources())
        assert _setup(local_storage, use_pool=True, pool_name="mine").pool_for(_resources()) == "mine"

    def test_extra_default_args_disable_the_shared_filesystem_ray_setup(self, local_storage):
        assert _setup(local_storage).get_extra_default_args() == {
            "setup_ray_for_slurm_shared_resources_environment": False
        }

    def test_item_timeout_adds_the_overhead_to_the_per_config_budget(self, local_storage):
        setup = _setup(local_storage, item_time_limit_overhead=600)
        assert setup.item_timeout_seconds(_resources(time_limit=3600)) == 4200

    def test_sky_binary_defaults_to_the_run_venv(self, local_storage):
        ps = PathSetup(workspace="/ws", python_path="/venv/bin/python")
        assert _setup(local_storage).sky_command(ps) == "/venv/bin/sky"
        assert _setup(local_storage, sky_binary="/opt/sky").sky_command(ps) == "/opt/sky"


class TestRenderSpecs:
    def test_per_job_mode_builds_the_venv_in_setup(self, local_storage):
        setup = _setup(local_storage, secrets=("HF_TOKEN",), workers=3)
        spec = yaml.safe_load(
            dump_yaml(
                setup.render_job_spec(
                    resources=_resources(num_gpus=1),
                    env=_env(),
                    queue_uri="gs://b/q",
                    run_uri="gs://b/r",
                    launch_id="bench_gpu-20260916-101010",
                    model_names=["TabPFN-3"],
                )
            )
        )
        assert spec["resources"] == {
            "infra": "gcp/europe-west4",
            "accelerators": {"A100-80GB": 1},
            "use_spot": True,
            "disk_size": 128,
        }
        for forbidden in ("cloud", "region", "zone"):
            assert forbidden not in spec["resources"]
        assert spec["envs"] == {
            "ENV_SPEC": "gs://b/me/tabarena/env/abc/env.json",
            "CACHE_ROOT": "$HOME/tabarena_sky/cache",
            "QUEUE_URI": "gs://b/q",
            "RUN_URI": "gs://b/r",
            "LAUNCH_ID": "bench_gpu-20260916-101010",
            "MODELS": "TabPFN-3",
        }
        assert spec["secrets"] == {"HF_TOKEN": ""}
        assert "uv venv --python" in spec["setup"]
        assert "tabflow_slurm.sky_worker" in spec["run"]
        assert "workdir" not in spec and "file_mounts" not in spec

    def test_pool_mode_moves_the_venv_build_to_the_pool(self, local_storage):
        setup = _setup(local_storage, use_pool=True, workers=5)
        job = setup.render_job_spec(
            resources=_resources(), env=_env(), queue_uri="gs://b/q", run_uri="gs://b/r", launch_id="L", model_names=[]
        )
        pool = yaml.safe_load(dump_yaml(setup.render_pool_spec(resources=_resources(), env=_env())))
        assert "setup" not in job
        assert job["resources"] == {"infra": "gcp/europe-west4", "cpus": "16+", "memory": "64+"}
        assert pool["name"] == "tabarena-cpu"
        assert pool["pool"] == {"workers": 5}
        assert pool["resources"]["use_spot"] is True and pool["resources"]["disk_size"] == 128
        assert pool["envs"] == {
            "ENV_SPEC": "gs://b/me/tabarena/env/abc/env.json",
            "CACHE_ROOT": "$HOME/tabarena_sky/cache",
        }
        assert "uv venv --python" in pool["setup"]

    def test_multiline_scripts_dump_as_literal_blocks(self):
        text = dump_yaml({"setup": "a\nb\n", "run": "c"})
        assert "setup: |" in text
        assert "run: c" in text


class TestGetRunCommands:
    def _ps(self, tmp_path) -> PathSetup:
        return PathSetup(workspace=tmp_path / "ws", python_path="/venv/bin/python")

    def test_no_jobs_returns_none(self, local_storage, tmp_path, batch_dir):
        jobs = _jobs_dict(batch_dir)
        jobs["jobs"] = []
        setup = _setup(local_storage)
        assert (
            setup.get_run_commands(
                jobs_dict=jobs,
                path_setup=self._ps(tmp_path),
                benchmark_name="bench",
                parallel_safe_benchmark_name="bench_cpu",
                resources_setup=_resources(),
                print_summary=False,
            )
            is None
        )
        assert local_storage.uploads == []

    def test_stages_the_queue_and_prints_one_launch(self, local_storage, tmp_path, batch_dir, staged_env, monkeypatch):
        ps = self._ps(tmp_path)
        ps.ensure_runtime_dirs("bench")
        setup = _setup(local_storage, workers=8, item_time_limit_overhead=600)
        commands = setup.get_run_commands(
            jobs_dict=_jobs_dict(batch_dir, n_bundles=3),
            path_setup=ps,
            benchmark_name="bench",
            parallel_safe_benchmark_name="bench_cpu",
            resources_setup=_resources(time_limit=3600),
            print_summary=False,
        )
        assert len(commands) == 1
        block = commands[0]
        launch = json.loads((ps.get_setup_out_path("bench") / "sky" / "bench_cpu" / "launch.json").read_text())
        launch_id = launch["launch_id"]
        assert launch_id.startswith("bench_cpu-")
        assert launch["n_tasks"] == 3 and launch["n_items"] == 3
        # One task JSON per bundle, self-contained, with the per-item budget and no shared-FS Ray setup.
        queue_uri = launch["queue_uri"]
        assert queue_uri == f"gs://b/me/tabarena/runs/bench/queue/{launch_id}"
        tasks = sorted(local_storage.list_names(f"{queue_uri}/tasks"))
        assert tasks == ["000000.json", "000001.json", "000002.json"]
        task = json.loads(local_storage.read_text(f"{queue_uri}/tasks/000001.json"))
        assert task["items"] == [{"experiment": "cfg_1", "dataset": "d", "fold": 0, "repeat": 0}]
        assert task["defaults"]["item_timeout_seconds"] == 4200
        assert task["defaults"]["setup_ray_for_slurm_shared_resources_environment"] is False
        assert "bundle_size" not in task
        # The batch copy (with its recorded preset) travels with the launch.
        assert local_storage.read_text(f"{queue_uri}/job_batch/task_source.json") == '{"preset": "TabArena-v0.1"}'
        # The command block: check, one launch capped at the bundle count, no pool.
        assert "/venv/bin/sky check gcp" in block
        assert f"/venv/bin/sky jobs launch -y -d -n {launch_id} --num-jobs 3 " in block
        assert "pool" not in block
        assert "sky_progress.sh" in block
        job_spec = yaml.safe_load((ps.get_setup_out_path("bench") / "sky" / "bench_cpu" / "job.yaml").read_text())
        assert job_spec["envs"]["QUEUE_URI"] == queue_uri
        assert job_spec["envs"]["MODELS"] == "TabPFN-3"
        assert "setup" in job_spec

    def test_pool_mode_prints_apply_status_launch_and_down(self, local_storage, tmp_path, batch_dir, staged_env):
        ps = self._ps(tmp_path)
        ps.ensure_runtime_dirs("bench")
        setup = _setup(local_storage, workers=2, use_pool=True, sky_binary="sky")
        block = setup.get_run_commands(
            jobs_dict=_jobs_dict(batch_dir, n_bundles=5),
            path_setup=ps,
            benchmark_name="bench",
            parallel_safe_benchmark_name="bench_gpu",
            resources_setup=_resources(num_gpus=1),
            print_summary=False,
        )[0]
        pool_yaml = ps.get_setup_out_path("bench") / "sky" / "pool_tabarena-gpu.yaml"
        assert pool_yaml.exists()
        assert f"sky jobs pool apply -y -p tabarena-gpu --workers 2 {pool_yaml}" in block
        assert "sky jobs pool status --all tabarena-gpu" in block
        assert "sky jobs launch -y -d --pool tabarena-gpu -n bench_gpu-" in block
        assert "--num-jobs 2 " in block
        assert "sky jobs pool down -y tabarena-gpu" in block
        job_spec = yaml.safe_load((ps.get_setup_out_path("bench") / "sky" / "bench_gpu" / "job.yaml").read_text())
        assert "setup" not in job_spec


class TestSyncResultsToLocal:
    def _remote_result(self, local_storage, name="a"):
        local_storage.write_text(f"gs://b/me/tabarena/runs/bench/output/data/m/k/0_0/{name}.pkl", name)

    def test_downloads_new_results_and_keeps_local_ones(self, local_storage, tmp_path, capsys):
        ps = PathSetup(workspace=tmp_path / "ws", python_path="/venv/bin/python")
        local_data = Path(ps.get_output_path("bench")) / "data"
        (local_data / "m" / "k2" / "0_0").mkdir(parents=True)
        (local_data / "m" / "k2" / "0_0" / "results.pkl").write_bytes(b"slurm")
        local_storage.write_text("gs://b/me/tabarena/runs/bench/output/data/m/k/0_0/results.pkl", "sky")
        _setup(local_storage).sync_results_to_local(path_setup=ps, benchmark_name="bench")
        assert (local_data / "m" / "k" / "0_0" / "results.pkl").read_text() == "sky"
        assert (local_data / "m" / "k2" / "0_0" / "results.pkl").read_bytes() == b"slurm"
        assert "1 -> 2 results.pkl" in capsys.readouterr().out

    def test_missing_prefix_is_a_noop(self, local_storage, tmp_path):
        ps = PathSetup(workspace=tmp_path / "ws", python_path="/venv/bin/python")
        _setup(local_storage).sync_results_to_local(path_setup=ps, benchmark_name="bench")
        assert not (Path(ps.get_output_path("bench")) / "data").exists()

    def test_memoized_per_process_unless_forced(self, local_storage, tmp_path):
        ps = PathSetup(workspace=tmp_path / "ws", python_path="/venv/bin/python")
        setup = _setup(local_storage)
        local_storage.write_text("gs://b/me/tabarena/runs/bench/output/data/m/k/0_0/results.pkl", "v1")
        setup.sync_results_to_local(path_setup=ps, benchmark_name="bench")
        local_storage.write_text("gs://b/me/tabarena/runs/bench/output/data/m/k/0_0/results.pkl", "v2")
        target = Path(ps.get_output_path("bench")) / "data" / "m" / "k" / "0_0" / "results.pkl"
        setup.sync_results_to_local(path_setup=ps, benchmark_name="bench")
        assert target.read_text() == "v1"
        setup.sync_results_to_local(path_setup=ps, benchmark_name="bench", force=True)
        assert target.read_text() == "v2"
