"""Tests for the lazy Ray helpers with a fake Ray module; the real ``ray`` is never imported or started."""

from __future__ import annotations

import asyncio
import inspect
import logging
import subprocess
import sys

import pytest

from tabarena.utils import ray_utils


class _FakeRef:
    def __init__(self, value):
        self.value = value


class _FakeMethod:
    def __init__(self, bound):
        self._bound = bound

    def remote(self, *args, **kwargs):
        result = self._bound(*args, **kwargs)
        if inspect.iscoroutine(result):
            result = asyncio.run(result)
        return _FakeRef(result)


class _FakeActorHandle:
    def __init__(self, instance):
        self._instance = instance

    def __getattr__(self, name):
        return _FakeMethod(getattr(self._instance, name))


class _FakeRemote:
    """What ``ray.remote(**opts)(target)`` returns: tasks run eagerly, classes become actor handles."""

    def __init__(self, target, options):
        self.target = target
        self.options = options

    def remote(self, *args, **kwargs):
        if inspect.isclass(self.target):
            return _FakeActorHandle(self.target(*args, **kwargs))
        return _FakeRef(self.target(*args, **kwargs))


class _FakeRay:
    class exceptions:
        class GetTimeoutError(Exception):
            pass

    def __init__(self, *, initialized: bool = False):
        self._initialized = initialized
        self.init_calls: list[dict] = []
        self.remote_options: list[dict] = []
        self.killed: list = []

    def is_initialized(self):
        return self._initialized

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        self._initialized = True

    def remote(self, **options):
        self.remote_options.append(options)
        return lambda target: _FakeRemote(target, options)

    def get(self, refs, timeout=None):
        if isinstance(refs, list):
            return [ref.value for ref in refs]
        return refs.value

    def wait(self, refs, num_returns=1, timeout=None):
        return list(refs), []

    def kill(self, actor):
        self.killed.append(actor)


@pytest.fixture
def fake_ray(monkeypatch):
    ray = _FakeRay()
    monkeypatch.setattr(ray_utils, "try_import_ray", lambda: ray)
    monkeypatch.delenv(ray_utils.DISABLE_RAY_WARMUP_ENV, raising=False)
    return ray


def test_module_import_does_not_import_ray():
    code = "import sys; import tabarena.utils.ray_utils; print('ray' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout  # noqa: S603
    assert out.strip() == "False"


def test_flags_read_environment(monkeypatch):
    monkeypatch.setenv(ray_utils.DISABLE_RAY_WARMUP_ENV, "1")
    monkeypatch.setenv(ray_utils.RAY_WORKER_WARMUP_ENV, "true")
    assert ray_utils.ray_warmup_disabled() is True
    assert ray_utils.ray_worker_warmup_enabled() is True
    monkeypatch.setenv(ray_utils.DISABLE_RAY_WARMUP_ENV, "0")
    monkeypatch.delenv(ray_utils.RAY_WORKER_WARMUP_ENV)
    assert ray_utils.ray_warmup_disabled() is False
    assert ray_utils.ray_worker_warmup_enabled() is False


def test_ensure_ray_initialized_mirrors_autogluon_init_args(fake_ray):
    out = ray_utils.ensure_ray_initialized(num_cpus=4, num_gpus=1)
    assert fake_ray.init_calls == [
        {
            "log_to_driver": False,
            "logging_level": logging.ERROR,
            "include_dashboard": False,
            "num_cpus": 4,
            "num_gpus": 1,
        }
    ]
    assert out["initialized_by_warmup"] is True
    assert out["already_initialized"] is False
    assert out["init_args"] == fake_ray.init_calls[0]


def test_ensure_ray_initialized_omits_num_gpus_when_zero(fake_ray):
    ray_utils.ensure_ray_initialized(num_cpus=8, num_gpus=0)
    assert "num_gpus" not in fake_ray.init_calls[0]
    assert fake_ray.init_calls[0]["num_cpus"] == 8


def test_ensure_ray_initialized_noop_when_running_or_unknown_cpus(fake_ray):
    fake_ray._initialized = True
    out = ray_utils.ensure_ray_initialized(num_cpus=4, num_gpus=0)
    assert out["already_initialized"] is True and fake_ray.init_calls == []

    fake_ray._initialized = False
    out = ray_utils.ensure_ray_initialized(num_cpus=None, num_gpus=0)
    assert "skipped" in out and fake_ray.init_calls == []


def test_kill_switch_prevents_init_and_pool(fake_ray, monkeypatch):
    monkeypatch.setenv(ray_utils.DISABLE_RAY_WARMUP_ENV, "1")
    out = ray_utils.ensure_ray_initialized(num_cpus=4, num_gpus=0)
    assert out["skipped"] == ray_utils.DISABLE_RAY_WARMUP_ENV
    pool = ray_utils.warmup_ray_workers(["colorsys"], 2)
    assert pool == {"requested": 2, "skipped": ray_utils.DISABLE_RAY_WARMUP_ENV}
    assert fake_ray.init_calls == [] and fake_ray.remote_options == []


@pytest.mark.parametrize(
    ("num_cpus", "num_jobs", "expected"),
    [(8, 8, (8, 1)), (16, 8, (8, 2)), (4, 8, (4, 1)), (3, 2, (2, 1)), (0, 8, (0, 1)), (8, 0, (0, 1))],
)
def test_plan_ray_worker_pool(num_cpus, num_jobs, expected):
    assert ray_utils.plan_ray_worker_pool(num_cpus=num_cpus, num_jobs=num_jobs) == expected


def test_warmup_ray_workers_imports_and_reports(fake_ray, monkeypatch):
    fake_ray._initialized = True
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    sys.modules.pop("wave", None)
    out = ray_utils.warmup_ray_workers(
        ["wave", "__main__", "wave", "this_module_does_not_exist_xyz"], 2, cpus_per_worker=2, timeout_s=0.05
    )
    assert out["requested"] == 2 and out["cpus_per_worker"] == 2
    assert out["modules"] == ["wave", "this_module_does_not_exist_xyz"]
    assert "wave" in sys.modules
    assert out["distinct_pids"] == 1 and len(out["pids"]) == 2  # eager fake: every task runs in this process
    assert set(out["failed_imports"]) == {"this_module_does_not_exist_xyz"}
    assert out["elapsed_s"] >= 0
    # The barrier actor is created with num_cpus=0 and killed afterwards; tasks carry the worker CPUs.
    assert {"num_cpus": 0} in fake_ray.remote_options
    assert {"num_cpus": 2, "max_retries": 0} in fake_ray.remote_options
    assert len(fake_ray.killed) == 1
    # The warm task removed the thread count Ray would have set for it.
    assert "OMP_NUM_THREADS" not in __import__("os").environ


def test_warmup_ray_workers_skips_without_ray_running(fake_ray):
    assert ray_utils.warmup_ray_workers(["wave"], 2)["skipped"] == "ray is not initialized"
    assert ray_utils.warmup_ray_workers(["wave"], 0)["skipped"] == "no workers requested"
