from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("tabflow_slurm.slurm_utils", reason="tabflow_slurm is not installed")

from tabflow_slurm.slurm_utils import (
    MAX_SOCKET_PATH_BYTES,
    OFFLINE_WEIGHTS_ENV,
    apply_offline_weights_env,
    make_ray_temp_dir,
    plasma_directory_for,
    ray_init_kwargs,
    ray_object_store_bytes,
    ray_socket_path_fits,
)

GB = 1024**3


def test_plasma_directory_uses_shm_when_large_enough_and_disk_otherwise(capsys):
    # memory_limit 10 GB: object store 3 GB, /dev/shm must hold 5 GB (today's 0.5 rule).
    store = ray_object_store_bytes(10)
    assert plasma_directory_for(dev_shm_bytes=6 * GB, object_store_bytes=store, fallback_dir="/x/ray") is None
    assert plasma_directory_for(dev_shm_bytes=4 * GB, object_store_bytes=store, fallback_dir="/x/ray") == "/x/ray"
    assert "WARNING" in capsys.readouterr().out
    # An explicit threshold wins over the default rule.
    assert (
        plasma_directory_for(dev_shm_bytes=4 * GB, object_store_bytes=store, min_shm_bytes=GB, fallback_dir="/x")
        is None
    )


def test_ray_init_kwargs_has_no_runtime_env_and_logs_to_files():
    kwargs = ray_init_kwargs(num_cpus=8, num_gpus=1, memory_limit=32, ray_dir="/scratch/r", plasma_directory=None)
    assert "runtime_env" not in kwargs
    assert kwargs["log_to_driver"] is False
    assert kwargs["object_store_memory"] == int(32 * GB * 0.3)
    assert kwargs["_temp_dir"] == "/scratch/r"


def test_apply_offline_weights_env_sets_the_three_variables(monkeypatch):
    for name in OFFLINE_WEIGHTS_ENV:
        monkeypatch.delenv(name, raising=False)
    apply_offline_weights_env(False)
    assert all(name not in __import__("os").environ for name in OFFLINE_WEIGHTS_ENV)
    apply_offline_weights_env(True)
    import os

    assert {name: os.environ[name] for name in OFFLINE_WEIGHTS_ENV} == {
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "AG_FETCH_PRETRAINED_WEIGHTS": "false",
    }


def test_ray_temp_dir_fits_socket_limit_or_falls_back_to_tmp(tmp_path, capsys):
    # A short per-job root (what the template passes) keeps Ray's socket paths under the limit.
    short_root = tempfile.mkdtemp(prefix="tj")
    try:
        ray_dir = make_ray_temp_dir(short_root)
        assert ray_dir.startswith(short_root)
        assert ray_socket_path_fits(ray_dir)
    finally:
        shutil.rmtree(short_root, ignore_errors=True)

    # A root that is too long falls back to a fresh dir under /tmp with a warning.
    long_root = tmp_path / ("x" * max(1, MAX_SOCKET_PATH_BYTES - len(str(tmp_path))))
    fallback = make_ray_temp_dir(str(long_root))
    try:
        assert not ray_socket_path_fits(str(long_root) + "/rabcdefgh")
        assert Path(fallback).parent == Path("/tmp")  # noqa: S108
        assert Path(fallback).name.startswith("ray_")
        assert ray_socket_path_fits(fallback)
        assert "too long" in capsys.readouterr().out
    finally:
        shutil.rmtree(fallback, ignore_errors=True)
