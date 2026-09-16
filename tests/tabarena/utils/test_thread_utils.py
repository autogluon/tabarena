"""Tests for the CPU affinity helpers and the runner's CPU budget check."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from tabarena.benchmark.experiment.experiment_runner import ExperimentRunner
from tabarena.utils import thread_utils
from tabarena.utils.resources import detect_num_cpus
from tabarena.utils.thread_utils import (
    THREAD_ENV_VARS,
    align_thread_env_to_affinity,
    check_cpu_budget,
    cpu_thread_info,
    usable_cpu_count,
)


def test_usable_cpu_count_follows_the_affinity_mask_and_backs_detect_num_cpus():
    if hasattr(os, "sched_getaffinity"):
        assert usable_cpu_count() == len(os.sched_getaffinity(0))
    else:
        assert usable_cpu_count() == (os.cpu_count() or 1)
    assert detect_num_cpus() == usable_cpu_count()


@pytest.mark.parametrize("mode", ["error", "warn", "off"])
def test_check_cpu_budget_reports_a_mismatch_per_mode(monkeypatch, mode):
    monkeypatch.setattr(thread_utils, "usable_cpu_count", lambda: 16)
    expected = {"num_cpus": 8, "usable_cpus": 16, "matches": False}

    if mode == "error":
        with pytest.raises(RuntimeError, match=r"num_cpus=8.*16 CPUs.*cpuset.*num_cpus=16") as excinfo:
            check_cpu_budget(8, on_mismatch=mode)
        assert "--cpus-per-task=8" in str(excinfo.value)
    elif mode == "warn":
        with pytest.warns(RuntimeWarning, match="CPU budget mismatch"):
            assert check_cpu_budget(8, on_mismatch=mode) == expected
    else:
        assert check_cpu_budget(8, on_mismatch=mode) == expected
    # A matching budget is silent in every mode; ``None`` resolves to the detected count.
    assert check_cpu_budget(16, on_mismatch=mode) == {"num_cpus": 16, "usable_cpus": 16, "matches": True}
    assert check_cpu_budget(None, on_mismatch=mode)["matches"] is True


def test_align_thread_env_sets_only_the_unset_variables(monkeypatch):
    monkeypatch.setattr(thread_utils, "usable_cpu_count", lambda: 6)
    for name in THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "2")

    assert align_thread_env_to_affinity() == {"OMP_NUM_THREADS": "6", "OPENBLAS_NUM_THREADS": "6"}
    assert os.environ["MKL_NUM_THREADS"] == "2"
    assert align_thread_env_to_affinity() == {}


def test_cpu_thread_info_snapshots_without_importing_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(get_num_threads=lambda: 3))
    info = cpu_thread_info()
    assert info["torch_num_threads"] == 3
    assert info["usable_cpus"] == usable_cpu_count()
    assert set(info["thread_env"]) == set(THREAD_ENV_VARS)
    assert info["threadpools"] is None or isinstance(info["threadpools"], list)

    monkeypatch.delitem(sys.modules, "torch")
    assert cpu_thread_info()["torch_num_threads"] is None
    assert "torch" not in sys.modules


def test_runner_check_records_the_snapshot_and_honors_the_mode(monkeypatch):
    monkeypatch.setattr(thread_utils, "usable_cpu_count", lambda: 16)
    runner = object.__new__(ExperimentRunner)
    runner.model = SimpleNamespace(num_cpus_budget=8)

    runner.cpu_budget_check = "error"
    with pytest.raises(RuntimeError, match="CPU budget mismatch"):
        runner.run_cpu_budget_check()

    runner.cpu_budget_check = "off"
    info = runner.run_cpu_budget_check()
    assert info["budget_check"] == {"num_cpus": 8, "usable_cpus": 16, "matches": False}
    assert info["usable_cpus"] == 16

    runner.cpu_thread_info = info
    runner.method_cls = ExperimentRunner
    runner.time_warmup_s = None
    runner.warmup_report = None
    assert runner._experiment_metadata(time_start=0.0, time_start_str="")["cpu_thread_info"] is info
