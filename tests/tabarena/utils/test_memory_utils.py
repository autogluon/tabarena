from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from tabarena.utils import memory_utils
from tabarena.utils.memory_utils import (
    CpuMemoryTracker,
    GpuMemoryTracker,
    _procfs_children_supported,
    _read_rss_procfs,
    _walk_descendants_procfs,
)

linux_only = pytest.mark.skipif(sys.platform != "linux", reason="procfs is Linux only")
procfs_only = pytest.mark.skipif(
    not _procfs_children_supported(os.getpid()),
    reason="kernel has no /proc/<pid>/task/<tid>/children (CONFIG_PROC_CHILDREN)",
)

# The two spawned interpreters each hold at least this much RSS, so a total that includes them
# must exceed the main process alone by at least this.
_IDLE_INTERPRETER_RSS = 5 * 1024 * 1024


def _rss_close(a: int, b: int) -> bool:
    """RSS drifts between two reads; accept max(2 percent, 8 MB) of difference."""
    return abs(a - b) <= max(0.02 * max(a, b), 8 * 1024 * 1024)


# ----------------------------------------------------------------------------------------------
# Existing behavior: sampling and the Event-based stop.
# ----------------------------------------------------------------------------------------------


def test_cpu_tracker_stops_without_waiting_out_the_interval():
    """Leaving the context returns at once; before, `__exit__` waited for the sampler's sleep."""
    tracker = CpuMemoryTracker(interval=2.0)
    with tracker:
        time.sleep(0.05)
        start = time.perf_counter()
    assert time.perf_counter() - start < 0.5
    assert tracker.min_rss is not None and tracker.peak_rss >= tracker.min_rss > 0
    assert not tracker._sampler_thread.is_alive()


def test_cpu_tracker_samples_the_end_state():
    """The exit takes a final sample, so a change right before leaving is seen."""
    tracker = CpuMemoryTracker(interval=60.0)
    with tracker:
        ballast = bytearray(64 * 1024 * 1024)  # allocated after the only periodic sample
        ballast[::4096] = b"x" * len(ballast[::4096])
    assert tracker.peak_rss > tracker.min_rss
    del ballast


def test_cpu_tracker_can_be_reused():
    tracker = CpuMemoryTracker(interval=0.01)
    for _ in range(2):
        with tracker:
            time.sleep(0.03)
        assert not tracker._sampler_thread.is_alive()


def test_gpu_tracker_stops_without_waiting_out_the_interval():
    tracker = GpuMemoryTracker(interval=2.0)
    if not tracker.enabled:
        pytest.skip("CUDA is not available")
    with tracker:
        time.sleep(0.05)
        start = time.perf_counter()
    assert time.perf_counter() - start < 0.5
    assert tracker.peak_allocated is not None


# ----------------------------------------------------------------------------------------------
# Backend selection and the new baseline attributes.
# ----------------------------------------------------------------------------------------------


def test_start_rss_baseline_recorded():
    tracker = CpuMemoryTracker(interval=0.01)
    assert tracker.start_rss is None and tracker.start_rss_self is None
    with tracker:
        time.sleep(0.02)
    assert tracker.backend in {"procfs", "psutil"}
    assert tracker.start_rss >= tracker.start_rss_self > 0
    assert tracker.min_rss <= tracker.start_rss <= tracker.peak_rss


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        CpuMemoryTracker(backend="sysfs")


@procfs_only
def test_auto_picks_procfs_where_supported():
    assert CpuMemoryTracker().backend == "procfs"
    assert CpuMemoryTracker(backend="psutil").backend == "psutil"


def test_backend_falls_back_to_psutil_when_children_file_missing(monkeypatch):
    monkeypatch.setattr(memory_utils, "_procfs_children_supported", lambda pid: False)
    tracker = CpuMemoryTracker(interval=0.01)
    assert tracker.backend == "psutil"
    with tracker:
        time.sleep(0.02)
    assert tracker.min_rss > 0 and tracker.peak_rss >= tracker.min_rss
    assert tracker.start_rss >= tracker.start_rss_self > 0
    with pytest.raises(ValueError, match="procfs"):
        CpuMemoryTracker(backend="procfs")


def test_backend_falls_back_to_psutil_when_page_size_unavailable(monkeypatch):
    def broken_sysconf(name):
        raise ValueError(name)

    monkeypatch.setattr(memory_utils.os, "sysconf", broken_sysconf)
    assert CpuMemoryTracker().backend == "psutil"
    with pytest.raises(ValueError, match="procfs"):
        CpuMemoryTracker(backend="procfs")


def test_backend_falls_back_to_psutil_when_procfs_root_is_empty(monkeypatch, tmp_path):
    """The support probe reads the real files, so an empty proc root disables procfs."""
    monkeypatch.setattr(memory_utils, "_PROCFS_ROOT", str(tmp_path))
    assert CpuMemoryTracker().backend == "psutil"
    with pytest.raises(ValueError, match="procfs"):
        CpuMemoryTracker(backend="procfs")


def test_psutil_backend_lists_descendants_with_psutil(monkeypatch):
    tracker = CpuMemoryTracker(backend="psutil")
    fake_children = [SimpleNamespace(pid=11), SimpleNamespace(pid=12)]
    monkeypatch.setattr(tracker._proc, "children", lambda recursive: fake_children)
    assert tracker._descendant_pids() == [11, 12]

    def failing_children(recursive):
        raise psutil.Error("boom")

    monkeypatch.setattr(tracker._proc, "children", failing_children)
    assert tracker._descendant_pids() == []


def test_include_children_false_skips_the_walk(monkeypatch):
    tracker = CpuMemoryTracker(include_children=False)

    def unexpected_walk(pid):
        raise AssertionError("descendants must not be walked when include_children is False")

    monkeypatch.setattr(memory_utils, "_walk_descendants_procfs", unexpected_walk)
    monkeypatch.setattr(tracker._proc, "children", lambda recursive: unexpected_walk(0))
    assert tracker._descendant_pids() == []
    assert tracker._get_current_rss() > 0


# ----------------------------------------------------------------------------------------------
# The procfs helpers on the real proc filesystem.
# ----------------------------------------------------------------------------------------------


@linux_only
def test_statm_rss_equals_psutil_rss():
    page_size = os.sysconf("SC_PAGE_SIZE")
    proc = psutil.Process()
    reads = []
    for _ in range(2):
        reads.append((_read_rss_procfs(os.getpid(), page_size), proc.memory_info().rss))
    for statm_rss, psutil_rss in reads:
        assert statm_rss is not None and statm_rss > 0
        assert statm_rss % page_size == 0
        # The self RSS may move between the two reads; anything within a few dozen pages is the same number.
        assert abs(statm_rss - psutil_rss) < 64 * page_size


@linux_only
def test_read_rss_procfs_returns_none_for_a_missing_process():
    # pid_max is at most 2**22 on Linux, so this pid never exists.
    assert _read_rss_procfs(2**23 + 7, os.sysconf("SC_PAGE_SIZE")) is None


@pytest.fixture
def child_tree():
    """A child process that itself spawns a grandchild; both sleep until torn down.

    Yields the set of the two pids once psutil sees both of them.
    """
    # psutil.Popen is a subprocess.Popen that also exposes the psutil.Process API (children, kill).
    child = psutil.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time\n"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "time.sleep(60)\n",
        ],
    )
    grandchildren: list[psutil.Process] = []
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            grandchildren = child.children()
            if grandchildren:
                break
            time.sleep(0.02)
        else:
            pytest.fail("grandchild did not appear within 10 s")
        yield {child.pid, *(p.pid for p in grandchildren)}
    finally:
        for p in grandchildren:
            try:
                p.kill()
            except psutil.Error:
                pass
        child.kill()
        child.wait(timeout=10)
        for p in grandchildren:
            try:
                p.wait(timeout=10)
            except psutil.Error:
                pass


@procfs_only
def test_procfs_walk_matches_psutil_descendants(child_tree):
    walked = set(_walk_descendants_procfs(os.getpid()))
    via_psutil = {p.pid for p in psutil.Process().children(recursive=True)}
    assert child_tree <= walked
    assert walked == via_psutil
    walked_list = _walk_descendants_procfs(os.getpid())
    assert len(walked_list) == len(set(walked_list)), "a pid was listed twice"


@procfs_only
def test_procfs_and_psutil_backends_sum_the_same_rss(child_tree):
    procfs = CpuMemoryTracker(backend="procfs")
    via_psutil = CpuMemoryTracker(backend="psutil")
    assert set(procfs._descendant_pids()) >= child_tree
    assert set(via_psutil._descendant_pids()) >= child_tree

    total_procfs = procfs._get_current_rss()
    total_psutil = via_psutil._get_current_rss()
    assert _rss_close(total_procfs, total_psutil), (total_procfs, total_psutil)

    # Each total includes the two idle interpreters, so it exceeds the main process alone.
    assert total_procfs - procfs._rss_self() >= 2 * _IDLE_INTERPRETER_RSS
    assert total_psutil - via_psutil._rss_self() >= 2 * _IDLE_INTERPRETER_RSS
    assert _rss_close(procfs._rss_self(), via_psutil._rss_self())


@procfs_only
def test_procfs_tracker_counts_descendants_present_at_enter(child_tree):
    tracker = CpuMemoryTracker(interval=0.01, backend="procfs")
    with tracker:
        time.sleep(0.05)
    assert tracker.start_rss - tracker.start_rss_self >= 2 * _IDLE_INTERPRETER_RSS
    assert tracker.peak_rss >= tracker.start_rss


@procfs_only
def test_no_children_fast_path(monkeypatch):
    if psutil.Process().children():
        pytest.skip("the test process already has children")
    tracker = CpuMemoryTracker(backend="procfs")
    assert tracker._descendant_pids() == []

    calls: list[int] = []
    real_read = memory_utils._read_rss_procfs

    def counting_read(pid, page_size):
        calls.append(pid)
        return real_read(pid, page_size)

    monkeypatch.setattr(memory_utils, "_read_rss_procfs", counting_read)
    total = tracker._get_current_rss()
    assert calls == [os.getpid()], "only the process's own statm is read when it has no children"
    assert abs(total - tracker._rss_self()) < 64 * os.sysconf("SC_PAGE_SIZE")


# ----------------------------------------------------------------------------------------------
# The procfs helpers on a fake proc tree (deterministic walk, races, garbage).
# ----------------------------------------------------------------------------------------------


def _write(path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def fake_proc(tmp_path, monkeypatch):
    """A fake /proc with pid 100 (two threads) and descendants 200, 300, 400, 500.

    400 has already exited (no task dir, no statm). 500 lists 200 again (a stale entry the seen set
    must absorb) plus a garbage token. Resident pages per pid: 100 has 10, 200 has 20, 300 has 30, 500 has 50.
    """
    root = tmp_path / "proc"
    _write(root / "100" / "task" / "100" / "children", "200 300\n")
    _write(root / "100" / "task" / "101" / "children", "400")
    _write(root / "100" / "statm", "99 10 5 1 0 3 0\n")
    _write(root / "200" / "task" / "200" / "children", "500")
    _write(root / "200" / "statm", "99 20 5 1 0 3 0\n")
    _write(root / "300" / "task" / "300" / "children", "")
    _write(root / "300" / "statm", "99 30 5 1 0 3 0\n")
    _write(root / "500" / "task" / "500" / "children", "200 abc")
    _write(root / "500" / "statm", "99 50 5 1 0 3 0\n")
    # A process whose children file vanished mid-walk: the task dir exists, the file does not.
    (root / "600" / "task" / "600").mkdir(parents=True)
    _write(root / "300" / "task" / "300" / "children", "600")
    _write(root / "600" / "statm", "garbage")
    monkeypatch.setattr(memory_utils, "_PROCFS_ROOT", str(root))
    return root


def test_walk_descendants_on_fake_tree(fake_proc):
    walked = _walk_descendants_procfs(100)
    assert sorted(walked) == [200, 300, 400, 500, 600]
    assert len(walked) == len(set(walked))
    # 400 is a child of the second thread only: the per-task union is what finds it.
    assert 400 in walked


def test_read_rss_on_fake_tree(fake_proc):
    assert _read_rss_procfs(100, 4096) == 10 * 4096
    assert _read_rss_procfs(400, 4096) is None  # exited
    assert _read_rss_procfs(600, 4096) is None  # unparsable statm


def test_get_current_rss_sums_the_fake_tree(fake_proc):
    tracker = CpuMemoryTracker(backend="psutil")  # the real support probe fails against the fake root
    tracker.backend = "procfs"
    tracker._pid = 100
    tracker._page_size = 4096
    assert tracker._rss_self() == 10 * 4096
    assert tracker._get_current_rss() == (10 + 20 + 30 + 50) * 4096


def test_rss_self_falls_back_to_psutil_when_statm_is_missing(fake_proc, monkeypatch):
    tracker = CpuMemoryTracker(backend="psutil")
    tracker.backend = "procfs"
    tracker._pid = 999  # absent from the fake tree
    tracker._page_size = 4096
    monkeypatch.setattr(tracker._proc, "memory_info", lambda: SimpleNamespace(rss=4242))
    assert tracker._rss_self() == 4242


def test_walk_of_an_unknown_root_is_empty(fake_proc):
    assert _walk_descendants_procfs(12345) == []


# ----------------------------------------------------------------------------------------------
# GpuMemoryTracker sampling with a fake torch (no CUDA needed).
# ----------------------------------------------------------------------------------------------


def _fake_gpu_tracker(cuda_ns: SimpleNamespace, *, enabled: bool = False, interval: float = 60.0) -> GpuMemoryTracker:
    """Build a tracker without running `__init__` (which imports torch) and attach a fake torch."""
    import threading

    tracker = GpuMemoryTracker.__new__(GpuMemoryTracker)
    tracker.interval = interval
    tracker._torch = SimpleNamespace(cuda=cuda_ns)
    tracker.enabled = enabled
    tracker.device = 0
    tracker._stop = threading.Event()
    tracker._sampler_thread = None
    tracker.min_allocated = tracker.peak_allocated = None
    tracker.min_reserved = tracker.peak_reserved = None
    return tracker


def _raise(*args, **kwargs):
    raise AssertionError("fallback must not be called")


def test_gpu_sample_reads_nested_stats_once():
    calls = []

    def nested(device):
        calls.append(device)
        return {"allocated_bytes": {"all": {"current": 5}}, "reserved_bytes": {"all": {"current": 7}}}

    tracker = _fake_gpu_tracker(
        SimpleNamespace(memory_stats_as_nested_dict=nested, memory_allocated=_raise, memory_reserved=_raise)
    )
    assert tracker._sample_gpu_memory() == (5, 7)
    assert calls == [0]


@pytest.mark.parametrize(
    "nested_result",
    [
        {"unexpected": {}},  # missing keys
        {},  # allocator not initialized on this device
        None,  # older torch returning nothing useful
        RuntimeError("no nested stats"),
    ],
)
def test_gpu_sample_falls_back_to_current_calls(nested_result):
    def nested(device):
        if isinstance(nested_result, Exception):
            raise nested_result
        return nested_result

    tracker = _fake_gpu_tracker(
        SimpleNamespace(memory_stats_as_nested_dict=nested, memory_allocated=lambda d: 11, memory_reserved=lambda d: 13)
    )
    assert tracker._sample_gpu_memory() == (11, 13)


def test_gpu_sample_without_nested_api_uses_current_calls():
    tracker = _fake_gpu_tracker(SimpleNamespace(memory_allocated=lambda d: 1, memory_reserved=lambda d: 2))
    assert tracker._sample_gpu_memory() == (1, 2)


def test_gpu_exit_reconciles_peaks_with_torch_counters():
    """The context keeps synchronizing, resetting peaks and raising its peaks to torch's max counters."""
    events = []
    cuda = SimpleNamespace(
        memory_stats_as_nested_dict=lambda d: {
            "allocated_bytes": {"all": {"current": 5}},
            "reserved_bytes": {"all": {"current": 7}},
        },
        memory_allocated=_raise,
        memory_reserved=_raise,
        synchronize=lambda d: events.append("synchronize"),
        empty_cache=lambda: events.append("empty_cache"),
        reset_peak_memory_stats=lambda d: events.append("reset_peak"),
        max_memory_allocated=lambda d: 100,
        max_memory_reserved=lambda d: 200,
    )
    tracker = _fake_gpu_tracker(cuda, enabled=True)
    with tracker:
        pass
    assert events[:3] == ["synchronize", "empty_cache", "reset_peak"]
    assert events[-1] == "synchronize"
    assert (tracker.min_allocated, tracker.min_reserved) == (5, 7)
    assert (tracker.peak_allocated, tracker.peak_reserved) == (100, 200)
    assert not tracker._sampler_thread.is_alive()
