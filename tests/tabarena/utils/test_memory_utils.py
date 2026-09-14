from __future__ import annotations

import time

import pytest

from tabarena.utils.memory_utils import CpuMemoryTracker, GpuMemoryTracker


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
