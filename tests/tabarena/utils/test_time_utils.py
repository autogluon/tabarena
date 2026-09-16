from __future__ import annotations

import time

import pytest

from tabarena.utils.time_utils import Timer


def test_default_clock_is_perf_counter():
    """The default clock is monotonic and not adjustable, so a realtime clock step cannot corrupt durations."""
    assert Timer()._time is time.perf_counter
    info = time.get_clock_info("perf_counter")
    assert info.monotonic is True
    assert info.adjustable is False


def test_duration_with_injected_clock():
    clock = iter([10.0, 12.5]).__next__
    with Timer(clock=clock) as timer:
        pass
    assert timer.start == 10.0
    assert timer.stop == 12.5
    assert timer.duration == 2.5


def test_live_duration_inside_block():
    with Timer() as timer:
        assert timer.stop is None
        d1 = timer.duration
        time.sleep(0.01)
        d2 = timer.duration
    assert 0 <= d1 <= d2
    assert timer.stop is not None
    assert timer.duration >= d2


def test_disabled_timer_reports_zero():
    with Timer(enabled=False) as timer:
        time.sleep(0.005)
    assert timer.duration == 0.0
    assert timer.start == 0.0
    assert timer.stop == 0.0


def test_disabled_timer_ignores_the_injected_clock():
    calls = []

    def clock() -> float:
        calls.append(1)
        return 1.0

    with Timer(clock=clock, enabled=False) as timer:
        pass
    assert timer.duration == 0.0
    assert calls == []


def test_duration_before_start_raises():
    timer = Timer()
    assert timer.start is None
    assert timer.stop is None
    with pytest.raises(RuntimeError, match="not been started"):
        _ = timer.duration


def test_reentering_restarts_the_measurement():
    clock = iter([1.0, 4.0, 10.0, 11.0]).__next__
    timer = Timer(clock=clock)
    with timer:
        pass
    assert timer.duration == 3.0
    with timer:
        assert timer.stop is None
    assert timer.duration == 1.0
