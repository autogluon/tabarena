from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class Timer:
    """Measure the wall-clock duration of a ``with`` block on a monotonic clock.

    The default clock is :func:`time.perf_counter` (``CLOCK_MONOTONIC`` on Linux). Unlike
    :func:`time.time` it is never stepped by NTP or after a VM migration, so a realtime clock
    adjustment during a fit cannot corrupt the measured ``time_train_s`` / ``time_infer_s`` /
    ``time_warmup_s``. Like every Linux clock except ``CLOCK_MONOTONIC_RAW`` it is still
    frequency-slewed, which is harmless for elapsed-time measurements.

    ``start`` and ``stop`` are opaque clock readings, not epoch timestamps: they are only
    meaningful as a difference. Callers that need wall-clock timestamps record them separately
    (the experiment runner stores them in the experiment metadata).

    Attributes:
        start: Clock reading taken by ``__enter__``; ``None`` until the block is entered.
        stop: Clock reading taken by ``__exit__``; ``None`` while still inside the block.

    Example:
        >>> with Timer() as timer:
        ...     do_work()
        >>> timer.duration  # seconds spent in the block
    """

    @staticmethod
    def _zero() -> float:
        return 0.0

    def __init__(self, clock: Callable[[], float] = time.perf_counter, enabled: bool = True):
        """Create a timer.

        Args:
            clock: Zero-argument callable returning seconds as a float. Defaults to
                :func:`time.perf_counter`; tests inject a fake clock through this parameter.
            enabled: When ``False`` the clock is replaced by a constant, so ``duration`` is
                always ``0.0`` and the block costs nothing to time.
        """
        self.start: float | None = None
        self.stop: float | None = None
        self._time = clock if enabled else Timer._zero

    def __enter__(self) -> Timer:
        self.stop = None  # a reused timer measures the new block only
        self.start = self._time()
        return self

    def __exit__(self, *args) -> None:
        self.stop = self._time()

    @property
    def duration(self) -> float:
        """Elapsed seconds.

        After the block has exited this is the closed interval between ``start`` and ``stop``.
        While still inside the block it is the live elapsed time since ``start``.

        Raises:
            RuntimeError: If the timer has not been entered yet.
        """
        if self.start is None:
            raise RuntimeError("Timer has not been started; use it as a context manager.")
        end = self.stop if self.stop is not None else self._time()
        return end - self.start
