"""CPU affinity and thread-pool helpers for the benchmark driver.

The protocol reports a ``num_cpus`` budget per fit but never forces the thread pools to it. These
helpers make the environment observable and consistent instead: :func:`usable_cpu_count` reads
the CPUs this process may actually run on (the affinity mask), :func:`check_cpu_budget` compares
that with the budget and complains when they differ, and :func:`cpu_thread_info` snapshots the
effective thread configuration for the result metadata.
None of them imports torch or any other heavy library.
"""

from __future__ import annotations

import contextlib
import os
import sys
import warnings
from typing import Literal

__all__ = [
    "THREAD_ENV_VARS",
    "check_cpu_budget",
    "cpu_thread_info",
    "usable_cpu_count",
]

#: The thread-count variables OpenMP, MKL and OpenBLAS read when their pools are created.
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def usable_cpu_count() -> int:
    """Number of CPUs this process may run on.

    The size of the affinity mask (``os.sched_getaffinity(0)``), which is what a cpuset or SLURM
    cgroup CPU confinement narrows. Falls back to ``os.cpu_count()`` where the platform has no
    affinity mask, and to 1 when even that is unknown.
    """
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        try:
            return len(getaffinity(0))
        except OSError:
            pass
    return os.cpu_count() or 1


def cpu_thread_info() -> dict:
    """Snapshot the effective CPU thread configuration of this process.

    Returns a JSON-able dict with ``usable_cpus`` (affinity count), ``cpu_count``, the current
    values of :data:`THREAD_ENV_VARS` (``None`` when unset), ``torch_num_threads`` (only when torch
    is already imported; this never imports it) and ``threadpools``, the pools threadpoolctl can
    see (``None`` when threadpoolctl is not installed).
    """
    info: dict = {
        "usable_cpus": usable_cpu_count(),
        "cpu_count": os.cpu_count(),
        "thread_env": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "torch_num_threads": None,
        "threadpools": None,
    }
    torch = sys.modules.get("torch")
    if torch is not None:
        # A half-imported or stubbed torch must not break the snapshot.
        with contextlib.suppress(Exception):
            info["torch_num_threads"] = int(torch.get_num_threads())
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return info
    info["threadpools"] = [
        {key: pool.get(key) for key in ("user_api", "internal_api", "prefix", "num_threads")}
        for pool in threadpool_info()
    ]
    return info


def check_cpu_budget(
    num_cpus: int | None,
    *,
    on_mismatch: Literal["error", "warn", "off"] = "error",
) -> dict:
    """Compare the fit's CPU budget with the CPUs this process may use.

    Args:
        num_cpus: The protocol's CPU budget for the fit. ``None`` means auto-detected, resolved
            through :func:`tabarena.utils.resources.detect_num_cpus`.
        on_mismatch: ``"error"`` raises ``RuntimeError``, ``"warn"`` emits a ``RuntimeWarning``,
            ``"off"`` only records the outcome.

    Returns:
        ``{"num_cpus", "usable_cpus", "matches"}``.

    Raises:
        RuntimeError: The two counts differ and ``on_mismatch="error"``. The message names both
            numbers and the two remedies: confine the job to ``num_cpus`` CPUs (a cpuset, for
            example SLURM ``--cpus-per-task`` with cgroup CPU confinement), or set ``num_cpus`` to
            the usable count.
        ValueError: ``on_mismatch`` is not one of the three modes.
    """
    if on_mismatch not in ("error", "warn", "off"):
        raise ValueError(f"on_mismatch must be 'error', 'warn' or 'off', got {on_mismatch!r}")
    if num_cpus is None:
        from tabarena.utils.resources import detect_num_cpus

        num_cpus = detect_num_cpus()
    num_cpus = int(num_cpus)
    usable = usable_cpu_count()
    result = {"num_cpus": num_cpus, "usable_cpus": usable, "matches": num_cpus == usable}
    if result["matches"] or on_mismatch == "off":
        return result
    message = (
        f"CPU budget mismatch: the fit is given num_cpus={num_cpus} but this process may run on "
        f"{usable} CPUs (affinity mask), so thread pools sized from the machine will not match the "
        f"budget the results report. Either confine the job to {num_cpus} CPUs with a cpuset (for "
        f"example SLURM --cpus-per-task={num_cpus} with cgroup CPU confinement) or set num_cpus={usable}."
    )
    if on_mismatch == "error":
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return result
