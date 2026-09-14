"""Helpers to auto-detect the compute resources of the current node."""

from __future__ import annotations


def detect_num_cpus() -> int:
    """Detect the number of CPUs this process may run on.

    Reads the affinity mask (see :func:`tabarena.utils.thread_utils.usable_cpu_count`), so a
    cpuset or SLURM cgroup CPU confinement is honored and an auto-detected ``num_cpus`` always
    passes :func:`tabarena.utils.thread_utils.check_cpu_budget`.
    """
    from tabarena.utils.thread_utils import usable_cpu_count

    return usable_cpu_count()


def detect_memory_limit_gb() -> int:
    """Detect the available memory on the current node, in GB."""
    from autogluon.common.utils.resource_utils import ResourceManager

    return int(ResourceManager.get_memory_size(format="GB"))


def detect_num_gpus() -> int:
    """Detect the number of available GPUs on the current node (0 if none)."""
    from autogluon.common.utils.resource_utils import ResourceManager

    return ResourceManager.get_gpu_count()
