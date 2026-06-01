"""Helpers to auto-detect the compute resources of the current node."""

from __future__ import annotations


def detect_num_cpus() -> int:
    """Detect the number of available CPUs on the current node."""
    from autogluon.common.utils.cpu_utils import get_available_cpu_count

    return get_available_cpu_count(only_physical_cores=False)


def detect_memory_limit_gb() -> int:
    """Detect the available memory on the current node, in GB."""
    from autogluon.common.utils.resource_utils import ResourceManager

    return int(ResourceManager.get_memory_size(format="GB"))
