"""Adapter for converting Data Foundry collections into TabArena UserTasks.

Primary entry points:

* :class:`DataFoundryAdapter` — drives a :class:`data_foundry.collections.DatasetCollection`
  (online ``HuggingFaceSource`` or offline ``LocalWarehouseSource``) and converts every
  container into a TabArena :class:`UserTask`, returning the joined task-metadata DataFrame.
* :func:`convert_curated_container_to_user_task` — single-container conversion, useful when
  iterating containers yourself.
* :func:`prepare_collection_for_tabarena` / :func:`get_metadata_for_benchmark_suite` —
  download (if needed) + convert + persist a task-metadata CSV under the OpenML cache,
  for downstream slurm jobs to discover.

The vocabulary "data_foundry_uri" is preserved in the metadata DataFrame and CSV for
backward compatibility with existing downstream code (e.g. ``tabflow_slurm``).
"""

from __future__ import annotations

from tabarena.benchmark.task.data_foundry.adapter import (
    DEFAULT_EVAL_METRICS,
    DataFoundryAdapter,
    convert_curated_container_to_user_task,
)
from tabarena.benchmark.task.data_foundry.metadata_cache import (
    get_metadata_for_benchmark_suite,
    get_path_to_metadata_cache,
    prepare_collection_for_tabarena,
)

__all__ = [
    "DEFAULT_EVAL_METRICS",
    "DataFoundryAdapter",
    "convert_curated_container_to_user_task",
    "get_metadata_for_benchmark_suite",
    "get_path_to_metadata_cache",
    "prepare_collection_for_tabarena",
]
