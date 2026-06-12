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
* :func:`load_reference_metadata` / :func:`materialize_task` /
  :func:`generate_reference_metadata` — load a
  reference metadata table without downloading, then materialize only the tasks that
  survive filtering (see
  :meth:`tabarena.benchmark.task.metadata.TaskMetadataCollection.from_preset` with
  ``"BeyondArena"``).

The vocabulary "data_foundry_uri" is preserved in the metadata DataFrame and CSV for
backward compatibility with existing downstream code (e.g. ``tabflow_slurm``).
"""

from __future__ import annotations

from tabarena.benchmark.task.data_foundry.adapter import (
    DEFAULT_EVAL_METRICS,
    DataFoundryAdapter,
    convert_curated_container_to_user_task,
)
from tabarena.benchmark.task.data_foundry.beyond_arena import (
    generate_reference_metadata,
    get_beyond_arena_collection,
    load_reference_metadata,
    materialize_task,
    reference_metadata_package_path,
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
    "generate_reference_metadata",
    "get_beyond_arena_collection",
    "get_metadata_for_benchmark_suite",
    "get_path_to_metadata_cache",
    "load_reference_metadata",
    "materialize_task",
    "prepare_collection_for_tabarena",
    "reference_metadata_package_path",
]
