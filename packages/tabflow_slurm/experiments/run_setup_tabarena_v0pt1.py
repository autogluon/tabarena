"""Code for running the benchmark on a full node in a GCP cluster.

Uses `TabArenaBenchmarkPlan` to launch several models with different per-model
hardware on top of one shared default setup: TabPFN-3 on a GPU node and
RandomForest on a CPU node. The differing `num_gpus` puts them in two groups, so
`setup_jobs()` emits two `sbatch` commands (one GPU run, one CPU run).
"""

from __future__ import annotations

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabarena.contexts import TabArenaContext
from tabflow_slurm import (
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    TabArenaBenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

benchmark_plan = TabArenaBenchmarkPlan(
    benchmark_name="example_tabarena_v0pt1_29052026",
    model_jobs=[
        # GPU model: override the base (CPU-only) resources to request a GPU.
        ModelJob(models=("TabPFN-3", 0), name="gpu", resources={"num_gpus": 1}),
        # Example for CPU model: no resource override, so it runs on the base CPU resources.
        ModelJob(models=("Linear", 1), name="cpu"),
    ],
    # The TabArena-v0.1 context owns the task metadata + subset predicates; `task_subset`
    # scopes `context.build_jobs` (here `subset="lite"` keeps each dataset's first split).
    #
    # Caches use their library defaults (OpenML `~/.cache/openml`, HuggingFace
    # `~/.cache/huggingface/hub`, TabArena `~/.cache/tabarena`). To relocate them — e.g. onto
    # shared cluster storage so the head node and every worker resolve the same files — pass a
    # CacheConfig to the context (the plan embeds it in the JobBatch; reuse it in the eval script):
    #     from tabarena.caching import CacheConfig
    #     context=TabArenaContext(cache_config=CacheConfig.from_root("/shared/tabarena-caches")),
    context=TabArenaContext(),
    task_subset=TaskSubset(subset="lite"),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(),
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/beyondarena_27052026/bin/python",
    ),
    resources_setup=TabArenaV0pt1ResourcesSetup(),
    scheduler_setup=GCPSlurmSetup(),
)

benchmark_plan.setup_jobs()
