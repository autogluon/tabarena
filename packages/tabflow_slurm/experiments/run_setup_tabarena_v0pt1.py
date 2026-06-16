"""Code for running the benchmark on a full node in a GCP cluster.

Uses `TabArenaBenchmarkPlan` to launch several models with different per-model
hardware on top of one shared default setup: TabPFN-3 on a GPU node and
RandomForest on a CPU node. The differing `num_gpus` puts them in two groups, so
`setup_jobs()` emits two `sbatch` commands (one GPU run, one CPU run).
"""

from __future__ import annotations

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabflow_slurm import (
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    TabArenaBenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

benchmark_plan = TabArenaBenchmarkPlan(
    benchmark_name="benchmark_chimeraboost_16062026",
    model_jobs=[
        ModelJob(models=("ChimeraBoost", 25)),
    ],
    tasks=TaskMetadataCollection.from_preset("TabArena-v0.1").subset_tasks(split_indices="lite"),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(),
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/beyondarena_27052026/bin/python",
    ),
    # Run on GCP
    # -> None for these two values so node values are picked up
    # -> CPU partition: 16 vCPUs, 64 GB RAM, 0 GB VRAM
    resources_setup=TabArenaV0pt1ResourcesSetup(memory_limit=None, num_cpus=None),
    scheduler_setup=GCPSlurmSetup(cpu_partition="cpun416mtspotinteractive"),
)

benchmark_plan.setup_jobs()

# # Example:
# benchmark_plan = TabArenaBenchmarkPlan(
#     benchmark_name="example_tabarena_v0pt1_29052026",
#     model_jobs=[
#         # GPU model: override the base (CPU-only) resources to request a GPU.
#         ModelJob(models=("TabPFN-3", 0), name="gpu", resources={"num_gpus": 1}),
#         # Example for CPU model: no resource override, so it runs on the base CPU resources.
#         ModelJob(models=("Linear", 1), name="cpu"),
#     ],
#     tasks=TaskMetadataCollection.from_preset("TabArena-v0.1").subset_tasks(split_indices="lite"),
#     experiment_bundle=TabArenaV0pt1ExperimentBundle(),
#     path_setup=PathSetup(
#         workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
#         python_path="/home/lennart_priorlabs_ai/.venvs/beyondarena_27052026/bin/python",
#     ),
#     resources_setup=TabArenaV0pt1ResourcesSetup(),
#     scheduler_setup=GCPSlurmSetup(),
# )
