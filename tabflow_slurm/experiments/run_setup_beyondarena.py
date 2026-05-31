"""Code for running the BeyondArena benchmark on a full node in a GCP cluster.

Uses `TabArenaBenchmarkPlan` to launch several models with different per-model
hardware on top of one shared default setup: TabPFN-3 on a GPU node and
RandomForest on a CPU node. The differing `num_gpus` puts them in two groups, so
`setup_jobs()` emits two `sbatch` commands (one GPU run, one CPU run).

The tasks come from the Data Foundry ``BeyondArena`` collection via
`BeyondArenaLiteMetadataBundle`: it loads reference metadata, applies the filters
below, and (on this head node) downloads + converts only the surviving datasets
into local OpenML tasks. Set `dataset_names_to_run=[...]` to run a chosen subset
and only those datasets are fetched.
"""

from __future__ import annotations

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
from tabarena.benchmark.task.metadata import BeyondArenaLiteMetadataBundle
from tabflow_slurm import (
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    TabArenaBenchmarkPlan,
    BeyondArenaResourcesSetup,
)

benchmark_plan = TabArenaBenchmarkPlan(
    benchmark_name="example_beyondarena_31052026",
    model_jobs=[
        # FIXME: need to use correct resource keys here and check how to make that easier.
        # GPU model: override the base (CPU-only) resources to request a GPU.
        ModelJob(models=("TabPFN-3", 0), name="gpu", resources={"num_gpus": 1}),
        # Example for CPU model: no resource override, so it runs on the base CPU resources.
        ModelJob(models=("Linear", 1), name="cpu"),
    ],
    # Pass `dataset_names_to_run=[...]` to filter to specific datasets before download.
    tasks_to_run_setup=BeyondArenaLiteMetadataBundle(),
    experiment_bundle=BeyondArenaExperimentBundle(),
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/beyondarena_27052026/bin/python",
    ),
    resources_setup=BeyondArenaResourcesSetup(),
    scheduler_setup=GCPSlurmSetup(),
)

benchmark_plan.setup_jobs()
