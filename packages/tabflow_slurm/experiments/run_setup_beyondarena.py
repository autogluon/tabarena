"""Code for running the BeyondArena benchmark on a full node in a GCP cluster.

The tasks come from the Data Foundry ``BeyondArena`` collection via
`BeyondArenaLiteMetadataBundle`: it loads reference metadata, applies the filters
below, and (on this head node) downloads + converts only the surviving datasets
into local OpenML tasks. Set `dataset_names_to_run=[...]` to run a chosen subset
and only those datasets are fetched.
"""

from __future__ import annotations

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
from tabarena.benchmark.task.metadata import BeyondArenaMetadataBundle
from tabflow_slurm import (
    BeyondArenaResourcesSetup,
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    TabArenaBenchmarkPlan,
)

benchmark_plan = TabArenaBenchmarkPlan(
    benchmark_name="example_beyondarena_31052026",
    model_jobs=[
        ModelJob(
            models=("TabPFN-3", 0),
            name="gpu",
            resources={
                "num_gpus": 1,
                "fake_memory_for_estimates": 80,  # we have 80 GB VRAM GPU.
                "time_limit": 3600 * 12,  # higher time limit for TFM job as for TabICL
            },
        ),
    ],
    tasks_to_run_setup=BeyondArenaMetadataBundle(),
    experiment_bundle=BeyondArenaExperimentBundle(),
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/beyondarena_27052026/bin/python",
    ),
    resources_setup=BeyondArenaResourcesSetup(),
    scheduler_setup=GCPSlurmSetup(bundle_size=1),
)

benchmark_plan.setup_jobs()
