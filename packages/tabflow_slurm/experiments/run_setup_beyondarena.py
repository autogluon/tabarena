"""Code for running the BeyondArena benchmark on a full node in a GCP cluster.

The tasks come from the Data Foundry ``BeyondArena`` collection, which the
``BeyondArenaContext`` owns: it loads reference metadata (no downloads); the
benchmark setup later materializes (downloads + converts) only the surviving
datasets into local OpenML tasks on this head node. Scope a chosen subset with
``task_subset=TaskSubset(dataset_names=[...])`` and only those datasets are fetched.
"""

from __future__ import annotations

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
from tabarena.contexts.beyond_arena import BeyondArenaContext
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
    # The BeyondArena context owns the Data Foundry collection. No `task_subset` here
    # runs the full suite; scope it with e.g. `task_subset=TaskSubset(dataset_names=[...])`.
    context=BeyondArenaContext(),
    experiment_bundle=BeyondArenaExperimentBundle(),
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/beyondarena_27052026/bin/python",
    ),
    resources_setup=BeyondArenaResourcesSetup(),
    scheduler_setup=GCPSlurmSetup(bundle_size=1),
)

benchmark_plan.setup_jobs()
