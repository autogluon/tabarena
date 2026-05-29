"""Code for running the benchmark on a full node in a GCP cluster."""

from __future__ import annotations

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TabArenaMetadataBundle
from tabflow_slurm import PathSetup, SlurmSetup, TabArenaBenchmarkSetup, TabArenaV0pt1ResourcesSetup

tabpfn_benchmark_setup = TabArenaBenchmarkSetup(
    benchmark_name="example_tabarena_v0pt1_29052026",
    tasks_to_run_setup=TabArenaMetadataBundle(
        task_metadata="tabarena-v0.1",
        split_indices_to_run="lite",
    ),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(
        models=[
            ("TabPFN-3", 0),
        ],
    ),
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/beyondarena_27052026/bin/python",
    ),
    resources_setup=TabArenaV0pt1ResourcesSetup(
        num_gpus=1,
    ),
    scheduler_setup=SlurmSetup(
        gpu_partition="gpua100highmemoryspotmt",
        cpu_partition="cpuhighmem16mtspot",
        extra_gres=None,
        exclusive_node=True,
    ),
)

tabpfn_benchmark_setup.setup_jobs()
