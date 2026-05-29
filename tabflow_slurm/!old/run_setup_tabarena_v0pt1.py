"""Code for running the benchmark on a full node in a GCP cluster."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .setup_slurm_base_v2 import BenchmarkSetup2026, PathSetup, SlurmSetup


@dataclass
class ExtraPathSetup(PathSetup):
    """Path setup for environment."""

    base_path: str = "/path/to/workspace/"
    tabarena_repo_name: str = "XXX"
    venv_name: str = "XXXX"
    openml_cache_from_base_path: str | Literal["auto"] = "auto"
    """We do not have workspace drives, so we can save in homedir"""


@dataclass
class TabArenaV0pt1SingleNodeBenchmarkSetup(BenchmarkSetup2026):
    """Benchmark setup for single node environment for TabArena-v0.1."""

    shuffle_features: bool = False
    """Was False by default in TabArena."""
    n_random_configs: int = 200
    """TabArena-v0.1 default"""
    dynamic_tabarena_validation_protocol: bool = False
    """Only used in v0.2 or larger with new data foundry task metadata integration."""
    preprocessing_pipelines: list[str] = field(default_factory=lambda: ["default"])

    # Auto-detect all cpus and memory on the node (can restrict jobs with this further)
    memory_limit: None = None
    num_cpus: None = None


TabArenaV0pt1SingleNodeBenchmarkSetup(
    benchmark_name="benchmark_xxx_22052026",
    task_metadata="tabarena-v0.1",
    num_gpus=1,
    models=[
        ("XXX", 0),
    ],
    # Can overwrite these classes as needed.
    path_setup=ExtraPathSetup(),
    slurm_setup=SlurmSetup(
        gpu_partition="gpua100highmemoryspotmt",
        cpu_partition="cpuhighmem16mtspot",
        extra_gres=None,
        exclusive_node=True,
    ),
).setup_jobs(array_job_limit=100)
