"""This file functions as a setup for the configuration of the benchmark.

We used hardcode dataclasses to specific what we benchmark we run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from experimental.feature_selection_benchmark.data_integration.fs_data_constants import (
    BENCHMARK_TASK_COLLECTION_NAME,
    DATA_FOUNDRY_CACHE,
)
from tabflow_slurm.benchmarking_setup.data_foundry_integration.data_foundry_task_creator import (
    get_metadata_for_benchmark_suite,
)
from tabflow_slurm.setup_slurm_base_v2 import BenchmarkSetup2026, PathSetup, SlurmSetup

ALL_TASK_METADATA = get_metadata_for_benchmark_suite(
    BENCHMARK_TASK_COLLECTION_NAME, data_foundry_cache=DATA_FOUNDRY_CACHE
)
FS_TIME_LIMIT = 3600


@dataclass
class UniPathSetup(PathSetup):
    """Path setup for Lennart's environment."""

    base_path: str = "/work/dlclarge1/purucker-fs_benchmark/"
    tabarena_repo_name: str = "fsbench"
    venv_name: str = "fs_bench_env"

@dataclass
class UniPathSetupDominika(PathSetup):
    """Path setup for Dominika's environment."""

    base_path: str = "/work/dlclarge1/purucker-fs_benchmark/"
    tabarena_repo_name: str = "fsbench_matusd"
    venv_name: str = "fsbenchvenv2"

@dataclass
class UniSlurmSetup(SlurmSetup):
    """We can use mostly the defaults."""
    

@dataclass
class TabArenaBenchmarkSetup(BenchmarkSetup2026):
    """We stick mostly to the default of TabArena-v0.1.
    This class serves as some default changes across runs.
    """

    benchmark_name: str = "feature_selection_benchmark_2026"
    task_metadata: str = "NOTSET"  # Placeholder due to dataclass inheritance problems

    n_random_configs: int = 25
    """Adjusted to our compute budget"""
    dynamic_tabarena_validation_protocol: bool = False
    """We stick to default benchmarking setting."""
    time_limit_for_model_agnostic_preprocessing: int = FS_TIME_LIMIT
    """Only here to add an overhead per config to the SLURM job."""

    # Add the path and slurm setup from above
    path_setup: PathSetup = field(default_factory=UniPathSetup)
    slurm_setup: SlurmSetup = field(default_factory=UniSlurmSetup)


@dataclass
class FSBenchmarkConfig:
    """Settings for the feature selection methods and experiments.

    This dataset class serves as a generator for the preprocessing pipelines that we want to run.
    """

    proxy_model_configs: list[str] = field(default_factory=lambda: ["lgbm"])
    """The proxy model(s) to use for the feature selection methods that require a proxy model."""
    fs_time_limits: list[int] = field(default_factory=lambda: [FS_TIME_LIMIT])
    """The time limit(s) for the feature selection methods to run (in seconds)."""
    total_selection_budget: int = 5
    """The total budget of feature selection steps (b in the feature-count formula)."""
    include_default: bool = False
    """Whether to include the default model (no feature selection) as a baseline in the benchmark."""

    def get_default_preprocessing_configs(self, fs_methods: list[str]) -> list[str]:
        """Generate the preprocessing pipelines to run."""
        from tabarena.benchmark.feature_selection_methods.feature_selection_benchmark_utils import (
            get_fs_benchmark_preprocessing_pipelines,
        )

        return get_fs_benchmark_preprocessing_pipelines(
            fs_methods=fs_methods,
            proxy_model_config=self.proxy_model_configs,
            time_limit=self.fs_time_limits,
            total_budget=self.total_selection_budget,
            include_default=self.include_default,
        )