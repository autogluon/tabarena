"""Compute and time-budget resources for the benchmark jobs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResourcesSetup:
    """Compute and time-budget resources for the benchmark jobs."""

    time_limit: int
    """Time limit for each fit of a model in seconds -- including time for validation.
    By default, 3600 seconds is used."""
    num_cpus: int | None
    """Number of CPUs to use for the job.
    If None, use all available CPUs."""
    num_gpus: int
    """Number of GPUs to use for the jobs (SLURM allocation and Ray)."""
    memory_limit: int | None
    """Memory/RAM limit for the jobs in GB.
    If None, use all available memory."""

    time_limit_for_model_agnostic_preprocessing: int | None = None
    """The time limit for the model agnostic preprocessing step."""
    time_limit_with_model_agnostic_preprocessing: bool = False
    """Whether the model agnostic preprocessing should influence the
    fit time of the model:
        - If False (default), we stop fitting a model after `time_limit`.
        - If True, we stop fitting a model after `time_limit` minus the
            time it took for model agnostic preprocessing.
    """
    num_gpus_model: int | None = None
    """Number of GPUs passed to a model for fitting.
    If None (default), uses the same value as ``num_gpus``.
    Set to 0 to reserve the GPU for preprocessing (e.g. sentence-transformer
    encoding) while fitting models on CPU only."""
    fake_memory_for_estimates: int | None = None
    """Experimental parameter that is to be ignored!

    If not None, this value is reported to models in place of `memory_limit`
    so the model's internal memory estimates are compared against it instead
    of the actually available memory on the system. Values in GB as
    `memory_limit`.

    This can be useful if:
        - To test or overrule (bad) memory estimates.
        - For models that use CPU memory as a proxy for GPU memory (e.g. most
          TFMs), this can be used if the job has much more VRAM than CPU memory.
    """

    @property
    def time_limit_per_config(self) -> int:
        """Maximal time limit per config plus some overhead."""
        total = self.time_limit
        if self.time_limit_for_model_agnostic_preprocessing is not None:
            total += self.time_limit_for_model_agnostic_preprocessing
        if self.time_limit_with_model_agnostic_preprocessing:
            total += 60 * 15  # constant SLURM overhead
        return total

    @property
    def effective_memory_limit(self) -> int | None:
        """Memory limit reported to models (honors the `fake_memory_for_estimates` override)."""
        return self.fake_memory_for_estimates if self.fake_memory_for_estimates is not None else self.memory_limit

    @property
    def effective_num_gpus_model(self) -> int:
        """Number of GPUs used for model fitting (`num_gpus_model`, falling back to `num_gpus`)."""
        return self.num_gpus_model if self.num_gpus_model is not None else self.num_gpus


@dataclass
class TabArenaV0pt1ResourcesSetup(ResourcesSetup):
    """Hardware resources and time budget used in TabArena-v0.1."""

    time_limit: int = 3600
    num_cpus: int | None = 8
    num_gpus: int = 0  # Default 0, only some models use GPU
    memory_limit: int | None = 32


@dataclass
class BeyondArenaResourcesSetup(ResourcesSetup):
    """Hardware resources and time budget used in BeyondArena.

    For BeyondArena, we used auto-detect for cpus and memory on the node.
    And selected nods with as many resources as we wanted.

    The nodes were different for GPU and CPU jobs:
        * CPU jobs used: 16 vCPUs, 64 GB RAM, 0 GB VRAM
        * GPUs jobs used (varied due to availability and cost):
            * 12 vCPUs, 85 GB RAM, 40 GB VRAM (for MLPs with less than 250k rows)
            * 12 vCPUs, 170 GB RAM, 80 GB VRAM
            * 24 vCPUs, 180 GB RAM, 96 GB VRAM
            * 13 vCPUs, 125 GB RAM, 80 GB VRAM

    Adjust your benchmark partitions, memory limit and num_cpus as
    needed to reflect this setting, when benchmarking on a shared
    resource cluster.
    """

    time_limit: int = 3600 * 4
    num_gpus: int = 0  # Default 0, only some models use GPU
    # When using GPU, we set the below to match the GPU VRAM
    # fake_memory_for_estimates: int = 80 # 40/80/96 depending on the GPU node used

    memory_limit: int | None = None
    num_cpus: int | None = None
