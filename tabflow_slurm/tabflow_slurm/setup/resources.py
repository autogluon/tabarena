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
    time_limit: int = 3600
    num_cpus: int | None = 8
    num_gpus: int = 0  # Depends on the model
    memory_limit: int | None = 32
