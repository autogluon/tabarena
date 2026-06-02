"""Runtime utilities for executing a single benchmark job on a (SLURM) node."""

from __future__ import annotations

import logging

import openml


def setup_slurm_job(
    *,
    openml_cache_dir: str,
    num_cpus: int,
    num_gpus: int,
    memory_limit: int,
    setup_ray_for_slurm_shared_resources_environment: bool,
) -> None | str:
    """Ensure correct caching and usage of directories for OpenML and TabRepo.

    Parameters
    ----------
    openml_cache_dir : str
        The path to the OpenML cache directory, or "auto" to use the default OpenML cache directory.
    num_cpus : int
        The number of CPUs to use for the experiment (needed for proper Ray setup).
    num_gpus : int
        The number of GPUs to use for the experiment (needed for proper Ray setup).
    memory_limit : int
        The memory limit to use for the experiment (needed for proper Ray setup).
    setup_ray_for_slurm_shared_resources_environment : bool
        If running on a SLURM cluster, we need to initialize Ray with extra options and a unique tempr dir.
        Otherwise, given the shared filesystem, Ray will try to use the same temp dir for all workers and
        crash (semi-randomly).
    """
    # Silence loky resource tracker clean up logs
    logging.getLogger("loky.backend.resource_tracker").setLevel(logging.CRITICAL)

    if openml_cache_dir == "auto":
        print("Using the default OpenML cache directory.")
    else:
        print(f"Setting OpenML cache directory to: {openml_cache_dir}")
        openml.config.set_root_cache_directory(root_cache_directory=openml_cache_dir)

    # SLURM save Ray setup in a shared resource system
    ray_dir = None
    if setup_ray_for_slurm_shared_resources_environment:
        print("Setting up Ray for SLURM job in a shared resources environment.")
        import os
        import tempfile

        import ray

        os.environ["RAY_DISABLE_RETRIES"] = "1"

        ray_dir = tempfile.mkdtemp() + "/ray"

        min_plasma_storage_size = int(memory_limit * 0.5)
        ray_mem_in_b = int(int(memory_limit) * (1024.0**3))

        _plasma_directory = None
        dev_shm_size = ray._private.utils.get_shared_memory_bytes() / 1e9
        if dev_shm_size < min_plasma_storage_size:
            print(
                "WARNING: /dev/shm is full, switching to /tmp usage! "
                f"Available shared memory size: {dev_shm_size} GB, "
                f"Required minimum for Ray plasma store: {min_plasma_storage_size} GB.",
            )
            # Likely slower but runs at least.
            _plasma_directory = ray_dir

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            ray.init(
                address="local",
                _memory=ray_mem_in_b,
                object_store_memory=int(ray_mem_in_b * 0.3),
                _temp_dir=ray_dir,
                include_dashboard=False,
                logging_level=logging.INFO,
                log_to_driver=True,
                num_gpus=num_gpus,
                num_cpus=num_cpus,
                _plasma_directory=_plasma_directory,
                # Ensure Loky uses forkserver and avoids bugs from running parallel across ray workers
                runtime_env={
                    "env_vars": {
                        "LOKY_START_METHOD": "forkserver",
                    },
                },
            )
    return ray_dir
