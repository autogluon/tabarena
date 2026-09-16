"""Runtime utilities for executing a single benchmark job on a (SLURM) node.

Everything here runs in the per-item worker process before the fit: the offline-weights
environment, the interpreter provenance line, and the Ray runtime for models whose bagged fit
uses Ray fold workers. Cache configuration (OpenML / HuggingFace / TabArena / TabPFN) travels with
the run inside the ``JobBatch`` and is applied by ``run_tabarena_experiment.load_job``.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile

#: Environment exported when a job runs with ``offline_weights``.
#:
#: ``HF_HUB_OFFLINE=1`` makes ``huggingface_hub`` resolve ``refs/<revision>`` from the local cache
#: without the etag HEAD request and raise ``LocalEntryNotFoundError`` on a miss.
#: ``HF_HUB_DISABLE_PROGRESS_BARS=1`` keeps download bars out of the logs.
#: ``AG_FETCH_PRETRAINED_WEIGHTS=false`` makes AutoGluon's built-in foundation models resolve cached
#: checkpoints only and raise ``PretrainedWeightsUnavailableError`` instead of downloading inside
#: the timed fit. A cache miss therefore becomes a failed item recorded like any other failure.
OFFLINE_WEIGHTS_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "AG_FETCH_PRETRAINED_WEIGHTS": "false",
}

#: Fraction of the memory limit given to Ray's object store (unchanged from the original setup).
OBJECT_STORE_FRACTION = 0.3
#: Fraction of the memory limit ``/dev/shm`` must hold for the object store to live there.
MIN_SHM_FRACTION = 0.5
#: Longest AF_UNIX socket path Linux accepts (``sizeof(sun_path) - 1``).
MAX_SOCKET_PATH_BYTES = 107
#: Worst-case suffix Ray appends to its temp dir for the plasma store socket (7-digit pid).
_RAY_SOCKET_SUFFIX = "/session_2026-01-01_00-00-00_000000_4194304/sockets/plasma_store"


def apply_offline_weights_env(enabled: bool) -> None:
    """Export :data:`OFFLINE_WEIGHTS_ENV` when ``enabled``.

    Call it before the first ``huggingface_hub`` import: that library freezes ``HF_HUB_OFFLINE``
    into ``huggingface_hub.constants`` at import time, so a later export does not reach it (a
    warning is printed in that case). AutoGluon reads ``AG_FETCH_PRETRAINED_WEIGHTS`` at call time.
    Ray workers inherit the variables because Ray copies ``os.environ`` when it starts them.
    """
    if not enabled:
        return
    os.environ.update(OFFLINE_WEIGHTS_ENV)
    if "huggingface_hub.constants" in sys.modules:
        print(
            "WARNING: offline-weights environment applied after huggingface_hub was imported; "
            "HF_HUB_OFFLINE may not take effect in this process.",
        )


def log_interpreter_provenance() -> None:
    """Print which interpreter runs this job and where ``autogluon.tabular`` resolves from.

    Fails fast when ``autogluon.tabular`` resolves to a namespace package (``origin`` is ``None``),
    which happens when the working directory or ``sys.path[0]`` shadows the installed package.
    Nothing is imported: only ``importlib.util.find_spec`` is consulted.
    """
    import importlib.util

    try:
        spec = importlib.util.find_spec("autogluon.tabular")
    except (ModuleNotFoundError, ValueError):
        spec = None
    origin = None if spec is None else spec.origin
    path0 = sys.path[0] if sys.path else None
    print(
        f"python={sys.executable} safe_path={sys.flags.safe_path} sys.path[0]={path0!r} autogluon.tabular={origin}",
    )
    if spec is not None and origin is None:
        raise RuntimeError(
            "autogluon.tabular resolves to a namespace package: the working directory or sys.path shadows the "
            "installed package. Run python from the repository root (or with -P), never from the workspace root.",
        )


def ray_object_store_bytes(memory_limit_gb: float) -> int:
    """Object store size for a job with ``memory_limit_gb`` gigabytes (:data:`OBJECT_STORE_FRACTION`)."""
    return int(memory_limit_gb * (1024.0**3) * OBJECT_STORE_FRACTION)


def plasma_directory_for(
    *,
    dev_shm_bytes: float,
    object_store_bytes: int,
    min_shm_bytes: float | None = None,
    fallback_dir: str,
) -> str | None:
    """Where Ray's plasma store lives: ``None`` for ``/dev/shm``, else ``fallback_dir`` on disk.

    ``/dev/shm`` is used when it holds at least ``min_shm_bytes``. The default threshold reproduces
    the original rule (shared memory at least :data:`MIN_SHM_FRACTION` of the memory limit, with the
    object store at :data:`OBJECT_STORE_FRACTION` of it), so it is
    ``object_store_bytes * MIN_SHM_FRACTION / OBJECT_STORE_FRACTION``.
    """
    if min_shm_bytes is None:
        min_shm_bytes = object_store_bytes * MIN_SHM_FRACTION / OBJECT_STORE_FRACTION
    if dev_shm_bytes >= min_shm_bytes:
        return None
    print(
        "WARNING: /dev/shm is too small for the Ray object store, switching to disk! "
        f"Available shared memory: {dev_shm_bytes / 1e9:.2f} GB, "
        f"required minimum: {min_shm_bytes / 1e9:.2f} GB.",
    )
    return fallback_dir


def ray_init_kwargs(
    *,
    num_cpus: int,
    num_gpus: int,
    memory_limit: int,
    ray_dir: str,
    plasma_directory: str | None,
    log_to_driver: bool = False,
) -> dict:
    """The ``ray.init`` keyword arguments of a SLURM job (no ``runtime_env``).

    Worker logs stay in Ray's session directory unless ``log_to_driver``; the runner copies them into
    the SLURM output directory when an item fails.
    """
    ray_mem_in_b = int(int(memory_limit) * (1024.0**3))
    return {
        "address": "local",
        "_memory": ray_mem_in_b,
        "object_store_memory": ray_object_store_bytes(memory_limit),
        "_temp_dir": ray_dir,
        "include_dashboard": False,
        "logging_level": logging.INFO,
        "log_to_driver": log_to_driver,
        "num_gpus": num_gpus,
        "num_cpus": num_cpus,
        "_plasma_directory": plasma_directory,
    }


def ray_socket_path_fits(ray_dir: str) -> bool:
    """Whether Ray's longest socket path under ``ray_dir`` stays within :data:`MAX_SOCKET_PATH_BYTES`."""
    return len((ray_dir + _RAY_SOCKET_SUFFIX).encode("utf-8")) <= MAX_SOCKET_PATH_BYTES


def make_ray_temp_dir(ray_temp_root: str | None = None) -> str:
    """A fresh Ray temp dir whose socket paths fit the AF_UNIX limit.

    Under ``ray_temp_root`` (the per-job scratch the submit template passes) the dir is a short
    ``mkdtemp`` name; without a root it is ``mkdtemp()/ray`` under ``TMPDIR``. When the result would
    push Ray's socket paths past 107 bytes, a dir under ``/tmp`` is used instead and a warning names it.
    """
    if ray_temp_root is not None:
        os.makedirs(ray_temp_root, exist_ok=True)
        ray_dir = tempfile.mkdtemp(prefix="r", dir=ray_temp_root)
    else:
        ray_dir = tempfile.mkdtemp() + "/ray"
    if ray_socket_path_fits(ray_dir):
        return ray_dir
    fallback = tempfile.mkdtemp(prefix="ray_", dir="/tmp")
    print(f"WARNING: Ray temp dir {ray_dir!r} is too long for AF_UNIX socket paths; using {fallback!r} instead.")
    return fallback


def setup_slurm_job(
    *,
    num_cpus: int,
    num_gpus: int,
    memory_limit: int,
    setup_ray_for_slurm_shared_resources_environment: bool,
    ray_temp_root: str | None = None,
    log_to_driver: bool = False,
) -> None | str:
    """Set up Ray (and silence loky logs) for executing a single benchmark job on a node.

    Parameters
    ----------
    num_cpus : int
        The number of CPUs to use for the experiment (needed for proper Ray setup).
    num_gpus : int
        The number of GPUs to use for the experiment (needed for proper Ray setup).
    memory_limit : int
        The memory limit to use for the experiment (needed for proper Ray setup).
    setup_ray_for_slurm_shared_resources_environment : bool
        If running on a SLURM cluster, initialize Ray with a unique temp dir and explicit resources.
        Otherwise, given the shared filesystem, Ray would use the same temp dir for all workers and
        crash (semi-randomly).
    ray_temp_root : str | None
        Directory to create the Ray temp dir under (the per-job node-local scratch); ``None`` uses
        ``TMPDIR``. See :func:`make_ray_temp_dir` for the socket-path guard.
    log_to_driver : bool
        Whether Ray forwards worker stdout/stderr to this process (off by default, as in AutoGluon's
        own Ray init; the runner copies the worker logs when an item fails).

    Returns:
    -------
    None | str
        The Ray temp dir when Ray was started (the caller removes it), else ``None``.
    """
    # Silence loky resource tracker clean up logs
    logging.getLogger("loky.backend.resource_tracker").setLevel(logging.CRITICAL)
    log_interpreter_provenance()

    ray_dir = None
    if setup_ray_for_slurm_shared_resources_environment:
        print("Setting up Ray for SLURM job in a shared resources environment.")
        import warnings

        import ray

        os.environ["RAY_DISABLE_RETRIES"] = "1"

        ray_dir = make_ray_temp_dir(ray_temp_root)
        plasma_directory = plasma_directory_for(
            dev_shm_bytes=ray._private.utils.get_shared_memory_bytes(),
            object_store_bytes=ray_object_store_bytes(memory_limit),
            fallback_dir=ray_dir,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            ray.init(
                **ray_init_kwargs(
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                    memory_limit=memory_limit,
                    ray_dir=ray_dir,
                    plasma_directory=plasma_directory,
                    log_to_driver=log_to_driver,
                ),
            )
    return ray_dir
