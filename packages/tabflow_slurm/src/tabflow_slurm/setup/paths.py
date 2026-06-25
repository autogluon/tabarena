"""Path configuration and bundled-script resolution for the benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# The runner script and submit template live at the package root, one level
# up from this `setup/` subpackage.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def get_run_script_path() -> Path:
    """Absolute path to the bundled ``run_tabarena_experiment.py`` runner script.

    This is the script a single SLURM array task executes to fit one
    ``(task, fold, repeat, config)`` item. It is resolved relative to this
    installed package, so callers never need to hardcode a checkout location.
    """
    return _PACKAGE_ROOT / "run_tabarena_experiment.py"


def get_submit_script_path() -> Path:
    """Absolute path to the bundled SLURM ``submit_template.sh`` array script.

    This is the ``sbatch`` array script that reads the generated job JSON and
    invokes the runner. It is resolved relative to this installed package.
    """
    return _PACKAGE_ROOT / "submit_template.sh"


@dataclass
class PathSetup:
    """Configure paths for the benchmark.

    The user must provide a `workspace` directory and the `python_path` to use
    for the SLURM jobs. `run_script` and `submit_script` default to the scripts
    bundled with this package (see `get_run_script_path` / `get_submit_script_path`).

    Cache locations (OpenML / HuggingFace / TabArena) are NOT configured here — set them on
    the arena context via `TabArenaContext(cache_config=CacheConfig(...))` (the same surface
    used by the sequential/async API). The setup embeds that config in the `JobBatch`, and each
    worker applies it automatically (see `tabarena.caching.CacheConfig`).

    The workspace is a persistent directory that all SLURM jobs can access. We
    create and use the following structure inside it:
        - WORKSPACE
            - output            -- all output data from running the benchmark
                                   (one `benchmark_name` subfolder each)
            - slurm_out         -- all SLURM output logs
                                   (one `benchmark_name` subfolder each)
            - setup_out         -- generated configs YAML + SLURM job JSON
                                   (one `benchmark_name` subfolder each)
    """

    workspace: str | Path
    """Persistent workspace directory that all jobs can access. Holds the
    `output/`, `slurm_out/`, and `setup_out/` folders."""
    python_path: str | Path
    """Python executable to use for the SLURM jobs. Should point to the cluster
    venv (e.g. pass `sys.executable` if this setup runs inside that venv)."""
    run_script: str | Path = field(default_factory=get_run_script_path)
    """Path to the python script that runs a single benchmark experiment.
    Defaults to the `run_tabarena_experiment.py` bundled with this package
    (see `get_run_script_path`)."""
    submit_script: str | Path = field(default_factory=get_submit_script_path)
    """Path to the SLURM (array) submit script invoked by `sbatch`.
    Defaults to the `submit_template.sh` bundled with this package
    (see `get_submit_script_path`)."""

    @property
    def _workspace(self) -> Path:
        return Path(self.workspace)

    @property
    def run_script_path(self) -> str:
        """Python script to run a single benchmark experiment."""
        return str(self.run_script)

    @property
    def submit_script_path(self) -> str:
        """SLURM (array) submit script invoked by `sbatch`."""
        return str(self.submit_script)

    def get_setup_out_path(self, benchmark_name: str) -> Path:
        """Directory holding the generated job-batch artifact + SLURM job JSON."""
        return self._workspace / "setup_out" / benchmark_name

    def get_job_batch_dir(self, *, benchmark_name: str, safe_benchmark_name: str) -> str:
        """Directory of the run's ``JobBatch`` artifact (experiments + task metadata + jobs).

        This is the self-contained sweep specification the compute nodes load (see
        :class:`tabarena.benchmark.experiment.job_batch.JobBatch`); the per-array-task
        JSON items reference its experiments by name.
        """
        return str(self.get_setup_out_path(benchmark_name) / f"job_batch_{safe_benchmark_name}")

    def get_slurm_job_json_path(self, *, benchmark_name: str, safe_benchmark_name: str) -> str:
        """JSON file with the job data to run used by SLURM.
        This is generated from the configs and metadata.
        """
        return str(self.get_setup_out_path(benchmark_name) / f"slurm_run_data_{safe_benchmark_name}.json")

    def get_output_path(self, benchmark_name: str) -> str:
        """Output directory for the benchmark."""
        return str(self._workspace / "output" / benchmark_name)

    def get_slurm_log_output_path(self, benchmark_name: str) -> str:
        """Directory for the SLURM output logs."""
        return str(self._workspace / "slurm_out" / benchmark_name)

    def ensure_runtime_dirs(self, benchmark_name: str) -> None:
        """Create the output, log, and setup directories for this benchmark.

        Cache directories are not created here: the OpenML / HuggingFace / TabArena caches are
        configured via the context's ``CacheConfig`` (embedded in the ``JobBatch``) and the
        underlying libraries create their cache dirs lazily on first write.
        """
        Path(self.get_output_path(benchmark_name)).mkdir(parents=True, exist_ok=True)
        Path(self.get_slurm_log_output_path(benchmark_name)).mkdir(parents=True, exist_ok=True)
        self.get_setup_out_path(benchmark_name).mkdir(parents=True, exist_ok=True)
