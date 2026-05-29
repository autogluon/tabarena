"""Path configuration and bundled-script resolution for the benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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
    bundled with this package (see `get_run_script_path` /
    `get_submit_script_path`), and `openml_cache` is optional.

    The workspace is a persistent directory that all SLURM jobs can access. We
    create and use the following structure inside it:
        - WORKSPACE
            - output            -- all output data from running the benchmark
                                   (one `benchmark_name` subfolder each)
            - slurm_out         -- all SLURM output logs
                                   (one `benchmark_name` subfolder each)
            - setup_out         -- generated configs YAML + SLURM job JSON
                                   (one `benchmark_name` subfolder each)
            - .openml-cache     -- the OpenML cache (unless overridden)
    """

    workspace: str | Path
    """Persistent workspace directory that all jobs can access. Holds the
    `output/`, `slurm_out/`, and `setup_out/` folders, and (by default) the
    OpenML cache."""
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
    openml_cache: str | Path | Literal["auto"] | None = None
    """OpenML cache directory, used to store dataset and task data from OpenML.

        - If None (default), use `<workspace>/.openml-cache`.
        - If "auto", use OpenML's own default cache location.
        - Otherwise, the given path is used as a custom OpenML cache folder.
    """

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

    @property
    def openml_cache_path(self) -> str:
        """Resolved OpenML cache directory ("auto" or an absolute path)."""
        if self.openml_cache == "auto":
            return "auto"
        if self.openml_cache is None:
            return str(self._workspace / ".openml-cache")
        return str(self.openml_cache)

    def get_setup_out_path(self, benchmark_name: str) -> Path:
        """Directory holding the generated configs YAML + SLURM job JSON."""
        return self._workspace / "setup_out" / benchmark_name

    def get_configs_path(self, *, benchmark_name: str, safe_benchmark_name: str) -> str:
        """YAML file with the configs to run."""
        return str(self.get_setup_out_path(benchmark_name) / f"benchmark_configs_{safe_benchmark_name}.yaml")

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
        """Create the output, log, setup, and (optional) OpenML cache directories."""
        if self.openml_cache_path != "auto":
            Path(self.openml_cache_path).mkdir(parents=True, exist_ok=True)
        Path(self.get_output_path(benchmark_name)).mkdir(parents=True, exist_ok=True)
        Path(self.get_slurm_log_output_path(benchmark_name)).mkdir(parents=True, exist_ok=True)
        self.get_setup_out_path(benchmark_name).mkdir(parents=True, exist_ok=True)
