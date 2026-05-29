"""Benchmark setup building blocks.

Compose a benchmark run from the dataclasses below and call
`TabArenaBenchmarkSetup(...).setup_jobs()` to generate the configs YAML, the
SLURM job JSON, and the `sbatch` command(s) to launch.

The runner script and SLURM submit template are bundled with the parent
package; `get_run_script_path()` / `get_submit_script_path()` return their
installed locations (these are also the defaults used by `PathSetup`).
"""

from __future__ import annotations

from tabflow_slurm.setup.benchmark import TabArenaBenchmarkSetup
from tabflow_slurm.setup.candidates import JobCandidate, should_run_job, should_run_job_batch
from tabflow_slurm.setup.paths import PathSetup, get_run_script_path, get_submit_script_path
from tabflow_slurm.setup.resources import ResourcesSetup
from tabflow_slurm.setup.scheduler import SchedulerSetup, SlurmSetup

__all__ = [
    "JobCandidate",
    "PathSetup",
    "ResourcesSetup",
    "SchedulerSetup",
    "SlurmSetup",
    "TabArenaBenchmarkSetup",
    "get_run_script_path",
    "get_submit_script_path",
    "should_run_job",
    "should_run_job_batch",
]
