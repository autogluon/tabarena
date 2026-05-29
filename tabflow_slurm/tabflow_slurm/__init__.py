"""TabArena SLURM benchmark setup.

The benchmark setup building blocks live in the `setup` subpackage and are
re-exported here for convenience. Compose a run from them and call
`TabArenaBenchmarkSetup(...).setup_jobs()` to generate the configs YAML, the
SLURM job JSON, and the `sbatch` command(s) to launch.

The runner script and SLURM submit template are bundled with this package;
`get_run_script_path()` / `get_submit_script_path()` return their installed
locations (these are also the defaults used by `PathSetup`).
"""

from __future__ import annotations

from tabflow_slurm.setup import (
    JobCandidate,
    ModelConstraints,
    ModelPipelinesToRunSetup,
    PathSetup,
    ResourcesSetup,
    SchedulerSetup,
    SlurmSetup,
    TabArenaBenchmarkSetup,
    get_run_script_path,
    get_submit_script_path,
    should_run_job,
    should_run_job_batch,
)

__all__ = [
    "JobCandidate",
    "ModelConstraints",
    "ModelPipelinesToRunSetup",
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
