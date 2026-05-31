"""TabArena SLURM benchmark setup.

The benchmark building blocks live in the `setup` subpackage and are re-exported
here for convenience. `TabArenaBenchmarkPlan` is the single public entry point:
compose it from the building blocks (`PathSetup`, `SlurmSetup`, `ResourcesSetup`,
a metadata bundle, an experiment bundle) plus a list of `ModelJob`s, then call
`.setup_jobs()` to generate the configs YAML, the SLURM job JSON, and the
`sbatch` command(s) to launch.

The runner script and SLURM submit template are bundled with this package;
`get_run_script_path()` / `get_submit_script_path()` return their installed
locations (these are also the defaults used by `PathSetup`).
"""

from __future__ import annotations

from tabflow_slurm.setup import (
    BeyondArenaResourcesSetup,
    GCPSlurmSetup,
    JobCandidate,
    ModelJob,
    PathSetup,
    ResourcesSetup,
    SchedulerSetup,
    SingleModel,
    SlurmSetup,
    TabArenaBenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
    get_run_script_path,
    get_submit_script_path,
    should_run_job,
    should_run_job_batch,
)

__all__ = [
    "BeyondArenaResourcesSetup",
    "GCPSlurmSetup",
    "JobCandidate",
    "ModelJob",
    "PathSetup",
    "ResourcesSetup",
    "SchedulerSetup",
    "SingleModel",
    "SlurmSetup",
    "TabArenaBenchmarkPlan",
    "TabArenaV0pt1ResourcesSetup",
    "get_run_script_path",
    "get_submit_script_path",
    "should_run_job",
    "should_run_job_batch",
]
