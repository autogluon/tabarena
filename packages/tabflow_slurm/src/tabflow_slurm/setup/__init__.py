"""Benchmark setup building blocks.

`TabArenaBenchmarkPlan` is the single public entry point: compose it from the
building blocks below (`PathSetup`, `SlurmSetup`, `ResourcesSetup`, an arena
context, an experiment bundle) plus a list of `ModelJob`s, then
call `.setup_jobs()` to generate the `JobBatch` artifact, the SLURM job JSON,
and the `sbatch` command(s) to launch. The per-run engine
(`tabflow_slurm.setup.benchmark.TabArenaBenchmarkSetup`) is internal — the plan
builds and drives it for you.

The runner script and SLURM submit template are bundled with the parent
package; `get_run_script_path()` / `get_submit_script_path()` return their
installed locations (these are also the defaults used by `PathSetup`).
"""

from __future__ import annotations

from tabflow_slurm.setup.paths import PathSetup, get_run_script_path, get_submit_script_path
from tabflow_slurm.setup.plan import ModelJob, SingleModel, TabArenaBenchmarkPlan
from tabflow_slurm.setup.resources import BeyondArenaResourcesSetup, ResourcesSetup, TabArenaV0pt1ResourcesSetup
from tabflow_slurm.setup.scheduler import GCPSlurmSetup, LocalSequentialSetup, SchedulerSetup, SlurmSetup

__all__ = [
    "BeyondArenaResourcesSetup",
    "GCPSlurmSetup",
    "LocalSequentialSetup",
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
]
