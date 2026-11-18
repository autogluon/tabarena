from __future__ import annotations

from pathlib import Path

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.models.kumo_cellridge.hpo import gen_kumo_cellridge

if __name__ == "__main__":
    output = Path("results/kumo-cellridge-1024-lite").resolve()
    bundle = TabArenaV0pt1ExperimentBundle(
        models=[(gen_kumo_cellridge, 0)],
        sequential_local_fold_fitting=True,
        model_artifacts_base_path=str(output / "models"),
    )
    experiments = bundle.build_experiments(num_cpus=4, num_gpus=1, time_limit=3600)
    context = TabArenaContext(backend="native")
    jobs = context.build_jobs(experiments, subset="lite")
    context.run_jobs(jobs, expname=output / "official-cache", register=False, debug_mode=True, raise_on_failure=True)
