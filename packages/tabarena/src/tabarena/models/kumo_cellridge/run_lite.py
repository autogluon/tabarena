from __future__ import annotations

import argparse
from pathlib import Path

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.models.kumo_cellridge.hpo import gen_kumo_cellridge
from tabarena.models.kumo_cellridge.model import KumoCellRidgeModel
from tabarena.utils.config_utils import ConfigGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the CellRidge checkpoint on TabArena-Lite.")
    parser.add_argument("--max-context-size", type=int, help="Override the released 200,000-row support cap.")
    args = parser.parse_args()
    generator = gen_kumo_cellridge
    limit = args.max_context_size
    if limit is not None:
        if limit <= 0:
            parser.error("--max-context-size must be positive")
        generator = ConfigGenerator(
            model_cls=KumoCellRidgeModel,
            search_space={},
            manual_configs=[{"max_context_size": limit}],
        )
    output = Path(f"results/kumo-cellridge-1024-s{limit or KumoCellRidgeModel.default_max_context_size}-lite").resolve()
    bundle = TabArenaV0pt1ExperimentBundle(
        models=[(generator, 0)],
        sequential_local_fold_fitting=True,
        model_artifacts_base_path=str(output / "models"),
    )
    experiments = bundle.build_experiments(num_cpus=4, num_gpus=1, time_limit=3600)
    context = TabArenaContext(backend="native")
    jobs = context.build_jobs(experiments, subset="lite")
    context.run_jobs(jobs, expname=output / "official-cache", register=False, debug_mode=True, raise_on_failure=True)
