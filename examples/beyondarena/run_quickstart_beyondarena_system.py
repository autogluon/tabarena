"""Quickstart: benchmark a *system* (like AutoGluon) on a small slice of BeyondArena and compare to the cached baselines.

A system owns its whole pipeline: preprocessing, validation, tuning and ensembling, all inside the
budget TabArena hands it. This is the BeyondArena counterpart of
``examples/benchmarking/run_quickstart_tabarena_system.py``, which also shows how to write your own
``ExternalSystemModel``. Here we run the AutoGluon wrapper that ships with TabArena
(``tabarena.systems.AutoGluonSystemModel``) instead. If your method is one model that TabArena
should tune under its shared protocol, see ``run_quickstart_beyondarena_model.py`` next to this file.

The public BeyondArena leaderboard lists models only for now, so a system is compared against the
cached model baselines.
"""

from __future__ import annotations

from pathlib import Path

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
from tabarena.contexts import BeyondArenaContext
from tabarena.systems import AutoGluonSystemModel
from tabarena.utils.config_utils import SystemConfigGenerator

if __name__ == "__main__":
    # Output dirs, resolved next to this script so they don't depend on the working directory.
    here = Path(__file__).parent
    run_name = "beyondarena_new_system"
    results_dir = str(here / "experiments" / run_name)  # the runner's `expname`
    eval_dir = here / "eval" / run_name  # leaderboard `output_dir`

    # 1: suite metadata -> filter. `core` is the recommended protocol for every BeyondArena run
    #    (see `run_quickstart_beyondarena_model.py` for the full predicate list). A system fits a
    #    whole AutoGluon predictor per split, so this quickstart also narrows to `lite` (the first
    #    split of each dataset) and to tiny, low-dimensional datasets to stay fast.
    subset = ["core", "lite", "tiny", "!high-dim"]

    # 2: build the system experiments. A `SystemConfigGenerator` pairs the system's class with a
    #    `name` (a system has no AutoGluon registry name) and its configs, one benchmarked variant
    #    each; `system_experiments=True` emits one no-validation experiment per config. The
    #    `time_limit` (seconds) is forwarded to the system as its fit budget.
    generator = SystemConfigGenerator(
        model_cls=AutoGluonSystemModel,
        name="DemoAutoGluonSystem",
        manual_configs=[{"preset": "medium_quality"}],
    )
    experiments = BeyondArenaExperimentBundle(
        models=[(generator, 0)],
        system_experiments=True,
    ).build_experiments(time_limit=30)

    # 3: build_and_run_jobs scopes the context's BeyondArena task metadata to `subset`, pairs the
    #    config with each split, materializes the selected tasks, runs the system locally, and
    #    registers the results as an in-memory method.
    context = BeyondArenaContext()
    context.build_and_run_jobs(
        experiments,
        expname=results_dir,
        subset=subset,
        new_result_prefix="[New] ",
        debug_mode=True,  # <-- also lets you attach a local debugger
    )

    # 4: compare against the cached BeyondArena baselines; the registered method is picked up
    #    automatically and the leaderboard is scoped to the tasks just run.
    leaderboard = context.compare(output_dir=eval_dir)
    print(leaderboard.to_markdown())
