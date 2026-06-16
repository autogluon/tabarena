"""Benchmark a model on TabArena as an *outer* (no-bagging) experiment — shown with TabICLv2.

Same single-hub formula as ``benchmarking/run_quickstart_tabarena.py`` (bundle ->
context.run_experiments -> compare), but ``outer_experiments=True`` makes the bundle fit each
model directly on all the training data (an ``AGModelWrapper``: no train/val split, bagging, or
ensemble).
"""

from __future__ import annotations

from pathlib import Path

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.models import TabICLv2Model
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.utils.config_utils import ConfigGenerator

if __name__ == "__main__":
    # Output dirs, resolved next to this script so they don't depend on the working directory.
    here = Path(__file__).parent
    run_name = "tabiclv2"
    results_dir = str(here / "experiments" / run_name)  # the runner's `expname` (results cache)
    eval_dir = here / "eval" / run_name  # leaderboard / figures `output_dir`

    # 1: build the experiments. `outer_experiments=True` emits no-validation `AGModelWrapper`
    #    fits (no bagging) for each model. Here we run two TabICLv2 configs: the default and an
    #    `n_estimators=1` variant. A `ConfigGenerator` with explicit `manual_configs` (and 0
    #    random configs) runs exactly those — yielding `TA-TabICLv2_c1` (default) and
    #    `TA-TabICLv2_c2` (n_estimators=1). For a registry model at its default, pass the name
    #    instead, e.g. `("TabICLv2", 0)`; see `run_quickstart_tabarena.py` for HPO / custom models.
    tabiclv2 = ConfigGenerator(
        search_space={},
        model_cls=TabICLv2Model,
        manual_configs=[{}, {"n_estimators": 1}],
    )
    experiments = TabArenaV0pt1ExperimentBundle(
        models=[(tabiclv2, 0)],
        outer_experiments=True,
    ).build_experiments()

    # 2: the context is the hub. run_experiments scopes to the small datasets' first split
    #    (subset=["small", "lite"] == r0f0), runs locally, and registers the configs as in-memory
    #    methods (pre-filtering task_metadata to the tasks just run, so `compare` scopes to them
    #    with nothing extra). debug_mode=True -> in-process native backend.
    context = TabArenaContext()
    context.run_experiments(
        experiments,
        expname=results_dir,
        subset=["small", "lite"],
        new_result_prefix="[New] ",
        debug_mode=True,  # <-- also lets you attach a local debugger
    )

    # 3: compare against the cached TabArena baselines; the registered configs are picked up
    #    automatically and carried into the website-format leaderboard with their metadata.
    leaderboard = context.compare(output_dir=eval_dir)
    leaderboard_website = context.leaderboard_to_website_format(leaderboard=leaderboard)
    print("\n=== TabArena leaderboard ===")
    print(leaderboard_website.to_markdown(index=False))
    print(f"\nView saved figures in {eval_dir}")
