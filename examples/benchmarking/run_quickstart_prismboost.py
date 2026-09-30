"""Quickstart: benchmark PrismBoost on three TabArena-Lite datasets.

The model class lives in ``tabarena.models.prismboost`` (an importable module)
so Ray workers can unpickle it when ``debug_mode=False``. This script keeps
``debug_mode=True`` for a local in-process smoke run.

After this works, run TabArena-Lite on all datasets with HPO::

    context.build_and_run_jobs(
        experiments,
        expname=results_dir,
        subset="lite",
        new_result_prefix="[New] ",
        debug_mode=False,
    )
    # and pass ``(PrismBoostModel.config_generator(), 25)`` for default + ~25 configs.
"""

from __future__ import annotations

from pathlib import Path

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.models.prismboost.model import PrismBoostModel

DATASETS = ["blood-transfusion-service-center", "QSAR_fish_toxicity", "anneal"]


if __name__ == "__main__":
    here = Path(__file__).parent
    run_name = "quickstart_prismboost"
    results_dir = str(here / "experiments" / run_name)
    eval_dir = here / "eval" / run_name

    experiments = TabArenaV0pt1ExperimentBundle(
        models=[
            (PrismBoostModel.config_generator(), 1),
        ],
    ).build_experiments()

    context = TabArenaContext()
    context.build_and_run_jobs(
        experiments,
        expname=results_dir,
        subset="lite",
        build_kwargs={"dataset_names": DATASETS},
        new_result_prefix="[New] ",
        debug_mode=True,
    )

    leaderboard = context.compare(output_dir=eval_dir)
    leaderboard_website = context.leaderboard_to_website_format(leaderboard=leaderboard)
    print("\n=== TabArena leaderboard (website format) ===")
    print(leaderboard_website.to_markdown(index=False))
    print(f"\nView saved figures in {eval_dir}")
