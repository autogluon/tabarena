"""TabArena-Lite: first split of every dataset, official 8-fold bagging.

Default-only first (recommended next step after the 3-dataset smoke)::

    python examples/benchmarking/run_lite_prismboost.py --n-configs 0

Author-submission Lite (default + 25 random HPO configs)::

    python examples/benchmarking/run_lite_prismboost.py --n-configs 25

Results land in ``experiments/lite_prismboost/``; leave that folder intact for the PR.
Flip ``--debug`` on if Ray workers fail to import the model class.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.models.prismboost.model import PrismBoostModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-configs",
        type=int,
        default=0,
        help="Extra random HPO configs on top of the default (0 = default only; 25 = Lite submission).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="In-process sequential backend (no Ray). Slower, easier to debug.",
    )
    args = parser.parse_args()

    here = Path(__file__).parent
    run_name = "lite_prismboost"
    results_dir = str(here / "experiments" / run_name)
    eval_dir = here / "eval" / run_name

    experiments = TabArenaV0pt1ExperimentBundle(
        models=[(PrismBoostModel.config_generator(), args.n_configs)],
    ).build_experiments()

    context = TabArenaContext()
    context.build_and_run_jobs(
        experiments,
        expname=results_dir,
        subset="lite",
        debug_mode=args.debug,
    )

    leaderboard = context.compare(output_dir=eval_dir)
    leaderboard_website = context.leaderboard_to_website_format(leaderboard=leaderboard)
    print("\n=== TabArena-Lite leaderboard (website format) ===")
    print(leaderboard_website.to_markdown(index=False))
    print(f"\nResults: {results_dir}")
    print(f"Figures: {eval_dir}")


if __name__ == "__main__":
    main()
