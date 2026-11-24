"""Run Tabra Blends on the first official split of each TabArena dataset."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabarena.contexts import TabArenaContext
from tabarena.systems.tabra_blends import gen_tabra_blends


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Use the small Lite datasets.")
    parser.add_argument("--output", type=Path, default=Path("results/tabra-blends-lite"))
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--memory-gb", type=float, default=110)
    args = parser.parse_args()

    experiments = TabArenaV0pt1ExperimentBundle(
        models=[(gen_tabra_blends, 0)],
        system_experiments=True,
    ).build_experiments(time_limit=3600, num_cpus=args.cpus, num_gpus=1, memory_limit=args.memory_gb)

    context = TabArenaContext()
    scope = TaskSubset(subset=["small", "lite"] if args.smoke else "lite")
    output = args.output.resolve()
    context.build_and_run_jobs(experiments, expname=output / "raw", task_subset=scope, debug_mode=True)
    leaderboard = context.compare(output_dir=output / "evaluation", subset=scope.subset, plot=False)
    print(context.leaderboard_to_website_format(leaderboard=leaderboard).to_markdown(index=False))


if __name__ == "__main__":
    main()
