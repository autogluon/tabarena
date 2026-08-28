"""
Official TabArena-Full evaluation script for Zero-Shot ISAB (ZS-ISAB).
Uses TabArenaV0pt1ExperimentBundle and TabArenaContext with subset="all" across all 51 curated datasets.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add source directory
sys.path.insert(0, str(Path(__file__).parent / "packages" / "tabarena" / "src"))

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.models.zsisab.info import zsisab_info


def main():
    print("=" * 70)
    print("RUNNING OFFICIAL TABARENA-FULL BENCHMARK FOR ZS-ISAB (ALL 51 DATASETS)")
    print("=" * 70)

    output_dir = Path(__file__).parent / "tabarena_full_results"

    # Bundle ZS-ISAB default zero-shot config
    experiments = TabArenaV0pt1ExperimentBundle(
        models=[
            (zsisab_info.search_space, 0),
        ],
    ).build_experiments()

    context = TabArenaContext()
    context.build_and_run_jobs(
        experiments,
        expname=str(output_dir / "cache"),
        subset="all",  # Full 51 curated dataset suite
        new_result_prefix="[New] ",
        debug_mode=True,
    )

    print("\nGenerating TabArena-Full official leaderboard...")
    leaderboard = context.compare(output_dir=output_dir / "eval")
    website_lb = context.leaderboard_to_website_format(leaderboard=leaderboard)

    print("\n" + "=" * 70)
    print("OFFICIAL TABARENA-FULL LEADERBOARD OUTPUT:")
    print("=" * 70)
    print(website_lb.to_markdown(index=False))

    # Save Markdown and CSV outputs matching ChimeraBoost packaging
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "tabarena_full_leaderboard.md", "w") as f:
        f.write(website_lb.to_markdown(index=False))
    website_lb.to_csv(output_dir / "tabarena_full_leaderboard.csv", index=False)
    print(f"\nSaved leaderboard outputs to {output_dir}")


if __name__ == "__main__":
    main()
