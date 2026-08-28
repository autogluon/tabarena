"""
Official BeyondArena-Lite evaluation script for Zero-Shot ISAB (ZS-ISAB).
Uses BeyondArenaExperimentBundle and BeyondArenaContext matching examples/beyondarena/run_quickstart_beyondarena.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add source directory
sys.path.insert(0, str(Path(__file__).parent / "packages" / "tabarena" / "src"))

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
from tabarena.contexts import BeyondArenaContext
from tabarena.models.zsisab.info import zsisab_info


def main():
    print("=" * 70)
    print("RUNNING OFFICIAL BEYONDARENA-LITE BENCHMARK FOR ZS-ISAB")
    print("=" * 70)

    output_dir = Path(__file__).parent / "beyondarena_lite_results"

    experiments = BeyondArenaExperimentBundle(
        models=[
            ("Linear", 0),
            (zsisab_info.search_space, 0),
        ],
    ).build_experiments()

    context = BeyondArenaContext()
    context.build_and_run_jobs(
        experiments,
        expname=str(output_dir / "cache"),
        subset=["lite"],
        new_result_prefix="[New] ",
        debug_mode=True,
    )

    print("\nGenerating BeyondArena official leaderboard...")
    leaderboard = context.compare(output_dir=output_dir / "eval")

    print("\n" + "=" * 70)
    print("OFFICIAL BEYONDARENA-LITE LEADERBOARD OUTPUT:")
    print("=" * 70)
    print(leaderboard.to_markdown())


if __name__ == "__main__":
    main()
