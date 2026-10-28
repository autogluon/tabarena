"""Reproduce the *frozen* leaderboard from the BeyondArena paper (arXiv 2606.30410).

Unlike ``examples/beyondarena/run_generate_beyondarena_leaderboard.py`` (the latest results for
every method, on the ``core2k`` protocol), this pins the method collection to
``beyond_method_metadata_2026_06_paper_collection`` -- the 11 methods with the results the paper
published -- and evaluates every split, as the paper does, so the leaderboard matches it.

Results download to ``~/.cache/tabarena/`` on first run. ``compare()`` writes the leaderboard
CSVs and figures under ``output_dir``, among them ``results_per_split.csv`` with the per-split
numbers behind the paper's per-dataset tables. ``leaderboard_to_website_format()`` reshapes the
returned DataFrame into the columns shown on the website.
"""

from __future__ import annotations

from pathlib import Path

from tabarena.contexts import BeyondArenaContext
from tabarena.contexts.beyondarena.methods import beyond_method_metadata_2026_06_paper_collection

if __name__ == "__main__":
    output_path = Path("output_leaderboard_beyondarena2026")  # folder to save all figures and tables

    context = BeyondArenaContext(
        methods=beyond_method_metadata_2026_06_paper_collection.method_metadata_lst,
    )
    leaderboard = context.compare(
        output_dir=output_path,
        subset=["all"],  # every split, as in the paper
    )
    leaderboard_website = context.leaderboard_to_website_format(
        leaderboard=leaderboard,
    )

    print("BeyondArena Paper Leaderboard:")
    print(leaderboard_website.to_markdown(index=False))
    print("")
