"""Reproduce the *frozen* NeurIPS 2025 leaderboard from the TabArena paper.

Unlike ``examples/plots/run_generate_main_leaderboard.py`` (which uses the latest
results for every method), this pins the method collection to
``tabarena_method_metadata_2025_06_12_collection_main`` -- the camera-ready set of
methods/results as of the paper -- so the leaderboard matches what was published.

Results download to ``~/.cache/tabarena/`` on first run. ``compare()`` writes all
figures/tables under ``output_dir``; ``leaderboard_to_website_format()`` reshapes
the returned DataFrame into the columns shown on the website.
"""

from __future__ import annotations

from pathlib import Path

from tabarena.contexts import TabArenaContext
from tabarena.contexts.tabarena.methods import (
    tabarena_method_metadata_2025_06_12_collection_main,
)

if __name__ == "__main__":
    output_path = Path("output_leaderboard_neurips2025")  # folder to save all figures and tables

    tabarena_context = TabArenaContext(
        methods=tabarena_method_metadata_2025_06_12_collection_main.method_metadata_lst,
    )
    leaderboard = tabarena_context.compare(
        output_dir=output_path,
    )
    leaderboard_website = tabarena_context.leaderboard_to_website_format(
        leaderboard=leaderboard,
    )

    print("NeurIPS 2025 Leaderboard:")
    print(leaderboard_website.to_markdown(index=False))
    print("")
