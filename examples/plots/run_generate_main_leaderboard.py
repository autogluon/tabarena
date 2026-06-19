"""Generate TabArena's main (living) leaderboard.

Builds the leaderboard from the *latest* results for every benchmarked method
(the default ``TabArenaContext()`` method collection, which grows as new methods
and results land). Use this to see where methods stand on the current TabArena
suite.

For the *frozen* NeurIPS 2025 leaderboard from the paper, see
``examples/reproducibility/run_generate_main_leaderboard_neurips2025.py``.

Results download to ``~/.cache/tabarena/`` on first run. ``compare()`` writes all
figures/tables under ``output_dir`` and returns the leaderboard DataFrame;
``leaderboard_to_website_format()`` reshapes it into the columns shown on the
website.
"""

from __future__ import annotations

from pathlib import Path

from tabarena.nips2025_utils.tabarena_context import TabArenaContext

if __name__ == "__main__":
    save_path = "output_leaderboard"  # folder to save all figures and tables

    tabarena_context = TabArenaContext()
    leaderboard = tabarena_context.compare(output_dir=Path(save_path))
    leaderboard_website = tabarena_context.leaderboard_to_website_format(leaderboard=leaderboard)

    print("Leaderboard:")
    print(leaderboard_website.to_markdown(index=False))
    print("")
