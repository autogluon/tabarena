"""Generate TabArena's main (living) leaderboard.

Builds the leaderboard from the *latest* results for every benchmarked method
(the default ``TabArenaContext()`` method collection, which grows as new methods
and results land). Use this to see where methods stand on the current TabArena
suite.

For the *frozen* NeurIPS 2025 leaderboard from the paper, see
``examples/reproducibility/run_generate_main_leaderboard_neurips2025.py``.

Results download to ``~/.cache/tabarena/`` on first run. ``compare()`` returns the
leaderboard DataFrame and writes the leaderboard CSVs and figures under ``output_dir``.
One of those CSVs is ``results_per_split.csv``, the numbers the leaderboard aggregates:
one row per dataset, split and method with the test and validation error, the train and
inference time, and an ``imputed`` flag. The per-split results are not hosted anywhere;
this script writes them, and ``run_export_results_per_split.py`` next to it writes only
the CSVs and skips the figures. ``leaderboard_to_website_format()`` reshapes the
leaderboard into the columns shown on the website.
"""

from __future__ import annotations

from pathlib import Path

from tabarena.contexts import TabArenaContext

if __name__ == "__main__":
    save_path = "output_leaderboard"  # folder for the leaderboard CSVs (incl. results_per_split.csv) and figures

    tabarena_context = TabArenaContext()
    leaderboard = tabarena_context.compare(output_dir=Path(save_path))
    leaderboard_website = tabarena_context.leaderboard_to_website_format(leaderboard=leaderboard)

    print("Leaderboard:")
    print(leaderboard_website.to_markdown(index=False))
    print("")
