"""Generate the official BeyondArena leaderboard from the cached results.

Results download to ``~/.cache/tabarena/`` on first run. ``compare()`` returns the leaderboard
DataFrame and writes the leaderboard CSVs and figures under ``output_dir``, among them
``results_per_split.csv``: one row per dataset, split and method with the test and validation
error, the train and inference time, and an ``imputed`` flag. BeyondArena's per-split results
are not hosted anywhere else; this script writes them (pass ``plot=False`` to ``compare`` to
skip the figures). ``leaderboard_to_website_format()`` reshapes the leaderboard into the
website's columns.
"""

from __future__ import annotations

from tabarena.contexts import BeyondArenaContext

if __name__ == "__main__":
    output_dir = "output_beyondarena_leaderboard"  # folder to save all figures and tables

    context = BeyondArenaContext()
    # `core` is BeyondArena's recommended, default evaluation protocol: each dataset's first
    # `folds_to_use` splits — already enough for stable rankings, so there is no need to evaluate
    # the full split set (`subset=["all"]`). This is what the official BeyondArena leaderboard uses.
    leaderboard = context.compare(output_dir=output_dir, subset=["core"])
    leaderboard_website = context.leaderboard_to_website_format(leaderboard=leaderboard)

    print("Leaderboard:")
    print(leaderboard_website.to_markdown(index=False))
