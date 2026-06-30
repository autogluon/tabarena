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
