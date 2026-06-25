from __future__ import annotations

from tabarena.contexts.beyond_arena import BeyondArenaContext

if __name__ == "__main__":
    output_dir = "output_beyondarena_leaderboard"  # folder to save all figures and tables

    context = BeyondArenaContext()
    leaderboard = context.compare(output_dir=output_dir)
    leaderboard_website = context.leaderboard_to_website_format(leaderboard=leaderboard)

    print("Leaderboard:")
    print(leaderboard_website.to_markdown(index=False))
