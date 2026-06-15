from __future__ import annotations

from pathlib import Path

from tabarena.evaluation.context.beyond_arena import BeyondArenaContext

if __name__ == "__main__":
    save_path = "output_beyondarena_leaderboard"  # folder to save all figures and tables

    tabarena_context = BeyondArenaContext()
    leaderboard = tabarena_context.compare(output_dir=Path(save_path))
    leaderboard_website = tabarena_context.leaderboard_to_website_format(leaderboard=leaderboard)

    print("Leaderboard:")
    print(leaderboard_website.to_markdown(index=False))
    print("")
