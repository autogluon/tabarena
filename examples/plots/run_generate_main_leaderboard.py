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
