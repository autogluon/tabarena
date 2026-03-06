from __future__ import annotations

from pathlib import Path

from tabarena.nips2025_utils.tabarena_context import TabArenaContext


if __name__ == "__main__":
    save_path = "output_leaderboard"  # folder to save all figures and tables

    output_path_verified = Path(save_path) / "verified"
    output_path_unverified = Path(save_path) / "unverified"

    tabarena_context = TabArenaContext(include_unverified=False)
    leaderboard_verified = tabarena_context.compare(output_dir=output_path_verified)
    leaderboard_website_verified = tabarena_context.leaderboard_to_website_format(leaderboard=leaderboard_verified)

    tabarena_context_unverified = TabArenaContext(include_unverified=True)
    leaderboard_unverified = tabarena_context_unverified.compare(output_dir=output_path_unverified)
    leaderboard_website_unverified = tabarena_context_unverified.leaderboard_to_website_format(leaderboard=leaderboard_unverified)

    print(f"Verified Leaderboard:")
    print(leaderboard_website_verified.to_markdown(index=False))
    print("")

    print(f"Unverified Leaderboard:")
    print(leaderboard_website_unverified.to_markdown(index=False))
    print("")
