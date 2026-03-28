from __future__ import annotations

from pathlib import Path

from tabarena.nips2025_utils.artifacts._tabarena_method_metadata import tabarena_method_metadata_2025_06_12_collection_main
from tabarena.nips2025_utils.tabarena_context import TabArenaContext


if __name__ == "__main__":
    output_path = Path("output_leaderboard_neurips2025")  # folder to save all figures and tables

    tabarena_context = TabArenaContext(
        methods=tabarena_method_metadata_2025_06_12_collection_main.method_metadata_lst
    )
    leaderboard = tabarena_context.compare(
        output_dir=output_path,
    )
    leaderboard_website = tabarena_context.leaderboard_to_website_format(
        leaderboard=leaderboard,
    )

    print(f"NeurIPS 2025 Leaderboard:")
    print(leaderboard_website.to_markdown(index=False))
    print("")
