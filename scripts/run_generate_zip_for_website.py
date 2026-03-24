"""Collect and preprocess data for leaderboard website."""

from __future__ import annotations

import shutil
from pathlib import Path

from tabarena.website.process_artifacts_to_website import process_one_folder
from tabarena.website.process_pngs import process_png_bulk


if __name__ == "__main__":
    path_to_output_to_use = (
        Path(__file__).parent.parent / "examples" / "plots" / "output_website_artifacts"
    )
    path_copy = Path(__file__).parent / "clean_website_artifacts"

    file_paths = path_to_output_to_use.glob("**/website_leaderboard.csv")

    for path in file_paths:
        base_input_path = Path(path).parent
        base_output_path = path_copy / base_input_path.relative_to(
            path_to_output_to_use
        )
        process_one_folder(
            base_input_path=Path(path).parent,
            base_output_path=base_output_path,
        )
        process_png_bulk(path=base_output_path)

    shutil.make_archive(
        "clean_website_artifacts",
        "zip",
        root_dir="clean_website_artifacts/website_data",
    )
