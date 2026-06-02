"""Collect and preprocess data for leaderboard website."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


def process_one_folder(
        *, base_input_path: Path, base_output_path: Path,
):
    base_output_path.mkdir(parents=True, exist_ok=True)

    figure_file_type = "png"

    # N datasets file
    n_datasets = len(
        pd.read_csv(base_input_path / "results_per_split.csv", low_memory=False)["dataset"].unique()
    )
    (base_output_path / f"n_datasets_{n_datasets}").touch()

    for file_name in [
        "website_leaderboard.csv",
    ]:
        shutil.copy(
            base_input_path / file_name,
            base_output_path / file_name,
            )

    # Copy plots
    for fig_path in [
        f"tuning-impact-elo.{figure_file_type}",
        f"pareto_front_improvability_vs_time_infer.{figure_file_type}",
        f"winrate_matrix.{figure_file_type}",
        (
                Path("tuning_trajectories")
                / "placeholder_name",
                f"pareto_n_configs_imp.{figure_file_type}",
        ),
    ]:
        # FIXME: cannot use this on my cluster as I am not able to install poppler.
        #   Hence, LB code needs to create zips.
        # import zipfile
        # from pdf2image import convert_from_path
        # pdf_path = base_input_path / fig_path
        # zip_path = (base_output_path / fig_path).with_suffix(".png.zip")
        # png_path = zip_path.with_suffix(".png")
        # # PDF to PNG
        # images = convert_from_path(str(pdf_path), dpi=800)
        # images[0].save(png_path, "PNG")
        # # PNG to ZIP
        # with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        #     zipf.write(png_path, arcname=png_path.name)
        # png_path.unlink(missing_ok=True)

        # Copy files
        if isinstance(fig_path, tuple):
            shutil.copy(
                base_input_path / fig_path[0] / fig_path[1],
                base_output_path / fig_path[1],
                )
        else:
            shutil.copy(
                base_input_path / fig_path,
                base_output_path / fig_path,
                )
