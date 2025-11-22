"""Collect and preprocess data for leaderboard website."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from bencheval.website_format import format_leaderboard


def process_one_folder(
    *, base_input_path: Path, base_output_path: Path, method_metadata_info
):
    # LB CSV
    path_to_lb = base_input_path / "tabarena_leaderboard.csv"
    path_to_website_lb = base_output_path / "website_leaderboard.csv"
    lb = pd.read_csv(path_to_lb)
    leaderboard_website_verified = format_leaderboard(
        df_leaderboard=lb, include_type=True, method_metadata_info=method_metadata_info
    )
    leaderboard_website_verified.to_csv(path_to_website_lb, index=False)

    # N datasets file
    n_datasets = len(
        pd.read_csv(base_input_path / "results_per_split.csv")["dataset"].unique()
    )
    (base_output_path / f"n_datasets_{n_datasets}").touch()

    # Copy plots
    for fig_path in [
        "tuning-impact-elo.pdf",
        "pareto_front_improvability_vs_time_infer.pdf",
        "winrate_matrix.pdf",
        (
            Path("tuning_trajectories")
            / "placeholder_name",
            "pareto_n_configs_imp.pdf",
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


if __name__ == "__main__":
    path_to_output_to_use = (
        Path(__file__).parent.parent / "examples" / "plots" / "output_website_artifacts"
    )
    path_copy = Path(__file__).parent / "clean_website_artifacts"

    file_paths = path_to_output_to_use.glob("**/tabarena_leaderboard.csv")
    method_metadata_info = pd.read_csv(
        path_to_output_to_use / "method_metadata_info.csv"
    )

    for path in file_paths:
        base_input_path = Path(path).parent
        base_output_path = path_copy / base_input_path.relative_to(
            path_to_output_to_use
        )
        base_output_path.mkdir(parents=True, exist_ok=True)
        process_one_folder(
            base_input_path=Path(path).parent,
            base_output_path=base_output_path,
            method_metadata_info=method_metadata_info,
        )

    shutil.make_archive(
        "clean_website_artifacts",
        "zip",
        root_dir="clean_website_artifacts/website_data",
    )
