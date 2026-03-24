from __future__ import annotations

import shutil
from pathlib import Path

from run_plot_pareto_over_tuning_time import plot_tuning_trajectories_all
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.website.process_artifacts_to_website import process_one_folder
from tabarena.website.process_pngs import process_png_bulk


def generate_website_artifacts(output_path: str | Path):
    elo_bootstrap_rounds = 200  # 1 for toy, 200 for official
    save_path = output_path  # folder to save all figures and tables

    # Set to True if you have the appropriate latex packages installed for nicer figure style
    # TODO: use_latex=True makes some of the plots look worse, but makes the tuning-impact-elo figures look better.
    #  To avoid needing 2 people to compute this properly, we should selectively disable `use_latex` for certain figures.
    #  Alternatively, make all figures without `use_latex` look nice, and drop the latex logic entirely.
    use_latex: bool = False
    download_results = "auto"  # Set to False to avoid re-download

    include_unverified = True
    run_ablations = False
    figure_file_type = "png"

    tabarena_context = TabArenaContext(include_unverified=include_unverified)
    tabarena_context.load_results_paper(download_results=download_results)

    evaluator_kwargs = {
        "figure_file_type": figure_file_type,
    }
    file_ext = f".{figure_file_type}"

    if run_ablations:
        tabarena_context.plot_runtime_per_method(
            save_path=Path(save_path) / "ablation" / "all-runtimes",
        )

        tabarena_context.generate_per_dataset_tables(
            save_path=Path(save_path) / "per_dataset",
        )

    tabarena_context.evaluate_all(
        save_path=save_path,
        elo_bootstrap_rounds=elo_bootstrap_rounds,
        use_latex=use_latex,
        use_website_folder_names=True,
        evaluator_kwargs=evaluator_kwargs,
    )

    plot_tuning_trajectories_all(
        tabarena_context=tabarena_context,
        fig_save_dir=save_path,
        ban_bad_methods=True,
        file_ext=file_ext,
    )

    zip_results = True
    if zip_results:
        file_prefix = "tabarena_website_artifacts"
        shutil.make_archive(file_prefix, "zip", root_dir=save_path)


def convert_to_website_format(input_path: str | Path, output_path: str | Path):
    path_to_output_to_use = Path(input_path)
    path_copy = Path(output_path)

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
        root_dir="clean_website_artifacts/data",
    )


if __name__ == "__main__":
    raw_artifacts_save_path = "output_website_artifacts"
    clean_artifacts_save_path = "clean_website_artifacts"

    # 1. Generate 'output_website_artifacts' folder
    generate_website_artifacts(output_path=raw_artifacts_save_path)

    # 2. Generate 'clean_website_artifacts' folder
    convert_to_website_format(input_path=raw_artifacts_save_path, output_path=clean_artifacts_save_path)

    # 3. Manually copy/paste the 'clean_website_artifacts/data/'
    # folder contents into the HuggingFace Space 'data/' directory.

    # 4. Commit all HuggingFace Space file changes and push.
