from __future__ import annotations

import shutil
from pathlib import Path

from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from run_plot_pareto_over_tuning_time import plot_tuning_trajectories


if __name__ == "__main__":
    elo_bootstrap_rounds = 200  # 1 for toy, 200 for official
    save_path = "output_website_artifacts"  # folder to save all figures and tables

    # Set to True if you have the appropriate latex packages installed for nicer figure style
    # TODO: use_latex=True makes some of the plots look worse, but makes the tuning-impact-elo figures look better.
    #  To avoid needing 2 people to compute this properly, we should selectively disable `use_latex` for certain figures.
    #  Alternatively, make all figures without `use_latex` look nice, and drop the latex logic entirely.
    use_latex: bool = False

    include_unverified = True
    run_ablations = True

    tabarena_context = TabArenaContext(include_unverified=include_unverified)

    if run_ablations:
        tabarena_context.plot_runtime_per_method(
            save_path=Path(save_path) / "ablation" / "all-runtimes",
        )

        tabarena_context.generate_per_dataset_tables(
            save_path=Path(save_path) / "per_dataset",
        )

        # extend this to get tuning_trajectories for different subsets
        subset_map = {
            "all": [],
            "medium": ["medium"],
            "small": ["small"],
            "tabpfn": ["tabpfn"],
        }

        # will take a few minutes
        plot_tuning_trajectories(
            tabarena_context=tabarena_context,
            subset_map=subset_map,
            fig_save_dir=Path(save_path) / "tuning_trajectories",
            average_seeds=False,
            exclude_imputed=True,
            ban_bad_methods=True,
        )

    tabarena_context.evaluate_all(
        save_path=save_path,
        elo_bootstrap_rounds=elo_bootstrap_rounds,
        use_latex=use_latex,
    )

    zip_results = True
    upload_to_s3 = False
    if zip_results:
        file_prefix = f"tabarena_website_artifacts"
        file_name = f"{file_prefix}.zip"
        shutil.make_archive(file_prefix, "zip", root_dir=save_path)
        if upload_to_s3:
            from autogluon.common.utils.s3_utils import upload_file
            upload_file(file_name=file_name, bucket="tabarena", prefix=save_path)
