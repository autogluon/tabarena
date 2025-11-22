from __future__ import annotations

import shutil
from pathlib import Path

from autogluon.common import TabularDataset
from run_plot_pareto_over_tuning_time import plot_tuning_trajectories_all
from tabarena.nips2025_utils.tabarena_context import TabArenaContext

if __name__ == "__main__":
    elo_bootstrap_rounds = 200  # 1 for toy, 200 for official
    save_path = "output_website_artifacts"  # folder to save all figures and tables

    # Set to True if you have the appropriate latex packages installed for nicer figure style
    # TODO: use_latex=True makes some of the plots look worse, but makes the tuning-impact-elo figures look better.
    #  To avoid needing 2 people to compute this properly, we should selectively disable `use_latex` for certain figures.
    #  Alternatively, make all figures without `use_latex` look nice, and drop the latex logic entirely.
    use_latex: bool = False
    download_results = False  # Set to False to avoid re-download

    include_unverified = True
    run_ablations = True

    tabarena_context = TabArenaContext(include_unverified=include_unverified)
    tabarena_context.load_results_paper(download_results=download_results)

    method_metadata_info = tabarena_context.method_metadata_collection.info()
    method_metadata_info = method_metadata_info.rename(
        columns={
            "method": "ta_name",
            "artifact_name": "ta_suite",
        }
    )
    TabularDataset.save(
        path=Path(save_path) / "method_metadata_info.csv", df=method_metadata_info
    )

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
    )

    plot_tuning_trajectories_all(
        tabarena_context=tabarena_context,
        fig_save_dir=save_path,
        ban_bad_methods=True,
    )

    zip_results = True
    upload_to_s3 = False
    if zip_results:
        file_prefix = "tabarena_website_artifacts"
        file_name = f"{file_prefix}.zip"
        shutil.make_archive(file_prefix, "zip", root_dir=save_path)
        if upload_to_s3:
            from autogluon.common.utils.s3_utils import upload_file

            upload_file(file_name=file_name, bucket="tabarena", prefix=save_path)
