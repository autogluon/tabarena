from __future__ import annotations

import shutil
from pathlib import Path

from run_plot_pareto_over_tuning_time import plot_tuning_trajectories

from tabarena.nips2025_utils.tabarena_context import TabArenaContext

if __name__ == "__main__":
    download_results: bool | str = "auto"  # results must be downloaded for the script to work
    elo_bootstrap_rounds = 200  # 1 for toy, 200 for paper
    save_path = "output_paper_results"  # folder to save all figures and tables
    use_latex: bool = False  # Set to True if you have the appropriate latex packages installed for nicer figure style

    plot_n_configs = False

    tabarena_context = TabArenaContext()
    df_results_holdout = None  # TODO: Mitra does not yet have holdout results saved in S3, need to add
    include_portfolio = False
    # df_results_holdout = tabarena_context.load_results_paper(download_results=download_results, holdout=True)

    df_results = tabarena_context.load_results_paper(download_results=download_results)

    configs_hyperparameters = tabarena_context.load_configs_hyperparameters(download=download_results)

    tabarena_context_all = TabArenaContext(
        methods=tabarena_context.method_metadata_collection.method_metadata_lst
    )

    if plot_n_configs:
        plot_tuning_trajectories(
            fig_save_dir=Path(save_path) / "n_configs",
            average_seeds=True,
        )

        plot_tuning_trajectories(
            fig_save_dir=Path(save_path) / "no_average_seeds" / "n_configs",
            average_seeds=False,
        )

    tabarena_context_all.evaluate_all(
        df_results=df_results,
        df_results_holdout=df_results_holdout,
        configs_hyperparameters=configs_hyperparameters,
        save_path=save_path,
        elo_bootstrap_rounds=elo_bootstrap_rounds,
        use_latex=use_latex,
        include_portfolio=include_portfolio,
    )

    zip_results = True
    upload_to_s3 = False
    if zip_results:
        file_prefix = f"tabarena51_paper_results"
        file_name = f"{file_prefix}.zip"
        shutil.make_archive(file_prefix, "zip", root_dir=save_path)
        if upload_to_s3:
            from autogluon.common.utils.s3_utils import upload_file
            upload_file(file_name=file_name, bucket="tabarena", prefix=save_path)
