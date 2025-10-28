from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.tabarena.tabarena import TabArena
from autogluon.common.loaders import load_pd

from tabarena.paper.paper_utils import get_method_rename_map
from tabarena.nips2025_utils.artifacts._tabarena_method_metadata import tabarena_method_metadata_2025_06_12_collection_main
from tabarena.plot.plot_pareto_frontier import plot_optimal_arrow


def plot_hpo(
    df: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    save_path: str | Path,
    max_Y: bool = True,
    max_X: bool = False,
    method_col: str = "name",
    xlog: bool = True,
    color_by_rank: bool = True,
    sort_col: str | None = None,
    method_order: list[str] | None = None,
    optimal_arrow: bool = True,
):
    """
    Plot HPO trajectories for multiple methods.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing results.
    xlabel : str
        Column name for x-axis (e.g. training time).
    ylabel : str
        Column name for y-axis (e.g. validation score).
    save_path : str | Path
        Path to save figure.
    max_Y : bool, default=True
        Whether higher y-values are better.
    max_X : bool, default=False
        Whether higher x-values are better.
    method_col : str, default='name'
        Column identifying each method.
    xlog : bool, default=True
        Whether to use log scale for x-axis.
    color_by_rank : bool, default=True
        Whether to color methods by rank.
    sort_col : str | None, default=None
        If provided, sorts each method’s points by this numeric column (ascending),
        and highlights the point with the highest value of this column using a different marker.
    """
    # Build a 60-color palette from tab20 / tab20b / tab20c
    colors60 = (
        list(sns.color_palette("tab20", 20))
        + list(sns.color_palette("tab20b", 20))
        + list(sns.color_palette("tab20c", 20))
    )

    method_names = list(df[method_col].unique())

    # Determine peak per method (max if max_Y else min)
    if max_Y:
        peak_per_method = {m: df.loc[df[method_col] == m, ylabel].max() for m in method_names}
    else:
        peak_per_method = {m: df.loc[df[method_col] == m, ylabel].min() for m in method_names}

    # Sort by peak and create a stable color map (alphabetical)
    sorted_methods = sorted(method_names, key=lambda m: peak_per_method[m], reverse=max_Y)
    base_methods_for_colors = sorted_methods if color_by_rank else sorted(method_names)

    if method_order:
        sorted_methods = method_order + [m for m in sorted_methods if m not in method_order]
        base_methods_for_colors = method_order + [m for m in base_methods_for_colors if m not in method_order]
    color_map = {m: colors60[i % len(colors60)] for i, m in enumerate(base_methods_for_colors)}

    fig, ax = plt.subplots(figsize=(6, 4.5))
    if xlog:
        ax.set_xscale("log")

    if optimal_arrow:
        plot_optimal_arrow(ax=ax, max_X=max_X, max_Y=max_Y, size=0.45, scale=1.2)

    handles = []
    labels = []

    for method_name in sorted_methods:
        df_method = df[df[method_col] == method_name].copy()
        if df_method.empty:
            continue

        # --- Sort by sort_col if provided ---
        max_sort_pos = None
        if sort_col is not None and sort_col in df_method.columns:
            df_method = df_method.sort_values(by=sort_col, ascending=True)
            # position (0..n-1) of the row with the max sort_col
            max_sort_pos = int(df_method[sort_col].to_numpy().argmax())

        times = df_method[xlabel].to_numpy()
        scores = df_method[ylabel].to_numpy()
        color = color_map[method_name]

        # 1) Draw the connecting line (no markers)
        h, = ax.plot(
            times,
            scores,
            "-",                # no point markers
            label=method_name,
            color=color,
            alpha=0.9,
            linewidth=1.5,
        )

        # 2) Draw circle markers for all-but-the-max (if sort_col used)
        if max_sort_pos is not None:
            mask = np.ones(len(df_method), dtype=bool)
            mask[max_sort_pos] = False
            if mask.any():
                ax.scatter(
                    times[mask],
                    scores[mask],
                    marker="o",
                    s=64,          # ~markersize=16 equivalent
                    color=color,
                    alpha=0.9,
                    zorder=4,
                )
            # 3) Draw the max point bolded (single marker)
            ax.scatter(
                times[max_sort_pos],
                scores[max_sort_pos],
                marker="o",
                s=96,
                color=color,
                edgecolor="black",
                linewidth=1.3,
                alpha=0.9,
                zorder=5,
            )
        else:
            # Back-compat: no sort_col → keep circles for all points
            ax.scatter(
                times,
                scores,
                marker="o",
                s=64,
                color=color,
                alpha=0.9,
                zorder=4,
            )
        points_legend = ax.scatter([], [], marker="o", s=64, color=color, alpha=0.9)

        handles.append(points_legend)
        labels.append(method_name)

    # Flip legend order only if max_Y is False
    legend_fontsize = 9
    if max_Y:
        handles_legend = handles
        labels_legend = labels
    else:
        handles_legend = handles[::-1]
        labels_legend = labels[::-1]

    ax.legend(
        handles_legend,
        labels_legend,
        fontsize=legend_fontsize,
        ncol=1,
        labelspacing=0.25,
        handletextpad=0.5,
        borderpad=0.3,
        borderaxespad=0.3,
        columnspacing=0.6,
    )

    ax.grid(True)
    grid_color = ax.xaxis.get_gridlines()[0].get_color()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_color(grid_color)
    ax.spines['right'].set_color(grid_color)
    ax.spines['bottom'].set_color(grid_color)
    ax.spines['left'].set_color(grid_color)
    # Make major and minor tick lines gray, but labels stay black
    ax.tick_params(axis='both', which='both', color=grid_color, labelcolor='black')

    ax.set_ylabel(ylabel, fontsize=17)
    ax.set_xlabel(xlabel, fontsize=17)
    ax.tick_params(axis='both', labelsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(str(save_path))


def plot_pareto_n_configs(
    fig_save_dir: str | Path = Path("plots") / "n_configs",
    average_seeds: bool = True,
    exclude_imputed: bool = True,
    ban_bad_methods: bool = True,
):
    include_portfolio = False
    include_hpo_seeds = False
    imputed_methods = [
        "TABPFNV2_GPU",
        "TABICL_GPU",
    ]

    fig_save_dir = Path(fig_save_dir)

    # Hardcoded for the paper to match the other plots
    method_order = [
        'RealMLP',
        'TabM',
        'CatBoost',
        'ModernNCA',
        'LightGBM',
        'XGBoost',
        'TorchMLP',
        'TabDPT',
        'FastaiMLP',
        'EBM',
        'ExtraTrees',
        'RandomForest',
    ]

    method_rename_map = get_method_rename_map()
    method_rename_map["REALMLP"] = "RealMLP"
    framework_types = list(method_rename_map.keys())

    # results_file = "hpo_new_lb.parquet"
    # s3_path = "s3://tabarena/tmp/camera_ready/hpo_new_lb.parquet"

    results_file = "hpo_camera_ready_lb.parquet"
    s3_path = "s3://tabarena/tmp/camera_ready/hpo_camera_ready_lb.parquet"
    from autogluon.common.savers import save_pd

    try:
        results_hpo = load_pd.load(path=results_file)
    except:
        print(f"Downloading from s3: {s3_path}")
        # download from s3 if missing
        results_hpo = load_pd.load(path=s3_path)
        save_pd.save(path=results_file, df=results_hpo)
        results_hpo = load_pd.load(path=results_file)
    # save_pd.save(path=s3_path, df=results_hpo)

    if exclude_imputed:
        results_hpo = results_hpo[~results_hpo["config_type"].isin(imputed_methods)]

    use_old_lb = True
    if use_old_lb:
        methods_lst = [m for m in tabarena_method_metadata_2025_06_12_collection_main.method_metadata_lst if m.method_type != "portfolio"]
        tabarena_context = TabArenaContext(methods=methods_lst)
    else:
        tabarena_context = TabArenaContext()

    calibration_framework = "RF (default)"
    elo_bootstrap_rounds = 1

    result_baselines = tabarena_context.load_results_paper()
    task_metadata = tabarena_context.task_metadata

    results_hpo_mean = results_hpo.copy().groupby(["method", "dataset", "fold", "problem_type", "metric", "config_type"]).mean(
        numeric_only=True
    ).drop(columns=["seed"]).reset_index()

    results_lst = [
        results_hpo_mean,
    ]

    if include_hpo_seeds:
        results_hpo_seeds = results_hpo.copy()
        results_hpo_seeds["method"] = results_hpo_seeds["method"] + "-" + results_hpo_seeds["seed"].astype(str)
        results_hpo_seeds["config_type"] = results_hpo_seeds["config_type"] + "-" + results_hpo_seeds["seed"].astype(str)
        results_hpo_seeds = results_hpo_seeds.groupby(["method", "dataset", "fold", "problem_type", "metric", "config_type"]).mean(
            numeric_only=True).reset_index()
        results_lst.append(results_hpo_seeds)

    if include_portfolio:
        results_portfolio = load_pd.load(path="rebuttal_portfolio_n_configs.parquet")
        results_portfolio["config_type"] = results_portfolio["method"]
        results_portfolio["method"] = results_portfolio["method"] + "-" + results_portfolio["n_portfolio"].astype(str)
        results_lst.append(results_portfolio)

    results_hpo = pd.concat(results_lst, ignore_index=True)
    combined_data = pd.concat([result_baselines, results_hpo], ignore_index=True)

    # ----- add times per 1K samples -----
    dataset_to_n_samples_train = tabarena_context.task_metadata.set_index("name")["n_samples_train_per_fold"].to_dict()
    dataset_to_n_samples_test = tabarena_context.task_metadata.set_index("name")["n_samples_test_per_fold"].to_dict()

    combined_data['time_train_s_per_1K'] = combined_data['time_train_s'] * 1000 / combined_data["dataset"].map(
        dataset_to_n_samples_train)
    combined_data['time_infer_s_per_1K'] = combined_data['time_infer_s'] * 1000 / combined_data["dataset"].map(
        dataset_to_n_samples_test)

    tabarena_init_kwargs = dict(
        task_col="dataset",
        columns_to_agg_extra=[
            "time_train_s",
            "time_infer_s",
            "time_train_s_per_1K",
            "time_infer_s_per_1K",
        ],
        groupby_columns=["problem_type", "metric"],
        seed_column="fold",
    )

    arena = TabArena(
        **tabarena_init_kwargs,
        error_col="metric_error",
    )

    arena_val = TabArena(
        **tabarena_init_kwargs,
        error_col="metric_error_val",
    )

    combined_data = combined_data[~combined_data["metric_error_val"].isna()]

    combined_data = arena.fillna_data(
        data=combined_data,
        fillna_method=calibration_framework,
    )

    results_per_task = arena.compute_results_per_task(data=combined_data)

    leaderboard = arena.leaderboard(
        data=combined_data,
        include_elo=True,
        elo_kwargs=dict(
            calibration_framework=calibration_framework,
            calibration_elo=1000,
            BOOTSTRAP_ROUNDS=elo_bootstrap_rounds,
        ),
        average_seeds=average_seeds,
        include_baseline_advantage=True,
    )

    leaderboard_val = arena_val.leaderboard(
        data=combined_data,
        include_elo=True,
        elo_kwargs=dict(
            calibration_framework=calibration_framework,
            calibration_elo=1000,
            BOOTSTRAP_ROUNDS=elo_bootstrap_rounds,
        ),
        average_seeds=average_seeds,
        include_baseline_advantage=True,
    )

    leaderboard["elo_val"] = leaderboard_val["elo"]
    leaderboard["improvability_val"] = leaderboard_val["improvability"]
    leaderboard["baseline_advantage_val"] = leaderboard_val["baseline_advantage"]

    leaderboard = leaderboard.reset_index(drop=False)

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard)

    methods_map = results_hpo[["method", "n_portfolio", "n_ensemble", "config_type"]].drop_duplicates(subset=["method"]).set_index("method")
    leaderboard = leaderboard[leaderboard["method"].isin(methods_map.index)]
    leaderboard["n_portfolio"] = leaderboard["method"].map(methods_map["n_portfolio"])
    leaderboard["config_type"] = leaderboard["method"].map(methods_map["config_type"])

    leaderboard["name"] = leaderboard["config_type"]

    leaderboard = leaderboard.sort_values(by=["config_type", "n_portfolio"])

    leaderboard["Elo"] = leaderboard["elo"]
    leaderboard["Elo (Test)"] = leaderboard["Elo"]
    leaderboard["Elo (Val)"] = leaderboard["elo_val"]
    leaderboard["Elo (Val) - Elo (Test)"] = leaderboard["Elo (Val)"] - leaderboard["Elo (Test)"]
    leaderboard["Improvability (%)"] = leaderboard["improvability"] * 100
    leaderboard["Improvability (%) (Test)"] = leaderboard["Improvability (%)"]
    leaderboard["Improvability (%) (Val)"] = leaderboard["improvability_val"] * 100
    leaderboard["Improvability (%) (Test) - Improvability (%) (Val)"] = leaderboard["Improvability (%) (Test)"] - leaderboard["Improvability (%) (Val)"]

    leaderboard["Baseline Advantage (%)"] = leaderboard["baseline_advantage"] * 100
    leaderboard["Baseline Advantage (%) (Test)"] = leaderboard["Baseline Advantage (%)"]
    leaderboard["Baseline Advantage (%) (Val)"] = leaderboard["baseline_advantage_val"] * 100
    leaderboard["Baseline Advantage (%) (Test - Val)"] = (leaderboard["baseline_advantage"] - leaderboard[
        "baseline_advantage_val"]) * 100

    leaderboard['Train time per 1K samples (s) (median)'] = leaderboard["median_time_train_s_per_1K"]
    leaderboard['Inference time per 1K samples (s) (median)'] = leaderboard["median_time_infer_s_per_1K"]

    leaderboard["name"] = leaderboard["name"].map(method_rename_map).fillna(leaderboard["name"])

    file_ext = ".pdf"

    if ban_bad_methods:
        bad_methods = ["KNN", "LR"]
        leaderboard = leaderboard[~leaderboard["config_type"].isin(bad_methods)]

    plot_kwargs = {
        "sort_col": "n_portfolio",
        "method_order": method_order,
    }

    plot_hpo(
        df=leaderboard,
        xlabel="Train time per 1K samples (s) (median)",
        ylabel="Elo",
        save_path=fig_save_dir / f"pareto_n_configs_elo{file_ext}",
        max_Y=True,
        **plot_kwargs,
    )
    plot_hpo(
        df=leaderboard,
        xlabel="Train time per 1K samples (s) (median)",
        ylabel="Elo (Val)",
        save_path=fig_save_dir / f"pareto_n_configs_elo_val{file_ext}",
        max_Y=True,
        optimal_arrow=False,
        **plot_kwargs,
    )
    plot_hpo(
        df=leaderboard,
        xlabel="Train time per 1K samples (s) (median)",
        ylabel="Improvability (%)",
        save_path=fig_save_dir / f"pareto_n_configs_imp{file_ext}",
        max_Y=False,
        **plot_kwargs,
    )
    plot_hpo(
        df=leaderboard,
        xlabel="Inference time per 1K samples (s) (median)",
        ylabel="Elo",
        save_path=fig_save_dir / f"pareto_n_configs_elo_infer{file_ext}",
        max_Y=True,
        **plot_kwargs,
    )
    plot_hpo(
        df=leaderboard,
        xlabel="Inference time per 1K samples (s) (median)",
        ylabel="Improvability (%)",
        save_path=fig_save_dir / f"pareto_n_configs_imp_infer{file_ext}",
        max_Y=False,
        **plot_kwargs,
    )

    plot_hpo(
        df=leaderboard,
        xlabel="Train time per 1K samples (s) (median)",
        ylabel="Baseline Advantage (%)",
        save_path=fig_save_dir / f"pareto_n_configs_adv{file_ext}",
        max_Y=True,
        **plot_kwargs,
    )

    plot_hpo(
        df=leaderboard,
        xlabel="Inference time per 1K samples (s) (median)",
        ylabel="Baseline Advantage (%)",
        save_path=fig_save_dir / f"pareto_n_configs_adv_infer{file_ext}",
        max_Y=True,
        **plot_kwargs,
    )

    plot_hpo(
        df=leaderboard,
        xlabel="Baseline Advantage (%) (Val)",
        ylabel="Baseline Advantage (%) (Test)",
        save_path=fig_save_dir / f"pareto_n_configs_adv_vs{file_ext}",
        max_Y=True,
        max_X=False,
        xlog=False,
        **plot_kwargs,
    )

    plot_hpo(
        df=leaderboard,
        xlabel="Improvability (%) (Val)",
        ylabel="Improvability (%) (Test)",
        save_path=fig_save_dir / f"pareto_n_configs_imp_vs{file_ext}",
        max_Y=False,
        max_X=True,
        xlog=False,
        **plot_kwargs,
    )

    plot_hpo(
        df=leaderboard,
        xlabel="Elo (Val)",
        ylabel="Elo (Test)",
        save_path=fig_save_dir / f"pareto_n_configs_elo_vs{file_ext}",
        max_Y=True,
        max_X=False,
        xlog=False,
        **plot_kwargs,
    )

    plot_hpo(
        df=leaderboard,
        xlabel="Train time per 1K samples (s) (median)",
        ylabel="Baseline Advantage (%) (Test - Val)",
        save_path=fig_save_dir / f"pareto_n_configs_adv_overfit{file_ext}",
        max_Y=True,
        **plot_kwargs,
    )

    plot_hpo(
        df=leaderboard,
        xlabel="Train time per 1K samples (s) (median)",
        ylabel="Elo (Val) - Elo (Test)",
        save_path=fig_save_dir / f"pareto_n_configs_elo_overfit{file_ext}",
        max_Y=False,
        **plot_kwargs,
    )

    plot_hpo(
        df=leaderboard,
        xlabel="Train time per 1K samples (s) (median)",
        ylabel="Improvability (%) (Test) - Improvability (%) (Val)",
        save_path=fig_save_dir / f"pareto_n_configs_imp_overfit{file_ext}",
        max_Y=False,
        **plot_kwargs,
    )


if __name__ == "__main__":
    plot_pareto_n_configs()
