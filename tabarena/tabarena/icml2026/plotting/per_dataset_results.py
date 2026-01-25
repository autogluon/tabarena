from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_model_performance_across_datasets(
    df: pd.DataFrame,
    *,
    dataset_col: str = "dataset",
    metric_col: str = "metric_error",
    model_col: str = "model_name",

    mode: Literal["rank", "median_centered_signed"] = "median_centered_signed",

    normalization_reference_models: Optional[Sequence[str]] = None,
    display_models: Optional[Sequence[str]] = None,

    sort_datasets_by_model: Optional[str] = None,
    sort_datasets_by_best_of_models: Optional[Sequence[str]] = None,
    sort_direction: Literal["best_to_worst", "worst_to_best"] = "best_to_worst",

    clip_bad_side: bool = True,
    bad_side_cap: float = 1.0,
    clip_good_side: bool = False,
    good_side_cap: float = -1.0,

    dataset_order: Optional[Sequence[str]] = None,
    model_order: Optional[Sequence[str]] = None,

    figsize: Tuple[float, float] = (10, 4.8),
    auto_width_per_dataset: Optional[float] = None,

    # 🔹 Font sizes
    font_size: float = 11.0,
    title_font_size: Optional[float] = None,
    label_font_size: Optional[float] = None,
    tick_font_size: Optional[float] = None,
    legend_font_size: Optional[float] = None,
    legend_order: Optional[Sequence[str]] = None,

    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,

    jitter: float = 0.10,

    # 🔹 INCREASED default marker size
    marker_size: float = 90.0,
    alpha: float = 0.9,

    invert_rank_axis: bool = True,
    grid: bool = True,

    # 🔹 legend control (top by default)
    legend: bool = True,
    legend_ncol: Optional[int] = None,

    connect_models: bool = False,
    line_alpha: float = 0.35,

    show_model_averages: bool = False,
    average_line_style: str = "--",
    average_line_alpha: float = 0.6,
    average_line_width: float = 1.5,

    model_color_groups: Optional[Dict[str, Sequence[str]]] = None,
    model_markers: Optional[Dict[str, str]] = None,
    default_markers: Sequence[str] = ("o", "s", "^", "D", "v", "P", "X"),

    # 🔹 NEW: save figure
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:

    # ---------------- Aggregate ----------------
    agg = (
        df.groupby([dataset_col, model_col], as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: "metric_mean"})
    )

    # ---------------- Normalization ----------------
    if mode == "rank":
        agg["value"] = agg.groupby(dataset_col)["metric_mean"].rank(
            ascending=True, method="average"
        )
        value_label = "Rank (1 = best)"
    else:
        ref = (
            agg[agg[model_col].isin(normalization_reference_models)]
            if normalization_reference_models is not None
            else agg
        )
        stats = (
            ref.groupby(dataset_col)["metric_mean"]
            .agg(best="min", median="median", worst="max")
            .reset_index()
        )
        agg = agg.merge(stats, on=dataset_col, how="left")

        x, best, med, worst = (
            agg["metric_mean"],
            agg["best"],
            agg["median"],
            agg["worst"],
        )

        value = np.where(
            x <= med,
            (x - med) / (med - best).replace(0, np.nan),
            (x - med) / (worst - med).replace(0, np.nan),
        )
        if clip_bad_side:
            value = np.minimum(value, bad_side_cap)
        if clip_good_side:
            value = np.maximum(value, good_side_cap)

        agg["value"] = value
        value_label = "Normalized error (lower is better)"

    # ---------------- Dataset ordering ----------------
    all_ds = agg[dataset_col].unique()
    if dataset_order is None:
        if sort_datasets_by_best_of_models is not None:
            scores = (
                agg[agg[model_col].isin(sort_datasets_by_best_of_models)]
                .groupby(dataset_col)["value"]
                .min()
                .reindex(all_ds)
            )
            dataset_order = scores.sort_values(
                ascending=(sort_direction == "best_to_worst")
            ).index.tolist()
        elif sort_datasets_by_model is not None:
            scores = (
                agg[agg[model_col] == sort_datasets_by_model]
                .set_index(dataset_col)["value"]
                .reindex(all_ds)
            )
            dataset_order = scores.sort_values(
                ascending=(sort_direction == "best_to_worst")
            ).index.tolist()
        else:
            dataset_order = sorted(all_ds)

    # ---------------- Model ordering ----------------
    if model_order is None:
        base = agg if display_models is None else agg[agg[model_col].isin(display_models)]
        model_order = (
            base.groupby(model_col)["value"].mean().sort_values().index.tolist()
        )

    plot_df = agg if display_models is None else agg[agg[model_col].isin(display_models)]

    # ---------------- Font size ----------------
    title_fs = title_font_size or font_size
    label_fs = label_font_size or font_size
    tick_fs = tick_font_size or font_size
    legend_fs = legend_font_size or font_size

    # ---------------- Color & marker mapping ----------------
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_to_color, model_to_marker = {}, {}

    if model_color_groups:
        for c, (_, models) in zip(color_cycle, model_color_groups.items()):
            for m in models:
                model_to_color[m] = c

    remaining = [m for m in model_order if m not in model_to_color]
    for i, m in enumerate(remaining):
        model_to_color[m] = color_cycle[i % len(color_cycle)]

    marker_iter = iter(default_markers)
    for m in model_order:
        model_to_marker[m] = (
            model_markers[m] if model_markers and m in model_markers else next(marker_iter)
        )

    # ---------------- Plot prep ----------------
    ds_to_x = {d: i for i, d in enumerate(dataset_order)}
    plot_df = plot_df.copy()
    plot_df["x"] = plot_df[dataset_col].map(ds_to_x)
    rng = np.random.default_rng(0)
    plot_df["xj"] = plot_df["x"] + plot_df[model_col].map(
        {m: rng.uniform(-jitter, jitter) for m in model_order}
    )

    if auto_width_per_dataset is not None:
        fig_width = max(figsize[0], plot_df[dataset_col].nunique() * auto_width_per_dataset)
        fig, ax = plt.subplots(figsize=(fig_width, figsize[1]))
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # ---------------- Plot ----------------
    for m in model_order:
        sub = plot_df[plot_df[model_col] == m]
        if sub.empty:
            continue

        ax.scatter(
            sub["xj"],
            sub["value"],
            s=marker_size,
            alpha=alpha,
            color=model_to_color[m],
            marker=model_to_marker[m],
            label=m,
        )

        if connect_models and len(sub) >= 2:
            ax.plot(sub["x"], sub["value"], color=model_to_color[m], alpha=line_alpha)

        if show_model_averages:
            ax.hlines(
                sub["value"].mean(),
                -0.5,
                len(dataset_order) - 0.5,
                linestyles=average_line_style,
                linewidth=average_line_width,
                alpha=average_line_alpha,
                colors=model_to_color[m],
            )

    ax.set_xticks(range(len(dataset_order)))
    ax.set_xticklabels(dataset_order, rotation=30, ha="right", fontsize=tick_fs)
    ax.set_xlim(-0.6, len(dataset_order) - 0.4)

    ax.set_ylabel(ylabel or value_label, fontsize=label_fs)
    ax.set_xlabel(xlabel or "Dataset", fontsize=label_fs)

    ax.tick_params(axis="y", labelsize=tick_fs)

    if title:
        ax.set_title(title, fontsize=title_fs)

    if mode == "rank" and invert_rank_axis:
        ax.invert_yaxis()

    ax.grid(grid, axis="y", alpha=0.3)

    # 🔹 LEGEND AT TOP
    if legend:
        handles, labels = ax.get_legend_handles_labels()

        if legend_order is not None:
            label_to_handle = dict(zip(labels, handles))
            ordered_handles = [
                label_to_handle[l] for l in legend_order if l in label_to_handle
            ]
            ordered_labels = [
                l for l in legend_order if l in label_to_handle
            ]
        else:
            ordered_handles = handles
            ordered_labels = labels

        ncol = legend_ncol or min(len(ordered_labels), 10)
        ax.legend(
            ordered_handles,
            ordered_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=ncol,
            frameon=False,
            fontsize=legend_font_size,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.92))

    # 🔹 SAVE FIGURE
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax