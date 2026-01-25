import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def ablation_boxplot_colored_by_best(
    prep_ablation_df: pd.DataFrame,
    base_df: pd.DataFrame,
    *,
    dataset_col: str = "dataset",
    method_col: str = "method",
    metric_col: str = "metric_error",
    base_metric_col: str = "metric_error_baseLGB",  # column in base_df
    baseline_name: str = "Baseline",
    lower_is_better: bool = True,

    # score shown on x-axis (improvement vs baseline)
    mode: str = "log_ratio",   # "log_ratio" or "relative"
    eps: float = 1e-12,
    cap: float | tuple[float, float] | None = None,

    # methods shown and ordering
    method_order: list[str] | None = None,
    method_labels: dict[str, str] | None = None,  # method -> display label (can include '\n')

    # winner / color assignment
    winner_rule: str = "methods_or_baseline",  # "methods_only" | "methods_or_baseline" | "methods_if_beat_else_baseline"
    cmap_name: str = "tab20",

    # optional: remove per-method ties to baseline from the PLOT only
    drop_equal_to_baseline: bool = False,
    equal_atol: float = 0.0,
    equal_rtol: float = 0.0,

    # tie-breaking tolerance for winner selection (baseline wins ties)
    winner_atol: float = 0.0,
    winner_rtol: float = 0.0,

    # NEW: fat marker = mean of points belonging to the winner-group of that method
    show_winner_group_mean: bool = True,
    mean_marker: str = "D",
    # size for fat markers
    mean_marker_size_main: float = 100.0,   # when method == winner
    mean_marker_size_other: float = 50.0,   # when method != winner
    mean_marker_edge: bool = True,
    mean_marker_edge_lw: float = 0.8,

    # legend
    legend: bool = True,
    legend_title: str = "Winner on dataset",
    legend_loc: str = "center left",
    legend_bbox_to_anchor: tuple[float, float] = (1.02, 0.5),
    legend_fontsize: float | None = None,

    # plot style
    figsize: tuple[float, float] = (6.8, 3.6),
    jitter: float = 0.10,
    point_size: float = 10.0,
    point_alpha: float = 0.75,
    box_linewidth: float = 0.9,
    spine_linewidth: float = 0.8,
    font_size: float = 8.0,
    title: str | None = None,
    x_label: str | None = None,
    xlabel_fontsize: float | None = None,

    # saving
    save_path: str | None = None,
    dpi: int = 300,
    transparent: bool = True,
    show: bool = True,
):
    """
    Horizontal boxplots of improvement vs baseline for many methods (long DF),
    with per-dataset points colored by the dataset's winner method.

    NEW: per boxplot row, optionally add ONE "fat" marker at the mean score over datasets
         whose *winner method equals that row's method* (the "winner group" for that method).

    Returns:
      fig, ax, merged_long, best_method_by_dataset, color_by_method
    """

    # ---- Validate columns ----
    for c in [dataset_col, method_col, metric_col]:
        if c not in prep_ablation_df.columns:
            raise ValueError(f"prep_ablation_df missing column '{c}'")
    for c in [dataset_col, base_metric_col]:
        if c not in base_df.columns:
            raise ValueError(f"base_df missing column '{c}'")

    # ---- Baseline per dataset ----
    base_small = base_df[[dataset_col, base_metric_col]].dropna().copy()
    base_small = base_small.rename(columns={base_metric_col: "_base_metric"})
    base_small = base_small.drop_duplicates(subset=[dataset_col], keep="last")

    # ---- Merge baseline into long df ----
    d_all = prep_ablation_df[[dataset_col, method_col, metric_col]].dropna().copy()
    d_all = d_all.rename(columns={metric_col: "_metric"})
    d_all = d_all.merge(base_small, on=dataset_col, how="inner")

    if d_all.empty:
        raise ValueError("After merging on dataset, no rows remain. Check dataset overlap / NaNs.")

    # ---- Optional: drop ties from PLOT only ----
    d_plot = d_all
    if drop_equal_to_baseline:
        tie_mask_plot = np.isclose(
            d_plot["_metric"].to_numpy(dtype=float),
            d_plot["_base_metric"].to_numpy(dtype=float),
            rtol=equal_rtol,
            atol=equal_atol,
            equal_nan=False,
        )
        d_plot = d_plot.loc[~tie_mask_plot].copy()
        if d_plot.empty:
            raise ValueError("All rows removed by drop_equal_to_baseline. Relax tolerances?")

    # ---- Method order ----
    if method_order is None:
        method_order = sorted(d_plot[method_col].unique().tolist())
    else:
        present = set(d_plot[method_col].unique())
        method_order = [m for m in method_order if m in present]
    if not method_order:
        raise ValueError("No methods to plot (method_order empty after filtering).")

    # ---- Compute improvement scores for plotting (positive = better) ----
    base = d_plot["_base_metric"].to_numpy(dtype=float)
    mval = d_plot["_metric"].to_numpy(dtype=float)

    if mode == "log_ratio":
        base_safe = np.maximum(base, eps)
        m_safe = np.maximum(mval, eps)
        lr = np.log(m_safe / base_safe)
        score = (-lr) if lower_is_better else (lr)
        default_xlabel = "Improvement vs baseline (−log ratio)"
    elif mode == "relative":
        denom = np.maximum(np.abs(base), eps)
        score = ((base - mval) / denom) if lower_is_better else ((mval - base) / denom)
        default_xlabel = "Relative improvement vs baseline"
    else:
        raise ValueError("mode must be 'log_ratio' or 'relative'")

    d_plot = d_plot.copy()
    d_plot["_score"] = score

    if cap is not None:
        if isinstance(cap, (int, float)):
            low, high = -float(cap), float(cap)
        else:
            low, high = cap
        d_plot["_score"] = d_plot["_score"].clip(lower=low, upper=high)

    # ---- Winner computation (RAW metric, baseline wins ties) ----
    agg = d_all.groupby([dataset_col, method_col], as_index=False)["_metric"].mean()
    base_series = base_small.set_index(dataset_col)["_base_metric"]

    def _winner_methods_or_baseline_tiebreak(agg_methods: pd.DataFrame) -> dict:
        base_as_long = base_small.copy()
        base_as_long[method_col] = baseline_name
        base_as_long["_metric"] = base_as_long["_base_metric"]
        base_as_long = base_as_long[[dataset_col, method_col, "_metric"]]

        pool = pd.concat([agg_methods, base_as_long], ignore_index=True)
        pool["_is_base"] = (pool[method_col] == baseline_name)

        winners = {}
        for ds, g in pool.groupby(dataset_col, sort=False):
            vals = g["_metric"].to_numpy(dtype=float)

            best_val = vals.min() if lower_is_better else vals.max()
            if winner_atol or winner_rtol:
                is_best = np.isclose(vals, best_val, rtol=winner_rtol, atol=winner_atol)
            else:
                is_best = (vals == best_val)

            cand = g.loc[is_best].copy()

            # tie-break 1: baseline wins if among candidates
            if (cand["_is_base"]).any():
                winners[ds] = baseline_name
                continue

            # tie-break 2: deterministic by method name
            winners[ds] = cand.sort_values(method_col, kind="mergesort").iloc[0][method_col]

        return winners

    if winner_rule not in {
        "methods_only",
        "methods_or_baseline",
        "methods_if_beat_else_baseline",
    }:
        raise ValueError(
            "winner_rule must be one of: 'methods_only', 'methods_or_baseline', 'methods_if_beat_else_baseline'"
        )

    if winner_rule == "methods_only":
        if lower_is_better:
            idx = agg.groupby(dataset_col)["_metric"].idxmin()
        else:
            idx = agg.groupby(dataset_col)["_metric"].idxmax()
        best_method_by_dataset = (
            agg.loc[idx, [dataset_col, method_col]]
            .set_index(dataset_col)[method_col]
            .to_dict()
        )

    elif winner_rule == "methods_or_baseline":
        best_method_by_dataset = _winner_methods_or_baseline_tiebreak(agg)

    else:  # "methods_if_beat_else_baseline" -> draw => baseline
        if lower_is_better:
            idx_m = agg.groupby(dataset_col)["_metric"].idxmin()
        else:
            idx_m = agg.groupby(dataset_col)["_metric"].idxmax()

        best_m = agg.loc[idx_m].set_index(dataset_col)[method_col]
        best_m_val = agg.loc[idx_m].set_index(dataset_col)["_metric"]

        best_method_by_dataset = {}
        for ds, base_val in base_series.items():
            if ds in best_m.index:
                mval_ds = float(best_m_val.loc[ds])
                base_ds = float(base_val)
                draw = (
                    np.isclose(mval_ds, base_ds, rtol=winner_rtol, atol=winner_atol)
                    if (winner_atol or winner_rtol)
                    else (mval_ds == base_ds)
                )
                if lower_is_better:
                    method_beats = (mval_ds < base_ds) and not draw
                else:
                    method_beats = (mval_ds > base_ds) and not draw
                best_method_by_dataset[ds] = best_m.loc[ds] if method_beats else baseline_name
            else:
                best_method_by_dataset[ds] = baseline_name

    # ---- Colors: one per possible winner ----
    winners = sorted(set(best_method_by_dataset.values()))
    cmap = plt.get_cmap(cmap_name, max(len(winners), 1))
    color_by_method = {m: cmap(i) for i, m in enumerate(winners)}
    color_by_dataset = {ds: color_by_method[m] for ds, m in best_method_by_dataset.items()}

    # ---- Labels for y-ticks and legend ----
    def display_name(m: str) -> str:
        return method_labels.get(m, m) if method_labels else m

    y_labels = [display_name(m) for m in method_order]

    # ---- Prepare boxplot data + dataset arrays (for coloring points) ----
    data = []
    dataset_lists = []
    for m in method_order:
        sub = d_plot.loc[d_plot[method_col] == m, [dataset_col, "_score"]].copy()
        data.append(sub["_score"].to_numpy())
        dataset_lists.append(sub[dataset_col].to_numpy())

    if legend_fontsize is None:
        legend_fontsize = font_size

    # ---- Plot ----
    with plt.rc_context({
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "axes.linewidth": spine_linewidth,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig, ax = plt.subplots(figsize=figsize)

        common_props = dict(linewidth=box_linewidth)
        ax.boxplot(
            data,
            labels=y_labels,
            vert=False,
            showfliers=False,
            widths=0.62,
            boxprops=common_props,
            whiskerprops=common_props,
            capprops=common_props,
            medianprops=common_props,
        )

        rng = np.random.default_rng(0)
        for i, ds_arr in enumerate(dataset_lists, start=1):
            x = data[i - 1]
            y = i + rng.uniform(-jitter, jitter, size=len(x))
            colors = [color_by_dataset.get(ds, (0, 0, 0, 1)) for ds in ds_arr]
            ax.scatter(x, y, s=point_size, alpha=point_alpha, linewidths=0, zorder=3, c=colors)

        # NEW: one fat marker per WINNER COLOR per row,
        # with emphasis when winner == row method
        if show_winner_group_mean:
            for i, m in enumerate(method_order, start=1):
                ds_arr = np.asarray(dataset_lists[i - 1])
                x_arr = np.asarray(data[i - 1])
                if len(x_arr) == 0:
                    continue

                for winner_m, color in color_by_method.items():

                    # 🚫 baseline is never fat
                    if winner_m == baseline_name:
                        continue

                    mask = np.array(
                        [best_method_by_dataset.get(ds) == winner_m for ds in ds_arr],
                        dtype=bool,
                    )
                    if not mask.any():
                        continue

                    mean_x = float(np.mean(x_arr[mask]))

                    # ⭐ emphasize if this method is best in its own row
                    is_self = (winner_m == m)
                    size = mean_marker_size_main if is_self else mean_marker_size_other

                    ax.scatter(
                        [mean_x], [i],
                        s=size,
                        marker=mean_marker,
                        c=[color],
                        edgecolors="black" if mean_marker_edge else "none",
                        linewidths=mean_marker_edge_lw if mean_marker_edge else 0.0,
                        zorder=4 if is_self else 3.8,
                    )



        ax.axvline(0.0, color="black", linewidth=box_linewidth)
        ax.set_xlabel(
            x_label if x_label is not None else default_xlabel,
            fontsize=(xlabel_fontsize if xlabel_fontsize is not None else font_size),
        )
        ax.set_ylabel("")
        if title:
            ax.set_title(title)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(spine_linewidth)
        ax.spines["bottom"].set_linewidth(spine_linewidth)
        ax.tick_params(axis="both", width=spine_linewidth, length=3)

        for t in ax.get_yticklabels():
            t.set_va("center")
            t.set_linespacing(0.95)

        # Legend for winner colors
        if legend:
            handles = [
                Line2D([0], [0], marker="o", linestyle="None",
                       markersize=5, markerfacecolor=color_by_method[m],
                       markeredgewidth=0, label=display_name(m))
                for m in winners
            ]
            ax.legend(
                handles=handles,
                title=legend_title,
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox_to_anchor,
                frameon=False,
                borderaxespad=0.0,
            )

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01, transparent=transparent)

        if show:
            plt.show()

    merged_long = d_plot[[dataset_col, method_col]].copy()
    merged_long["metric"] = d_plot["_metric"].astype(float)
    merged_long["score"] = d_plot["_score"].astype(float)

    return fig, ax, merged_long, best_method_by_dataset, color_by_method


# -------------------- Example --------------------
# fig, ax, merged_long, best_by_ds, color_by_method = ablation_boxplot_colored_by_best(
#     prep_ablation_df=prep_ablation_df,
#     base_df=base_lgb_df,
#     base_metric_col="metric_error_baseLGB",
#     baseline_name="Base-LGB",
#     winner_rule="methods_or_baseline",
#     winner_atol=1e-12,  # baseline wins ties within tolerance
#     mode="log_ratio",
#     cap=1.0,
#     drop_equal_to_baseline=False,  # affects plot only
#     show_winner_group_mean=True,   # <-- fat marker enabled
#     mean_marker="D",
#     mean_marker_size=90,
#     title="Prep ablation vs Base-LGB",
#     save_path="prep_ablation_colored_with_means.pdf",
# )
