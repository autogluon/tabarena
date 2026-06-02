from __future__ import annotations

import copy
import os
from pathlib import Path

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from autogluon.common.loaders import load_pd

from bencheval.tabarena import TabArena
from tabarena.nips2025_utils.compare import subset_tasks
from tabarena.nips2025_utils.eval_all import (
    get_all_subset_combinations,
    get_website_folder_name,
)
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.paper.paper_utils import get_method_rename_map
from tabarena.plot.plot_pareto_frontier import plot_optimal_arrow
from tabarena.utils.parallel_for import parallel_for


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
    rank_by_y: bool = True,
    sort_col: str | None = None,
    method_order: list[str] | None = None,
    optimal_arrow: bool = True,
    ylim: tuple[float | None, float | None] | None = None,
    display_names: dict[str] | None = None,
    legend_in_plot: bool = False,
    figsize: tuple[float, float] = (8, 4.5),
    title: str | None = None,
    title_fontsize: float = 20,
    ylabel_display: str | None = None,
    xlabel_display: str | None = None,
    dataset_metadata: dict[str, str] | None = None,
    reverse_colors: bool = False,
    link_points: list[str] | None = None,
    method_color_overrides: dict[str, str] | None = None,
    show_pareto_frontier: bool = False,
    force_label_methods: list[str] | None = None,
    label_display_names: dict[str, str] | None = None,
    hidden_legend_methods: list[str] | None = None,
    legend_display_names: dict[str, str] | None = None,
    left_label_methods: list[str] | None = None,
    below_label_methods: list[str] | None = None,
    clamp_negative_ymin: bool = False,
):
    """Plot HPO trajectories for multiple methods.

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
    df = df.copy(deep=True)
    # Drop rows where the x or y plotting column is NaN. Methods whose
    # validation-axis values are unavailable (e.g. dropped from the val
    # leaderboard in `compute_tuning_trajectories_leaderboard` because
    # they had NaN `metric_error_val`) would otherwise propagate NaN
    # into the per-method peak ranking and the Pareto-frontier
    # computation, producing a corrupted frontier line and bogus axis
    # ranges. Methods that end up with zero rows after the filter are
    # silently skipped by the existing `df_method.empty` guard below.
    df = df.dropna(subset=[xlabel, ylabel])

    if display_names is not None:
        df[method_col] = df[method_col].map(display_names).fillna(df[method_col])
        if method_order is not None:
            method_order = [display_names.get(m, m) for m in method_order]

    if sort_col is not None:
        assert sort_col in df.columns
    # Build a 60-color palette from tab20 / tab20b / tab20c
    colors60 = (
        list(sns.color_palette("tab20", 20))
        + list(sns.color_palette("tab20b", 20))
        + list(sns.color_palette("tab20c", 20))
    )

    method_names = list(df[method_col].unique())

    # Determine peak per method (max if max_Y else min)
    if rank_by_y:
        if max_Y:
            peak_per_method = {m: df.loc[df[method_col] == m, ylabel].max() for m in method_names}
        else:
            peak_per_method = {m: df.loc[df[method_col] == m, ylabel].min() for m in method_names}
        reverse_sort = max_Y
    else:
        if max_X:
            peak_per_method = {m: df.loc[df[method_col] == m, xlabel].max() for m in method_names}
        else:
            peak_per_method = {m: df.loc[df[method_col] == m, xlabel].min() for m in method_names}
        reverse_sort = max_X

    # Sort by peak and create a stable color map (alphabetical)
    sorted_methods = sorted(method_names, key=lambda m: peak_per_method[m], reverse=reverse_sort)
    base_methods_for_colors = sorted_methods if color_by_rank else sorted(method_names)

    if method_order:
        sorted_methods = method_order + [m for m in sorted_methods if m not in method_order]
        base_methods_for_colors = method_order + [m for m in base_methods_for_colors if m not in method_order]
    # ``reverse_colors`` flips palette assignment without touching
    # ``sorted_methods`` — i.e. the legend still reads top-to-bottom in the
    # caller's order, but the last method takes ``colors60[0]`` (the dark
    # blue), the second-to-last takes ``colors60[1]``, etc. Use this when
    # the legend's bottom entry should visually anchor the plot.
    if reverse_colors:
        base_methods_for_colors = list(reversed(base_methods_for_colors))
    color_map = {m: colors60[i % len(colors60)] for i, m in enumerate(base_methods_for_colors)}
    if method_color_overrides:
        # Pin specific methods to a caller-chosen color (e.g. to visually group
        # related variants like TabPFN-3 and TabPFN-3-Thinking).
        color_map.update(method_color_overrides)

    fig, ax = plt.subplots(figsize=figsize)
    if xlog:
        ax.set_xscale("log")

    if optimal_arrow:
        plot_optimal_arrow(ax=ax, max_X=max_X, max_Y=max_Y, size=0.45, scale=1.2)

    handles = []
    labels = []

    # Remember the first trajectory point (post-``sort_col`` ordering when
    # present) for every method. Used as the anchor for both:
    #   * the dotted ``link_points`` chain (stitches several methods'
    #     entry points into a single dashed connector)
    #   * the ``force_label_methods`` annotations (always-visible text
    #     labels independent of the Pareto frontier)
    first_point_coords: dict[str, tuple[float, float]] = {}

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
        (_h,) = ax.plot(
            times,
            scores,
            "-",  # no point markers
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
                    s=64,  # ~markersize=16 equivalent
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

        if len(times):
            first_point_coords[method_name] = (
                float(times[0]),
                float(scores[0]),
            )

    # Shared dedupe set: ``force_label_methods`` annotations and the
    # Pareto-frontier vertex annotations both write into it so the same
    # method never gets labeled twice. ``link_points`` no longer pushes
    # labels here — callers must add those methods explicitly via
    # ``force_label_methods`` if they want them annotated.
    seen_labels: set[str] = set()

    # Methods whose annotation should sit on the left of the point rather
    # than the default upper-right offset (e.g. when a point hugs another
    # label and the right-side text would collide). Looked up against the
    # *internal* method name, not the display name.
    left_label_set = set(left_label_methods or [])
    # Methods whose annotation should sit *below* the point (negative y
    # offset, top-anchored text) — useful when the upper-right slot is
    # already taken by another label.
    below_label_set = set(below_label_methods or [])

    def _annotation_placement(method_name: str) -> dict:
        if method_name in below_label_set:
            return dict(xytext=(4, -4), ha="left", va="top")
        if method_name in left_label_set:
            return dict(xytext=(-4, 4), ha="right")
        return dict(xytext=(4, 4), ha="left")

    if link_points:
        ordered_xs = [first_point_coords[m][0] for m in link_points if m in first_point_coords]
        ordered_ys = [first_point_coords[m][1] for m in link_points if m in first_point_coords]
        if len(ordered_xs) >= 2:
            ax.plot(
                ordered_xs,
                ordered_ys,
                linestyle=":",
                color="dimgray",
                linewidth=1.5,
                alpha=0.85,
                zorder=3,
            )

    if force_label_methods:
        # Annotate each listed method at its first trajectory point,
        # regardless of whether the method lands on the Pareto frontier.
        # Same per-method color + bold + white-stroke style used by the
        # frontier-vertex annotations so the two sets read as one layer.
        # ``label_display_names`` swaps the printed text only — internal
        # bookkeeping (``seen_labels``, ``first_point_coords``,
        # ``color_map``) keeps using the original method name.
        for lbl in force_label_methods:
            if lbl in seen_labels or lbl not in first_point_coords:
                continue
            seen_labels.add(lbl)
            x, y = first_point_coords[lbl]
            display_lbl = (label_display_names or {}).get(lbl, lbl)
            txt = ax.annotate(
                display_lbl,
                xy=(x, y),
                textcoords="offset points",
                fontsize=10,
                color=color_map.get(lbl, "black"),
                fontweight="bold",
                zorder=6,
                **_annotation_placement(lbl),
            )
            txt.set_path_effects(
                [PathEffects.withStroke(linewidth=3, foreground="white")],
            )

    if show_pareto_frontier:
        # Mirrors the dashed-black frontier from ``plot_pareto`` in
        # ``plot_pareto_frontier.py``: compute the piece-wise constant
        # frontier across every plotted point (each method contributes all
        # of its trajectory points, not just the peak), then overlay it as
        # a step-style line at zorder=1 so per-method trajectories still
        # render on top.
        from tabarena.plot.plot_pareto_frontier import get_pareto_frontier

        frontier_df = df[df[method_col].isin(sorted_methods)]
        Xs = frontier_df[xlabel].to_numpy()
        Ys = frontier_df[ylabel].to_numpy()
        labels_for_front = frontier_df[method_col].tolist()
        if len(Xs) >= 2:
            pareto_front, pareto_names = get_pareto_frontier(
                Xs=Xs,
                Ys=Ys,
                names=labels_for_front,
                max_X=max_X,
                max_Y=max_Y,
                include_boundary_edges=True,
            )
            pf_X = [pt[0] for pt in pareto_front]
            pf_Y = [pt[1] for pt in pareto_front]

            # Apply caller-provided ``ylim`` first so the axis limits we
            # read for the frontier extension match the final rendered
            # frame (otherwise we'd anchor the line to the auto-limits
            # and the later ``ax.set_ylim(ylim)`` would clip it).
            if ylim is not None:
                ax.set_ylim(ylim)
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()

            # Extension logic: extend the dashed line past the front so
            # it spans the whole plot. Two pieces, on opposite sides:
            #   - Horizontal extension out to ``x_min`` / ``x_max`` on the
            #     "worse-X" side, at the front's extreme Y on that side.
            #   - Vertical drop to ``y_min`` / ``y_max`` on the
            #     "better-X" side, anchored at the front's extreme X.
            # ``get_pareto_frontier`` builds the front in the direction of
            # the better X first (descending when ``max_X``, ascending
            # otherwise), so for ``max_X`` the better-X end is ``pf_X[0]``
            # (and ``pf_Y[0]`` is the worst Y on the front there); for
            # ``not max_X`` the better-X end is ``pf_X[-1]`` (and
            # ``pf_Y[-1]`` is the worst Y on the front there).
            #
            # Previous mistake: for ``max_X`` the extension was anchored
            # at ``pf_X[-1]`` (the worse-X end), drawing the vertical
            # drop on the wrong side of the plot. See
            # ``pareto_n_configs_adv_overfit_v2``.
            if max_X:
                pf_X_first = pf_X[0]
                pf_X_last = x_min
                pf_Y_first = y_min if max_Y else y_max
                pf_Y_last = pf_Y[-1]
            else:
                pf_X_first = pf_X[0]
                pf_X_last = x_max
                pf_Y_first = y_min if max_Y else y_max
                pf_Y_last = pf_Y[-1]

            pf_X = [pf_X_first, *pf_X, pf_X_last]
            pf_Y = [pf_Y_first, *pf_Y, pf_Y_last]

            ax.plot(
                pf_X,
                pf_Y,
                linewidth=1.2,
                zorder=1,
                color="black",
                linestyle="--",
                alpha=0.7,
            )

            # Annotate each real frontier vertex (the inserted boundary
            # / vertical-drop entries come back with ``None`` labels and
            # are skipped). De-duplicate by label so methods that appear
            # at multiple consecutive vertices get one annotation.
            #
            # Match the text styling used by ``plot_pareto`` in
            # ``plot_pareto_frontier.py``: per-method color (pulled from
            # ``color_map``), bold weight, and a white stroke path
            # effect for readability against the trajectory lines.
            # ``seen_labels`` is shared with the ``link_points`` block —
            # a method labeled there is skipped here so it doesn't get
            # annotated twice.
            for (x, y), lbl in zip(pareto_front, pareto_names, strict=False):
                if lbl is None or lbl in seen_labels:
                    continue
                seen_labels.add(lbl)
                display_lbl = (label_display_names or {}).get(lbl, lbl)
                txt = ax.annotate(
                    display_lbl,
                    xy=(x, y),
                    textcoords="offset points",
                    fontsize=10,
                    color=color_map.get(lbl, "black"),
                    fontweight="bold",
                    zorder=6,
                    **_annotation_placement(lbl),
                )
                txt.set_path_effects(
                    [PathEffects.withStroke(linewidth=3, foreground="white")],
                )

            # Plotting the extended frontier + label annotations triggers
            # matplotlib's auto-margin on a log x-axis, which pushes the
            # final ``xlim`` past the value we just anchored ``pf_X_last``
            # to — leaving a gap between the dashed line and the right
            # spine. Re-pin the limits we measured (same trick as
            # ``plot_pareto`` at the bottom of its frontier block).
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

    # Order the legend so it reads top-to-bottom in the same vertical direction
    # as the y-axis, keeping the legend consistent with the plotted points.
    #   * Higher-is-better (``max_Y``): the best method sits at the top of the
    #     plot, so keep the build order — which honours an explicit
    #     ``method_order`` (e.g. the Elo-derived ranking the per-dataset
    #     trajectory pipeline shares across a dataset's Pareto plots to keep
    #     colors and legend positions aligned) — leaving the best method first.
    #   * Lower-is-better: the best method sits at the *bottom* of the plot, so
    #     the legend must end with the method that has the lowest (best) value.
    #     Sort by each method's peak (best) value descending so the lowest value
    #     lands last. This must run even when ``method_order`` is pinned —
    #     otherwise an Elo-derived (higher-is-better) order leaves the lowest
    #     value first, contradicting the plot's "lower is better" direction.
    legend_fontsize = 9
    if max_Y:
        handles_legend = handles
        labels_legend = labels
    else:
        legend_order = sorted(
            range(len(labels)),
            key=lambda i: peak_per_method.get(labels[i], float("inf")),
            reverse=True,
        )
        handles_legend = [handles[i] for i in legend_order]
        labels_legend = [labels[i] for i in legend_order]

    # Filter out methods that should still appear in the figure but be
    # hidden from the legend (e.g. cluster of TabPFN-3 variants where we
    # only want the canonical entry shown). The label-side mapping then
    # renames the kept labels for display (e.g. "TabPFN-3-Thinking" →
    # "TabPFN-3"). Both operations are legend-only — the original method
    # names continue to drive color lookups, frontier annotations, etc.
    if hidden_legend_methods:
        hidden_set = set(hidden_legend_methods)
        kept = [(h, lbl) for h, lbl in zip(handles_legend, labels_legend, strict=False) if lbl not in hidden_set]
        handles_legend = [h for h, _ in kept]
        labels_legend = [lbl for _, lbl in kept]
    if legend_display_names:
        labels_legend = [legend_display_names.get(lbl, lbl) for lbl in labels_legend]

    if legend_in_plot:
        legend1 = ax.legend(
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
        bbox1_axes = None
    else:
        # Outside plot: position legends using axes transform
        # Add small gap (0.01) between plot and legend
        # First legend: method colors (on top) - this is typically the wider one
        legend1 = ax.legend(
            handles_legend,
            labels_legend,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),  # Small gap from axes right edge
            frameon=True,
            fontsize=legend_fontsize,
            ncol=1,
            labelspacing=0.15,
            handletextpad=0.0,
            borderpad=0.1,
            borderaxespad=0.0,
            columnspacing=0.4,
        )

        # Need to draw to get legend bbox
        fig.canvas.draw()
        bbox1 = legend1.get_window_extent()
        bbox1_axes = bbox1.transformed(ax.transAxes.inverted())

    # Second legend: dataset metadata block (only when both the metadata is
    # provided and we're rendering the legend outside the axes — the in-plot
    # path is too cramped for an extra legend without overlapping the data).
    legend2 = None
    if dataset_metadata and not legend_in_plot:
        from matplotlib.lines import Line2D

        # Render keys in column 1, values in column 2, row-aligned.
        # matplotlib lays ``ncol=2`` legends out column-major: the first
        # ``ceil(2N / 2) = N`` labels fill column 1 top-to-bottom, the next
        # N fill column 2 — exactly the layout we want when we pass keys
        # first then values in matching order.
        keys_list = list(dataset_metadata.keys())
        values_list = [str(v) for v in dataset_metadata.values()]
        meta_handles = [Line2D([], [], linestyle="none", marker="", color="none") for _ in range(2 * len(keys_list))]
        meta_labels = keys_list + values_list

        # Preserve the model legend so adding the metadata one doesn't drop it.
        ax.add_artist(legend1)
        legend2 = ax.legend(
            meta_handles,
            meta_labels,
            loc="upper left",
            # Sit just below the model legend, sharing its left edge so the
            # two stack cleanly to the right of the plot.
            bbox_to_anchor=(1.01, bbox1_axes.y0 - 0.02),
            frameon=True,
            fontsize=legend_fontsize,
            title="Metadata",
            title_fontsize=legend_fontsize,
            ncol=2,
            labelspacing=0.2,
            handlelength=0,
            handletextpad=0,
            borderpad=0.3,
            borderaxespad=0.0,
            columnspacing=0.8,
        )

    if ylim is not None:
        ax.set_ylim(ylim)

    # Clamp y_min to 0 when the auto / explicit limits would dip below
    # zero. Used by improvability and metric-error plots, where negative
    # values are nonsensical and just leave a stripe of empty space at
    # the bottom of the axis. Other axes (Elo, Baseline Advantage) opt
    # out by leaving ``clamp_negative_ymin`` False.
    if clamp_negative_ymin:
        cur_y_min, cur_y_max = ax.get_ylim()
        if cur_y_min < 0:
            ax.set_ylim(0, cur_y_max)

    ax.grid(True)
    grid_color = ax.xaxis.get_gridlines()[0].get_color()
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_color(grid_color)
    ax.spines["right"].set_color(grid_color)
    ax.spines["bottom"].set_color(grid_color)
    ax.spines["left"].set_color(grid_color)
    # Make major and minor tick lines gray, but labels stay black
    ax.tick_params(axis="both", which="both", color=grid_color, labelcolor="black")

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    # ``xlabel`` / ``ylabel`` double as DataFrame column names (used for the
    # peak-per-method bookkeeping above); ``*_display`` decouples the axis
    # caption from the column key when the caller wants a richer label.
    ax.set_ylabel(ylabel_display if ylabel_display is not None else ylabel, fontsize=17)
    ax.set_xlabel(xlabel_display if xlabel_display is not None else xlabel, fontsize=17)
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # ``bbox_inches="tight"`` lets savefig expand the output canvas to
    # include artists positioned outside the axes — without it the
    # ``bbox_to_anchor=(1.01, …)`` legends (method legend + dataset
    # metadata legend stacked below it) get clipped when the second
    # legend pushes the stack taller than the fixed figure height.
    # ``bbox_extra_artists`` is the matplotlib-recommended way to make
    # sure those external legends are picked up by the tight bbox
    # calculation — ``ax.add_artist(legend1)`` demotes it from "the
    # axes legend" to a regular artist, and on some matplotlib versions
    # those don't always get measured by ``get_tightbbox`` alone.
    extra_artists = [a for a in (legend1, legend2) if a is not None]
    fig.savefig(
        str(save_path),
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=extra_artists or None,
    )
    plt.close(fig)


def compute_tuning_trajectories_leaderboard(
    combined_data: pd.DataFrame,
    tabarena_context: TabArenaContext,
    methods_map: pd.DataFrame,
    calibration_framework: str,
    fillna_method: str,
    exclude_imputed: bool,
    elo_bootstrap_rounds: int = 1,
    average_seeds: bool = False,
    subset: str | list[str] | None = None,
    name_col="config_type",
    folds: list[int] | None = None,
):
    combined_data = combined_data.copy()
    if subset is not None or folds is not None:
        if isinstance(subset, str):
            subset = [subset]
        combined_data = subset_tasks(
            df_results=combined_data,
            subset=subset,
            folds=folds,
            task_metadata_og=tabarena_context.task_metadata,
            predicates=tabarena_context.subset_predicates,
        )

    tabarena_init_kwargs = dict(
        task_col="dataset",
        columns_to_agg_extra=[
            "time_train_s",
            "time_infer_s",
            "time_train_s_per_1K",
            "time_infer_s_per_1K",
            "time_total_s",
            "time_total_s_per_1K",
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

    # FIXME: This isn't correct
    # combined_data = arena.fillna_data(
    #     data=combined_data,
    #     fillna_method=calibration_framework,
    # )
    # FIXME: Using this since it does it correctly
    if fillna_method is not None:
        combined_data = tabarena_context.fillna_metrics(
            df_to_fill=combined_data,
            df_fillna=combined_data[combined_data["method"] == fillna_method],
        )

    if exclude_imputed:
        imputed_methods_count = combined_data.groupby("method")["imputed"].sum()
        imputed_methods = sorted(imputed_methods_count[imputed_methods_count > 0].index)
        print(f"Excluding {len(imputed_methods)} imputed methods: {imputed_methods}")
        combined_data = combined_data[~combined_data["method"].isin(imputed_methods)]

    arena.compute_results_per_task(data=combined_data)

    leaderboard = arena.leaderboard(
        data=combined_data,
        include_elo=True,
        include_error=True,
        elo_kwargs=dict(
            calibration_framework=calibration_framework,
            calibration_elo=1000,
            BOOTSTRAP_ROUNDS=elo_bootstrap_rounds,
        ),
        average_seeds=average_seeds,
        include_baseline_advantage=True,
    )

    # The bencheval validator rejects any null values in the error column,
    # so a method with even one NaN `metric_error_val` row would crash the
    # val leaderboard. Drop those methods from the val pass only — the
    # test leaderboard (`arena`) is unaffected. After the merge below the
    # dropped methods receive NaN for the val-derived columns
    # (elo_val / improvability_val / baseline_advantage_val) since the
    # column assignment aligns on the method index.
    val_nan_mask = combined_data["metric_error_val"].isna()
    val_nan_methods = sorted(combined_data.loc[val_nan_mask, "method"].unique().tolist())
    if val_nan_methods:
        print(
            f"Dropping {len(val_nan_methods)} method(s) from validation-score "
            f"leaderboard due to NaN metric_error_val: {val_nan_methods}",
        )
    combined_data_val = combined_data[~combined_data["method"].isin(val_nan_methods)]

    leaderboard_val = arena_val.leaderboard(
        data=combined_data_val,
        include_elo=True,
        include_error=True,
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

    leaderboard = leaderboard[leaderboard["method"].isin(methods_map.index)]
    leaderboard["n_configs"] = leaderboard["method"].map(methods_map["n_configs"])
    leaderboard[name_col] = leaderboard["method"].map(methods_map[name_col])

    leaderboard["name"] = leaderboard[name_col]

    leaderboard = leaderboard.sort_values(by=[name_col, "n_configs"])

    leaderboard["Elo"] = leaderboard["elo"]
    leaderboard["Elo (Test)"] = leaderboard["Elo"]
    leaderboard["Elo (Val)"] = leaderboard["elo_val"]
    leaderboard["Elo (Val) - Elo (Test)"] = leaderboard["Elo (Val)"] - leaderboard["Elo (Test)"]
    leaderboard["Elo (Test) - Elo (Val)"] = leaderboard["Elo (Test)"] - leaderboard["Elo (Val)"]
    leaderboard["Improvability (%)"] = leaderboard["improvability"] * 100
    leaderboard["Improvability (%) (Test)"] = leaderboard["Improvability (%)"]
    leaderboard["Improvability (%) (Val)"] = leaderboard["improvability_val"] * 100
    leaderboard["Improvability (%) (Test) - Improvability (%) (Val)"] = (
        leaderboard["Improvability (%) (Test)"] - leaderboard["Improvability (%) (Val)"]
    )

    leaderboard["Baseline Advantage (%)"] = leaderboard["baseline_advantage"] * 100
    leaderboard["Baseline Advantage (%) (Test)"] = leaderboard["Baseline Advantage (%)"]
    leaderboard["Baseline Advantage (%) (Val)"] = leaderboard["baseline_advantage_val"] * 100
    leaderboard["Baseline Advantage (%) (Test - Val)"] = (
        leaderboard["baseline_advantage"] - leaderboard["baseline_advantage_val"]
    ) * 100

    leaderboard["Train time per 1K samples (s) (median)"] = leaderboard["median_time_train_s_per_1K"]
    leaderboard["Inference time per 1K samples (s) (median)"] = leaderboard["median_time_infer_s_per_1K"]
    leaderboard["Total time per 1K samples (s) (median)"] = leaderboard["median_time_total_s_per_1K"]

    leaderboard["Train time (s)"] = leaderboard["time_train_s"]
    leaderboard["Infer time (s)"] = leaderboard["time_infer_s"]
    leaderboard["Total time (s)"] = leaderboard["time_total_s"]
    leaderboard["Metric Error"] = leaderboard["metric_error"]

    return leaderboard


def plot_tuning_trajectories_all(
    tabarena_context: TabArenaContext = None,
    fig_save_dir: str | Path = Path("plots") / "n_configs",
    ban_bad_methods: bool = True,
    file_ext: str = ".pdf",
    extra_results=None,
    calibration_framework="auto",
    folds: list[int] | None = None,
    methods_to_display: list[str] | None = None,
    plot_kwargs: dict | None = None,
    include_baselines: bool = False,
    engine: str = "auto",
    progress_bar: bool = True,
    use_elo_method_order: bool = True,
):
    if isinstance(fig_save_dir, str):
        fig_save_dir = Path(fig_save_dir)
    fig_save_dir.mkdir(parents=True, exist_ok=True)

    if engine == "auto":
        engine = tabarena_context.engine if tabarena_context is not None else "sequential"

    all_combinations = get_all_subset_combinations()

    # One job per sub-benchmark subset combination. Resolve the per-combination output
    # dir and subset filters up front (and create the dirs in the parent) so the shared,
    # read-only inputs (`tabarena_context`, ...) are passed once via `parallel_for`'s
    # `context` (ray's object store) instead of being serialized per job.
    inputs = []
    for (
        use_imputation,
        problem_type,
        _,
        dataset_subset,
        lite,
        average_seeds,
    ) in all_combinations:
        custom_folder_name = get_website_folder_name(
            use_imputation=use_imputation,
            problem_type=problem_type,
            dataset_subset=dataset_subset,
            lite=lite,
        )
        subset_list = []
        if problem_type != "all":
            subset_list.append(problem_type)
        if dataset_subset is not None:
            subset_list.append(dataset_subset)
        if lite:
            subset_list.append("lite")
        (fig_save_dir / custom_folder_name).mkdir(parents=True, exist_ok=True)

        inputs.append(
            {
                "subset_map": {"placeholder_name": subset_list},
                "average_seeds": average_seeds,
                "exclude_imputed": not use_imputation,
                "fig_save_dir": fig_save_dir / custom_folder_name / "tuning_trajectories",
            }
        )

    parallel_for(
        f=plot_tuning_trajectories,
        inputs=inputs,
        context={
            "tabarena_context": tabarena_context,
            "ban_bad_methods": ban_bad_methods,
            "file_ext": file_ext,
            "extra_results": extra_results,
            "calibration_framework": calibration_framework,
            "folds": folds,
            "methods_to_display": methods_to_display,
            "plot_kwargs": plot_kwargs,
            "include_baselines": include_baselines,
            "use_elo_method_order": use_elo_method_order,
        },
        engine=engine,
        progress_bar=progress_bar,
        desc="Generating tuning trajectories per subset",
    )


def _build_dataset_metadata_map(
    task_metadata: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    """Build a ``dataset_id -> {label: value}`` mapping suitable for the
    second 'Dataset' legend on each per-dataset Pareto plot.

    Picks the first row per ``dataset`` in ``task_metadata`` (the frame is
    expected to already be one-row-per-dataset). Pulls warehouse-sourced
    fields (``num_cols_after_preprocessing``, ``domain``, ``dataset_year``,
    ``source``, ``missing_value_fraction``) when present; falls back
    silently when a column is unavailable. Returns an empty dict if
    ``task_metadata`` is missing or has no ``dataset`` column.
    """
    if task_metadata is None or "dataset" not in task_metadata.columns:
        return {}

    def _split_type(row: pd.Series) -> str:
        if pd.notna(row.get("time_on")):
            return "temporal"
        if pd.notna(row.get("group_on")):
            return "grouped"
        return "IID"

    def _fmt_int(value) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "?"
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return str(value)

    def _fmt_str(value) -> str | None:
        """Return ``None`` (signaling 'omit row') when value is missing,
        else a stringified value. Distinct from ``_fmt_int`` because text
        fields shouldn't print ``?`` — better to drop the line entirely
        than show ``Domain: ?`` when the warehouse simply doesn't have it.
        """
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        s = str(value).strip()
        return s if s else None

    def _fmt_pct(value) -> str | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        try:
            return f"{float(value) * 100:.1f}%"
        except (TypeError, ValueError):
            return None

    out: dict[str, dict[str, str]] = {}
    for _, row in task_metadata.drop_duplicates(subset="dataset").iterrows():
        problem_type = str(row.get("problem_type", "?"))

        # ``num_cols_after_preprocessing`` is the warehouse-side column count
        # (post one-hot etc.). When that's not present the plot falls back
        # to the raw ``num_features`` so older task_metadata frames still
        # get *some* number rather than ``?``.
        cols_value = row.get("num_cols_after_preprocessing")
        if cols_value is None or (isinstance(cols_value, float) and pd.isna(cols_value)):
            cols_value = row.get("num_features")

        meta: dict[str, str] = {
            "Samples": _fmt_int(row.get("num_instances")),
            "Features": _fmt_int(cols_value),
            "Problem": problem_type,
        }
        # Only multiclass tasks have a meaningful class count; binary/regression
        # would just clutter the legend.
        if problem_type == "multiclass":
            meta["Classes"] = _fmt_int(row.get("num_classes"))
        meta["Splits"] = _fmt_int(row.get("n_splits"))
        meta["Type"] = _split_type(row)

        # The remaining warehouse fields are optional — drop the line when
        # missing so each plot only shows rows for columns we actually have.
        missing_pct = _fmt_pct(row.get("missing_value_fraction"))
        if missing_pct is not None:
            meta["Missingness"] = missing_pct
        domain = _fmt_str(row.get("domain"))
        if domain is not None:
            meta["Domain"] = domain
        source = _fmt_str(row.get("source"))
        if source is not None:
            meta["Source"] = source
        year = row.get("dataset_year")
        if year is not None and not (isinstance(year, float) and pd.isna(year)):
            try:
                meta["Year"] = str(int(year))
            except (TypeError, ValueError):
                meta["Year"] = str(year)

        out[row["dataset"]] = meta
    return out


def plot_tuning_trajectories_per_dataset(
    tabarena_context: TabArenaContext,
    fig_save_dir: str | Path = Path("plots") / "n_configs_per_dataset",
    ban_bad_methods: bool = True,
    file_ext: str = ".pdf",
    extra_results=None,
    calibration_framework: str | None = "auto",
    fillna_method: str | None = "auto",
    folds: list[int] | None = None,
    methods_to_display: list[str] | None = None,
    plot_kwargs: dict | None = None,
    include_baselines: bool = False,
    engine: str = "auto",
    progress_bar: bool = True,
):
    if isinstance(fig_save_dir, str):
        fig_save_dir = Path(fig_save_dir)
    fig_save_dir.mkdir(parents=True, exist_ok=True)

    if engine == "auto":
        engine = tabarena_context.engine
    if isinstance(calibration_framework, str) and calibration_framework == "auto":
        calibration_framework = tabarena_context.calibration_method
    if isinstance(fillna_method, str) and fillna_method == "auto":
        fillna_method = tabarena_context.fillna_method

    method_rename_map = get_method_rename_map()  # TODO: avoid hard-coding

    # Heavy I/O — load HPO trajectories, baselines, and assemble combined_data once and share
    # across all per-dataset jobs (passed to ray's object store via parallel_for's `context`).
    combined_data, methods_map = _prepare_tuning_trajectories_data(
        tabarena_context=tabarena_context,
        extra_results=extra_results,
        methods_to_display=methods_to_display,
        include_baselines=include_baselines,
    )

    datasets = sorted(tabarena_context.task_metadata["dataset"].unique())

    use_imputation = False
    problem_type = "all"
    dataset_subset = None
    lite = False
    average_seeds = False

    subset_list = []
    if problem_type != "all":
        subset_list.append(problem_type)
    if dataset_subset is not None:
        subset_list.append(dataset_subset)
    if lite:
        subset_list.append("lite")

    # Build a one-shot dataset_id -> human-readable label map so the figure
    # title reads e.g. "airfoil_self_noise" rather than "Task-7163328506".
    # Prefers ``dataset_name`` (BeyondArena), falls back to ``name``
    # (default TabArena), finally to identity.
    _tm = tabarena_context.task_metadata
    if "dataset_name" in _tm.columns:
        _name_map = _tm.set_index("dataset")["dataset_name"].astype(str).to_dict()
    elif "name" in _tm.columns:
        _name_map = _tm.set_index("dataset")["name"].astype(str).to_dict()
    else:
        _name_map = {}

    # Per-dataset eval metric, used to fill in the y-axis label of the
    # ``pareto_n_configs_err_*`` plots as ``Test Error (<metric>)``. Each
    # dataset has exactly one ``eval_metric`` (verified via groupby), but
    # ``drop_duplicates`` keeps us robust to duplicated rows.
    if "eval_metric" in _tm.columns:
        _metric_map = (
            _tm[["dataset", "eval_metric"]]
            .drop_duplicates(subset="dataset")
            .set_index("dataset")["eval_metric"]
            .astype(str)
            .to_dict()
        )
    else:
        _metric_map = {}

    # Per-dataset metadata block rendered as a second legend on each Pareto
    # plot (rows / cols / problem type / splits / IID-vs-grouped-vs-temporal).
    _metadata_map = _build_dataset_metadata_map(_tm)

    inputs = []
    for dataset in datasets:
        (fig_save_dir / dataset).mkdir(parents=True, exist_ok=True)

        plot_kwargs_cur = copy.deepcopy(plot_kwargs)
        if plot_kwargs_cur is None:
            plot_kwargs_cur = {}
        plot_kwargs_cur["title"] = f"Dataset: {_name_map.get(dataset, dataset)}"

        inputs.append(
            {
                "datasets": [dataset],
                "fig_save_dir": fig_save_dir / dataset / "tuning_trajectories",
                "subset_map": list(subset_list),
                "plot_kwargs": plot_kwargs_cur,
                "error_ylabel_metric": _metric_map.get(dataset),
                "dataset_metadata": _metadata_map.get(dataset),
            }
        )

    parallel_for(
        f=_plot_tuning_trajectories_from_prepared,
        inputs=inputs,
        context={
            "combined_data": combined_data,
            "methods_map": methods_map,
            "tabarena_context": tabarena_context,
            "calibration_framework": calibration_framework,
            "fillna_method": fillna_method,
            "average_seeds": average_seeds,
            "exclude_imputed": not use_imputation,
            "elo_bootstrap_rounds": 1,
            "name_col": "config_type",
            "method_rename_map": method_rename_map,
            "ban_bad_methods": ban_bad_methods,
            "file_ext": file_ext,
            "folds": folds,
        },
        engine=engine,
        progress_bar=progress_bar,
        desc="Plotting tuning trajectories per dataset",
    )


def _prepare_tuning_trajectories_data(
    tabarena_context: TabArenaContext,
    extra_results: pd.DataFrame | None,
    methods_to_display: list[str] | None,
    include_baselines: bool,
    include_portfolio: bool = False,
    include_hpo_seeds: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the dataset-independent inputs (combined_data, methods_map) for tuning-trajectory plotting.

    Heavy I/O (loading per-method HPO trajectories, baselines, the paper results) lives here so the
    caller can run it once and share the results across many per-dataset plotting calls.
    """
    method_metadata_lst_og = tabarena_context.method_metadata_collection.method_metadata_lst
    method_metadata_lst = [m for m in method_metadata_lst_og if m.method_type == "config"]
    results_hpo_lst = []
    for m in method_metadata_lst:
        if methods_to_display is not None and m.method not in methods_to_display:
            continue
        results_hpo_trajectory = m.load_hpo_trajectories()
        results_hpo_trajectory["display_name"] = m.display_name
        results_hpo_lst.append(results_hpo_trajectory)
    if extra_results is not None:
        extra_results = extra_results.copy(deep=True)
        if "n_configs" not in extra_results.columns:
            extra_results["n_configs"] = np.nan
        if "n_iterations" not in extra_results.columns:
            extra_results["n_iterations"] = np.nan
        extra_results["n_configs"] = extra_results["n_configs"].fillna(1)
        extra_results["n_iterations"] = extra_results["n_iterations"].fillna(1)

        extra_results["config_type"] = extra_results["method"]
        extra_results["method"] = extra_results["method"] + "-" + extra_results["n_configs"].astype(str)
        results_hpo_lst.append(extra_results)

    results_hpo = pd.concat(results_hpo_lst, ignore_index=True)
    results_hpo["display_name"] = results_hpo["display_name"].fillna(results_hpo["config_type"])

    result_baselines = tabarena_context.load_results_paper()

    results_hpo_mean = (
        results_hpo.copy()
        .groupby(["method", "dataset", "fold", "problem_type", "metric", "config_type", "display_name"])
        .mean(
            numeric_only=True,
        )
        .drop(columns=["seed"])
        .reset_index()
    )
    results_hpo_mean["imputed"] = 0
    results_hpo_mean["imputed"] = results_hpo_mean["imputed"].astype(bool)

    results_lst = [
        results_hpo_mean,
    ]

    if include_baselines:
        results_baselines = []
        method_metadata_lst_baselines = [m for m in method_metadata_lst_og if m.method_type == "baseline"]
        for m in method_metadata_lst_baselines:
            if methods_to_display is not None and m.method not in methods_to_display:
                continue
            results_baseline = m.load_model_results()
            results_baseline["n_configs"] = 1
            results_baseline["n_iterations"] = 1
            results_baseline["config_type"] = results_baseline["method"]
            results_baseline["method"] = results_baseline["method"] + "-" + results_baseline["n_configs"].astype(str)
            results_baselines.append(results_baseline)

        results_lst += results_baselines

    if include_hpo_seeds:
        results_hpo_seeds = results_hpo.copy()
        results_hpo_seeds["method"] = results_hpo_seeds["method"] + "-" + results_hpo_seeds["seed"].astype(str)
        results_hpo_seeds["config_type"] = (
            results_hpo_seeds["config_type"] + "-" + results_hpo_seeds["seed"].astype(str)
        )
        results_hpo_seeds = (
            results_hpo_seeds.groupby(
                ["method", "dataset", "fold", "problem_type", "metric", "config_type", "display_name"]
            )
            .mean(numeric_only=True)
            .reset_index()
        )
        results_lst.append(results_hpo_seeds)

    if include_portfolio:  # TODO: This only works if you have legacy files, add general support for portfolio
        results_portfolio = load_pd.load(path="../tabarena/advanced/rebuttal/rebuttal_portfolio_n_configs.parquet")
        results_portfolio["config_type"] = results_portfolio["method"]
        results_portfolio = results_portfolio.rename(
            columns={
                "n_portfolio": "n_configs",
                "n_ensembles": "n_iterations",
            }
        )
        results_portfolio["method"] = results_portfolio["method"] + "-" + results_portfolio["n_configs"].astype(str)
        results_portfolio["imputed"] = 0
        results_portfolio["imputed"] = results_portfolio["imputed"].astype(bool)
        results_lst.append(results_portfolio)

    results_hpo = pd.concat(results_lst, ignore_index=True)

    results_hpo["display_name"] = results_hpo["display_name"].fillna(results_hpo["config_type"])
    results_hpo["config_type"] = results_hpo["display_name"]

    combined_data = pd.concat([result_baselines, results_hpo], ignore_index=True)

    # ----- add times per 1K samples -----
    dataset_to_n_samples_train = tabarena_context.task_metadata.set_index("name")["n_samples_train_per_fold"].to_dict()
    dataset_to_n_samples_test = tabarena_context.task_metadata.set_index("name")["n_samples_test_per_fold"].to_dict()

    combined_data["time_train_s_per_1K"] = (
        combined_data["time_train_s"] * 1000 / combined_data["dataset"].map(dataset_to_n_samples_train)
    )
    combined_data["time_infer_s_per_1K"] = (
        combined_data["time_infer_s"] * 1000 / combined_data["dataset"].map(dataset_to_n_samples_test)
    )

    # Combined train + inference runtime, both raw and per-1K-samples.
    # Computed at the row level (not as the sum of medians) so the
    # downstream median aggregation reflects the median total wall-time
    # of one method-run, not an artifact of summing two separate medians.
    combined_data["time_total_s"] = combined_data["time_train_s"] + combined_data["time_infer_s"]
    combined_data["time_total_s_per_1K"] = combined_data["time_train_s_per_1K"] + combined_data["time_infer_s_per_1K"]

    methods_map = (
        results_hpo[["method", "n_configs", "n_iterations", "config_type"]]
        .drop_duplicates(subset=["method"])
        .set_index("method")
    )

    return combined_data, methods_map


def _plot_tuning_trajectories_from_prepared(
    combined_data: pd.DataFrame,
    methods_map: pd.DataFrame,
    subset_map: dict[str, list[str]] | list[str],
    fig_save_dir: str | Path,
    tabarena_context: TabArenaContext,
    calibration_framework,
    fillna_method,
    average_seeds: bool,
    exclude_imputed: bool,
    elo_bootstrap_rounds: int,
    name_col: str,
    method_rename_map: dict,
    ban_bad_methods: bool,
    file_ext: str,
    folds: list[int] | None,
    plot_kwargs: dict | None,
    datasets: list[str] | None = None,
    error_ylabel_metric: str | None = None,
    dataset_metadata: dict[str, str] | None = None,
    hidden_methods: list[str] | None = None,
    show_titles: bool = False,
    show_coverage_legend: bool = False,
    subset_display_names: dict[str, str] | None = None,
    use_elo_method_order: bool = True,
):
    """Run the per-(dataset-subset, subset_map) leaderboard + plotting steps from already-prepared data."""
    fig_save_dir = Path(fig_save_dir)

    if datasets is not None:
        combined_data = combined_data[combined_data["dataset"].isin(datasets)]

    if isinstance(subset_map, list):
        subset_map = {None: subset_map}
    for subset_name, subset in subset_map.items():
        leaderboard = compute_tuning_trajectories_leaderboard(
            combined_data=combined_data,
            tabarena_context=tabarena_context,
            methods_map=methods_map,
            calibration_framework=calibration_framework,
            fillna_method=fillna_method,
            exclude_imputed=exclude_imputed,
            elo_bootstrap_rounds=elo_bootstrap_rounds,
            average_seeds=average_seeds,
            subset=subset,
            name_col=name_col,
            folds=folds,
        )
        leaderboard["name"] = leaderboard["name"].map(method_rename_map).fillna(leaderboard["name"])

        if ban_bad_methods:
            bad_methods = ["KNN", "Linear", "PerpetualBooster", "TabSTAR", "TabFlex"]
            leaderboard = leaderboard[~leaderboard["config_type"].isin(bad_methods)]

        if hidden_methods:
            leaderboard = leaderboard[~leaderboard["config_type"].isin(hidden_methods)]

        fig_save_dir_subset = fig_save_dir / subset_name if subset_name is not None else fig_save_dir

        # Build the coverage-counts metadata for the second legend whenever
        # *either* the titles or the dedicated coverage-legend toggle is
        # on. Decoupled so the script can show the legend without paying
        # for a title (and vice versa). The same subset filter that
        # ``compute_tuning_trajectories_leaderboard`` applied internally
        # is re-applied here so the legend reflects what's actually in
        # the plot. Reuses the existing ``dataset_metadata`` channel in
        # ``plot_hpo`` (keys/values render as a stacked legend block
        # under the method legend).
        subset_dataset_metadata = dataset_metadata
        subset_title_prefix: str | None = None
        if show_titles or show_coverage_legend:
            filtered = subset_tasks(
                df_results=combined_data,
                subset=subset,
                folds=folds,
                task_metadata_og=tabarena_context.task_metadata,
                predicates=tabarena_context.subset_predicates,
            )

        if show_coverage_legend:
            n_datasets = int(filtered["dataset"].nunique())
            n_tasks = len(filtered[["dataset", "fold"]].drop_duplicates())

            # If the caller already supplied per-plot metadata, append the
            # coverage counts so both render in the same legend block.
            base_meta: dict[str, str] = dict(dataset_metadata or {})
            base_meta.setdefault("Datasets", f"{n_datasets:,}")
            base_meta.setdefault("Tasks", f"{n_tasks:,}")
            subset_dataset_metadata = base_meta

        if show_titles:
            # ``subset_name`` is the human-readable label from ``subset_map``;
            # fall back to joining the raw predicate list when no name was
            # supplied. Apply ``subset_display_names`` (if provided) to each
            # raw predicate so titles read with caller-chosen labels rather
            # than the raw subset keys. Empty list / None subset → no prefix.
            if subset_name:
                subset_title_prefix = subset_name
            elif subset:
                parts = [subset_display_names.get(p, p) for p in subset] if subset_display_names else list(subset)
                subset_title_prefix = ", ".join(parts)

        plot_tuning_trajectories_from_leaderboard(
            leaderboard=leaderboard,
            fig_save_dir=fig_save_dir_subset,
            file_ext=file_ext,
            plot_kwargs=plot_kwargs,
            error_ylabel_metric=error_ylabel_metric,
            dataset_metadata=subset_dataset_metadata,
            show_titles=show_titles,
            title_prefix=subset_title_prefix,
            use_elo_method_order=use_elo_method_order,
        )


def plot_tuning_trajectories(
    tabarena_context: TabArenaContext = None,
    subset_map: dict[str, list[str]] | list[str] | None = None,
    fig_save_dir: str | Path = Path("plots") / "n_configs",
    average_seeds: bool = False,
    exclude_imputed: bool = True,
    ban_bad_methods: bool = True,
    include_baselines: bool = False,
    include_portfolio: bool = False,  # TODO: True not yet supported
    file_ext: str = ".pdf",
    extra_results: pd.DataFrame | None = None,
    datasets: list[str] | None = None,
    calibration_framework="auto",
    fillna_method="auto",
    folds: list[int] | None = None,
    methods_to_display: list[str] | None = None,
    hidden_methods: list[str] | None = None,
    plot_kwargs: dict | None = None,
    show_titles: bool = False,
    show_coverage_legend: bool = False,
    subset_display_names: dict[str, str] | None = None,
    use_elo_method_order: bool = True,
):
    name_col = "config_type"
    if subset_map is None:
        subset_map = []

    if tabarena_context is None:
        tabarena_context = TabArenaContext(
            include_unverified=True,
        )
    if isinstance(calibration_framework, str) and calibration_framework == "auto":
        calibration_framework = tabarena_context.calibration_method
    if isinstance(fillna_method, str) and fillna_method == "auto":
        fillna_method = tabarena_context.fillna_method

    fig_save_dir = Path(fig_save_dir)

    elo_bootstrap_rounds = 1
    method_rename_map = get_method_rename_map()  # TODO: avoid hard-coding

    combined_data, methods_map = _prepare_tuning_trajectories_data(
        tabarena_context=tabarena_context,
        extra_results=extra_results,
        methods_to_display=methods_to_display,
        include_baselines=include_baselines,
        include_portfolio=include_portfolio,
        include_hpo_seeds=False,
    )

    _plot_tuning_trajectories_from_prepared(
        combined_data=combined_data,
        methods_map=methods_map,
        subset_map=subset_map,
        fig_save_dir=fig_save_dir,
        tabarena_context=tabarena_context,
        calibration_framework=calibration_framework,
        fillna_method=fillna_method,
        average_seeds=average_seeds,
        exclude_imputed=exclude_imputed,
        elo_bootstrap_rounds=elo_bootstrap_rounds,
        name_col=name_col,
        method_rename_map=method_rename_map,
        ban_bad_methods=ban_bad_methods,
        file_ext=file_ext,
        folds=folds,
        plot_kwargs=plot_kwargs,
        datasets=datasets,
        hidden_methods=hidden_methods,
        show_titles=show_titles,
        show_coverage_legend=show_coverage_legend,
        subset_display_names=subset_display_names,
        use_elo_method_order=use_elo_method_order,
    )


_TEST_ERROR_METRIC_DISPLAY = {
    "log_loss": "logloss",
    # ``metric_error`` for ROC AUC is ``1 - AUC``; spell that out on the
    # y-axis so the plot reads as the actual quantity being plotted.
    "roc_auc": "1-AUC",
    "rmse": "RMSE",
    "root_mean_squared_error": "RMSE",
}


def _test_error_ylabel(metric: str | None) -> str:
    """Format the y-axis label for the metric-error Pareto plots. When
    ``metric`` is provided, returns ``"Test Error (<friendly>)"`` using a
    short friendly name (matching the per-dataset table captions); otherwise
    falls back to ``"Test Error"`` so the rename still applies even when no
    single metric describes the data.
    """
    if not metric:
        return "Test Error"
    friendly = _TEST_ERROR_METRIC_DISPLAY.get(metric, metric)
    return f"Test Error ({friendly})"


def plot_tuning_trajectories_from_leaderboard(
    leaderboard: pd.DataFrame,
    fig_save_dir: Path,
    file_ext: str = ".pdf",
    plot_kwargs: dict | None = None,
    error_ylabel_metric: str | None = None,
    dataset_metadata: dict[str, str] | None = None,
    show_titles: bool = False,
    title_prefix: str | None = None,
    use_elo_method_order: bool = True,
):
    if plot_kwargs is None:
        plot_kwargs = {}
    plot_kwargs = plot_kwargs.copy()
    plot_kwargs.setdefault("sort_col", "n_configs")
    plot_kwargs.setdefault("ylim_imp", (0, None))
    # Threaded into every ``plot_hpo`` call below; harmless when ``None``.
    plot_kwargs["dataset_metadata"] = dataset_metadata

    # Single shared title across every ``plot_hpo`` call in this set.
    # ``TabArena-<subset>`` prefix matches the format used by the winrate
    # matrix and tuning-impact bar plots so the three surfaces read as
    # one report. ``None`` and ``"all"`` collapse to the unsuffixed
    # ``TabArena`` prefix (aggregate / no-meaningful-subset case).
    if show_titles:
        if title_prefix and title_prefix != "all":
            plot_kwargs["title"] = f"TabArena-{title_prefix} Pareto Frontier"
        else:
            plot_kwargs["title"] = "TabArena Pareto Frontier"

    # Pin a single canonical method ordering for every plot in this set so
    # colors and legend positions stay identical across the err / improvability
    # / Elo Pareto plots. Without this each plot ranks methods by its own
    # y-axis, causing the same method to flip color and legend slot between
    # plots. Rank by best Elo per method (descending) since Elo is the
    # canonical "higher is better" summary; fall back gracefully when the
    # column or method column isn't present. Disable via
    # ``use_elo_method_order=False`` to let every plot order methods by its own
    # y-axis instead (an explicit ``plot_kwargs["method_order"]`` still wins).
    if use_elo_method_order and "method_order" not in plot_kwargs:
        method_col = plot_kwargs.get("method_col", "name")
        if "Elo" in leaderboard.columns and method_col in leaderboard.columns:
            plot_kwargs["method_order"] = (
                leaderboard.groupby(method_col)["Elo"].max().sort_values(ascending=False).index.tolist()
            )

    ylim_imp = plot_kwargs.pop("ylim_imp")
    err_ylabel = _test_error_ylabel(error_ylabel_metric)

    plot_hpo(
        df=leaderboard,
        xlabel="Train time (s)",
        ylabel="Improvability (%)",
        save_path=fig_save_dir / f"pareto_n_configs_imp_tot_train{file_ext}",
        max_Y=False,
        ylim=ylim_imp,
        clamp_negative_ymin=True,
        **plot_kwargs,
    )
    plot_hpo(
        df=leaderboard,
        xlabel="Infer time (s)",
        ylabel="Improvability (%)",
        save_path=fig_save_dir / f"pareto_n_configs_imp_tot_infer{file_ext}",
        max_Y=False,
        ylim=ylim_imp,
        clamp_negative_ymin=True,
        **plot_kwargs,
    )
    plot_hpo(
        df=leaderboard,
        xlabel="Train time (s)",
        ylabel="Metric Error",
        ylabel_display=err_ylabel,
        save_path=fig_save_dir / f"pareto_n_configs_err_tot_train{file_ext}",
        max_Y=False,
        # ylim=ylim_imp,
        clamp_negative_ymin=True,
        **plot_kwargs,
    )
    plot_hpo(
        df=leaderboard,
        xlabel="Infer time (s)",
        ylabel="Metric Error",
        ylabel_display=err_ylabel,
        save_path=fig_save_dir / f"pareto_n_configs_err_tot_infer{file_ext}",
        max_Y=False,
        # ylim=ylim_imp,
        clamp_negative_ymin=True,
        **plot_kwargs,
    )
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
        ylim=ylim_imp,
        clamp_negative_ymin=True,
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
        ylim=ylim_imp,
        clamp_negative_ymin=True,
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

    # ----- combined train + inference runtime variants -----
    # Mirror the train- and infer-only pareto plots above against the
    # row-level total runtime (train + inference).  Useful when ranking
    # methods by end-to-end deployment cost rather than either phase
    # in isolation.
    plot_hpo(
        df=leaderboard,
        xlabel="Total time (s)",
        ylabel="Improvability (%)",
        save_path=fig_save_dir / f"pareto_n_configs_imp_tot_total{file_ext}",
        max_Y=False,
        ylim=ylim_imp,
        clamp_negative_ymin=True,
        **plot_kwargs,
    )
    plot_hpo(
        df=leaderboard,
        xlabel="Total time (s)",
        ylabel="Metric Error",
        ylabel_display=err_ylabel,
        save_path=fig_save_dir / f"pareto_n_configs_err_tot_total{file_ext}",
        max_Y=False,
        clamp_negative_ymin=True,
        **plot_kwargs,
    )
    plot_hpo(
        df=leaderboard,
        xlabel="Total time per 1K samples (s) (median)",
        ylabel="Elo",
        save_path=fig_save_dir / f"pareto_n_configs_elo_total{file_ext}",
        max_Y=True,
        **plot_kwargs,
    )
    plot_hpo(
        df=leaderboard,
        xlabel="Total time per 1K samples (s) (median)",
        ylabel="Improvability (%)",
        save_path=fig_save_dir / f"pareto_n_configs_imp_total{file_ext}",
        max_Y=False,
        ylim=ylim_imp,
        clamp_negative_ymin=True,
        **plot_kwargs,
    )
    plot_hpo(
        df=leaderboard,
        xlabel="Total time per 1K samples (s) (median)",
        ylabel="Baseline Advantage (%)",
        save_path=fig_save_dir / f"pareto_n_configs_adv_total{file_ext}",
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
        xlabel="Baseline Advantage (%) (Test)",
        ylabel="Baseline Advantage (%) (Test - Val)",
        save_path=fig_save_dir / f"pareto_n_configs_adv_overfit_v2{file_ext}",
        max_Y=True,
        max_X=True,
        xlog=False,
        rank_by_y=False,
        **plot_kwargs,
    )

    plot_hpo(
        df=leaderboard,
        xlabel="Elo (Test)",
        ylabel="Elo (Test) - Elo (Val)",
        save_path=fig_save_dir / f"pareto_n_configs_elo_overfit_v2{file_ext}",
        max_Y=True,
        max_X=True,
        xlog=False,
        rank_by_y=False,
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
    plot_tuning_trajectories()
