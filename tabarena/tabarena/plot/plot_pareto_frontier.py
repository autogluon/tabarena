from __future__ import annotations

import os

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
import seaborn as sns


def aggregate_stats(df, on: str, groupby="method", method=["mean", "median", "std"]):
    return df[[groupby, on]].groupby(groupby).agg(method)[on]


def get_pareto_frontier(
    Xs,
    Ys,
    names=None,            # ← new
    *,
    max_X=True,
    max_Y=True,
    include_boundary_edges=True,
):
    """
    Compute the (piece‑wise constant) Pareto frontier and, in parallel,
    return the label associated with each frontier vertex.

    Parameters
    ----------
    Xs, Ys : Sequence[float]
        Coordinates of the points to consider.
    names : Sequence[str] or None, optional
        Label for each (X, Y) pair – e.g. `data["method"]`.
        If omitted, a list of ``None`` is used.
    max_X, max_Y : bool, default True
        If True the frontier favours larger values on that axis,
        otherwise it favours smaller values.
    include_boundary_edges : bool, default True
        If True, will include pareto front edges to the worst x and y values observed.

    Returns
    -------
    pareto_front : list[tuple[float, float]]
        The vertices that define the frontier, including the vertical
        “drop” segments needed for a step‑like plot.
    pareto_names : list[str | None]
        A label for each element in ``pareto_front``.
        Entries corresponding to the artificially inserted vertical
        drops are ``None``.
    """
    if names is None:
        names = [None] * len(Xs)
    if not (len(Xs) == len(Ys) == len(names)):
        raise ValueError("Xs, Ys and names must have the same length")

    # Sort primarily by X (descending if we maximise), secondarily by Y.
    pts = sorted(
        zip(Xs, Ys, names),
        key=lambda t: (t[0], t[1]),
        reverse=max_X,
    )

    pareto_front = [(pts[0][0], pts[0][1])]
    pareto_names = [pts[0][2]]
    best_y = pts[0][1]
    worst_y = pts[0][1]

    for x, y, label in pts[1:]:
        is_better = (y >= best_y) if max_Y else (y <= best_y)

        if is_better:
            # vertical segment to keep the frontier piece‑wise constant in X
            pareto_front.append((x, best_y))
            pareto_names.append(None)
            pareto_front.append((x, y))
            pareto_names.append(label)
            best_y = y

        is_worst = (y < worst_y) if max_Y else (y > worst_y)
        if is_worst:
            worst_y = y

    if include_boundary_edges:
        # add final horizontal segment to the worst point on X‑axis
        pareto_front.append((pts[-1][0], best_y))
        pareto_names.append(None)

        # add final vertical segment to the worst point on Y‑axis
        pareto_front.insert(0, (pts[0][0], worst_y))
        pareto_names.insert(0, None)

    return pareto_front, pareto_names


def plot_pareto(
    data: pd.DataFrame,
    x_name: str,
    y_name: str,
    title: str,
    palette='tab20',
    hue: str = "Method",
    *,
    aspect: float = 1.0,
    style_col: str | None = None,
    style_order: list[str] | None = None,
    style_markers: list[str] | dict | None = None,
    label_col: str = "Method",
    max_X: bool = False,
    max_Y: bool = True,
    sort_y: bool = False,
    ylim=None,
    save_path: str | None = None,
    add_optimal_arrow: bool = True,
    show: bool = True,
    legend_in_plot: bool = True,
):
    fig_size_ratio = 0.45
    fig_height = 10 * fig_size_ratio

    if sort_y:
        # Optionally sort for nicer vertical label spacing while preserving stable colors
        plot_df = data.sort_values(by=y_name, ascending=not max_Y)

        # ------------------------------
        # Compute hue_order inside here:
        # ------------------------------
        # For each hue category (e.g., method_type), take the "best" y value:
        #   - max if higher is better (max_Y=True)
        #   - min if lower is better (max_Y=False)
        agg_fun = "max" if max_Y else "min"
        y_per_hue = (plot_df.groupby(hue)[y_name]
                              .agg(agg_fun)
                              .sort_values(ascending=False))
        hue_order = list(y_per_hue.index)
    else:
        plot_df = data.copy()
        hue_order = None

    # Build stable color mapping per hue category (here: method_type)
    hue_levels = list(pd.unique(plot_df[hue]))

    # Ensure ≥20 visually distinct colors for many method types
    base_palette = sns.color_palette(palette, 20)
    if len(hue_levels) > 20:
        # Extend deterministically by combining tab20 + tab20b + tab20c if needed
        extended_palette = (
                sns.color_palette("tab20", 20)
                + sns.color_palette("tab20b", 20)
                + sns.color_palette("tab20c", 20)
        )
        colors = extended_palette[:len(hue_levels)]
    else:
        colors = base_palette[:len(hue_levels)]
    palette_map = dict(zip(hue_levels, colors))

    label_to_hue_dict = data.set_index(label_col)[hue].to_dict()
    label_to_color_dict = {l: palette_map[h] for l, h in label_to_hue_dict.items()}

    # Style (marker) mapping per run_type (optional; seaborn can auto-assign markers if you omit this dict)
    if style_col is not None:
        if style_order is None:
            style_order = list(pd.unique(plot_df[style_col]))
        if style_markers is None:
            markers_arg = True
        else:
            if isinstance(style_markers, list):
                markers_arg = {lvl: style_markers[i % len(style_markers)] for i, lvl in enumerate(style_order)}
            else:
                markers_arg = style_markers
            valid_vals = pd.unique(plot_df[style_col])
            style_order = [s for s in style_order if s in valid_vals]
            markers_arg = {k: v for k, v in markers_arg.items() if k in valid_vals}
    else:
        markers_arg = None

    g = sns.relplot(
        x=x_name,
        y=y_name,
        data=plot_df,
        hue=hue,
        hue_order=hue_order,
        palette=palette_map,
        style=style_col,
        style_order=style_order,
        markers=markers_arg,
        height=fig_height,
        aspect=aspect,
        s=150,
        alpha=0.8,
        linewidth=0.1,
        edgecolor="black",
        legend=False,
        zorder=2,
    )

    ax = g.ax
    ax.set_xlabel(x_name, fontsize=17)
    ax.set_ylabel(y_name, fontsize=17)
    ax.tick_params(axis='both', labelsize=9)

    # Compute Pareto frontier (use the plotted order)
    Xs = list(plot_df[x_name])
    Ys = list(plot_df[y_name])
    labels_for_front = list(plot_df[label_col])  # annotate with full method names
    pareto_front, pareto_names = get_pareto_frontier(
        Xs=Xs, Ys=Ys, names=labels_for_front, max_X=max_X, max_Y=max_Y
    )

    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]

    g.set(xscale="log")

    if ylim is not None:
        ax.set_ylim(ylim)

    # Draw Pareto frontier
    # Get current axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    if max_X:
        pf_X_first = x_min
        pf_X_last = pf_X[-1]
        pf_Y_first = pf_Y[0]
        if max_Y:
            pf_Y_last = y_min
        else:
            pf_Y_last = y_max
    else:
        pf_X_first = pf_X[0]
        pf_X_last = x_max
        pf_Y_last = pf_Y[-1]
        if max_Y:
            pf_Y_first = y_min
        else:
            pf_Y_first = y_max

    pf_X = [pf_X_first] + pf_X + [pf_X_last]
    pf_Y = [pf_Y_first] + pf_Y + [pf_Y_last]

    if add_optimal_arrow:
        plot_optimal_arrow(ax=ax, max_X=max_X, max_Y=max_Y, size=fig_size_ratio, scale=1.2)

    ax.plot(pf_X, pf_Y, linewidth=2 * fig_size_ratio, zorder=1, color='black', linestyle='--')

    ax.grid(True, zorder=-2)
    grid_color = ax.xaxis.get_gridlines()[0].get_color()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_color(grid_color)
    ax.spines['right'].set_color(grid_color)
    ax.spines['bottom'].set_color(grid_color)
    ax.spines['left'].set_color(grid_color)
    # Make major and minor tick lines gray, but labels stay black
    ax.tick_params(axis='both', which='both', color=grid_color, labelcolor='black')
    ax.set_axisbelow(True)

    # ------------------------------------------------------------------
    # Label every real vertex on the Pareto frontier
    # ------------------------------------------------------------------
    offset_pts = 6

    for (x, y), label in zip(pareto_front, pareto_names):
        if label is None:
            continue
        dx = offset_pts if max_X else -offset_pts
        dy = offset_pts if max_Y else -offset_pts
        ha = 'left' if max_X else 'right'
        va = 'bottom' if max_Y else 'top'
        txt = ax.annotate(
            label,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords='offset points',
            ha=ha,
            va=va,
            fontsize=9,
            color=label_to_color_dict[label],
            fontweight='bold',
        )
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

    # Restore original limits (prevents Matplotlib from auto-expanding them)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # --------------------------------------------------
    # Add unified two-block legend (color + marker + line)
    # --------------------------------------------------
    y_bottom = ax.get_ylim()[0]
    visible_hue_levels = list(plot_df.loc[plot_df[y_name] >= y_bottom, hue].dropna().unique())
    if hue_order is not None:
        ordered_visible = [h for h in hue_order if h in visible_hue_levels] + [h for h in visible_hue_levels if h not in hue_order]
    else:
        ordered_visible = visible_hue_levels

    color_handles = []
    color_labels = []
    for base_label in ordered_visible:
        color = palette_map.get(base_label, (0.33, 0.33, 0.33))
        handle = Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.0,
            markersize=7,
        )
        color_handles.append(handle)
        color_labels.append(str(base_label))

    marker_handles = []
    marker_labels = []
    if style_col is not None:
        if isinstance(markers_arg, dict):
            marker_map = markers_arg
        elif markers_arg is True:
            default_cycle = ['o', 'D', '^', 's', 'P', 'X', '*']
            marker_map = {lvl: default_cycle[i % len(default_cycle)] for i, lvl in enumerate(style_order or [])}
        else:
            marker_map = {}
        for lvl in (style_order or []):
            m = marker_map.get(lvl, 'o')
            h = Line2D(
                [0], [0],
                marker=m,
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.8,
                markersize=6,
            )
            marker_handles.append(h)
            marker_labels.append(str(lvl))

    frontier_proxy = Line2D([0], [0], linewidth=1.2, color='black', linestyle='--')
    marker_handles.append(frontier_proxy)
    marker_labels.append("Pareto Front")

    legend_fontsize = 9
    legend_in_plot_right = 0.98

    legend1 = g.fig.legend(
        color_handles, color_labels,
        loc="center left" if not legend_in_plot else ("lower right" if max_Y else "upper right"),#"lower right" if legend_in_plot else "center left",
        bbox_to_anchor=(0.79, 0.62) if not legend_in_plot else ((legend_in_plot_right, 0.085) if max_Y else (legend_in_plot_right, 0.977)),#(0.99, 0.06) if legend_in_plot else (0.79, 0.62),
        frameon=True,
        fontsize=legend_fontsize,
        ncol=1,
        labelspacing=0.25,
        handletextpad=0.5,
        borderpad=0.3,
        borderaxespad=0.3,
        columnspacing=0.6,
    )

    # Retrieve the bbox of the first legend (in figure coordinates)
    g.fig.canvas.draw()  # required so the legend layout is computed
    bbox1 = legend1.get_window_extent()
    bbox1_fig = bbox1.transformed(g.fig.transFigure.inverted())

    # Compute the left edge of legend1 in figure coords
    left_edge_legend1 = bbox1_fig.x0
    legend_in_plot_left = left_edge_legend1

    g.fig.legend(
        marker_handles, marker_labels,
        loc="center left" if not legend_in_plot else ("lower right" if max_Y else "upper right"),#"lower right" if legend_in_plot else "center left",
        bbox_to_anchor=(0.79, 0.26) if not legend_in_plot else ((legend_in_plot_left, 0.085) if max_Y else (legend_in_plot_left, 0.977)),#(0.85, 0.06) if legend_in_plot else (0.79, 0.26),
        frameon=True,
        fontsize=legend_fontsize,
        ncol=1,
        labelspacing=0.25,
        handletextpad=0.5,
        borderpad=0.3,
        borderaxespad=0.3,
        columnspacing=0.6,
    )

    if not legend_in_plot:
        g.fig.subplots_adjust(right=0.78)

    # Title + save/show
    # g.fig.suptitle(title, fontsize=14)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=600)
    if show:
        plt.show()
    plt.close()


def plot_optimal_arrow(
    ax,
    max_X: bool,
    max_Y: bool,
    size: float = 1,
    offset: float = 0.1,
    scale: float = 1.0,
):
    offset *= scale
    size *= scale

    ar_head_length = 0.42 * size
    ar_head_width = 0.30 * size
    ar_tail_width = 0.30 * size
    ar_text_size = 11 * size

    corner_x = 1 if max_X else 0
    corner_y = 1 if max_Y else 0
    start = (
        corner_x + (+offset if corner_x == 0 else -offset),
        corner_y + (+offset if corner_y == 0 else -offset),
    )
    end = (corner_x, corner_y)

    # --- Adjust start so arrow appears diagonal in display (pixel) space ---
    # Transform both points from axes fraction → display (pixel) coordinates
    start_disp = ax.transAxes.transform(start)
    end_disp = ax.transAxes.transform(end)

    dx_disp = abs(end_disp[0] - start_disp[0])
    dy_disp = abs(end_disp[1] - start_disp[1])

    if dx_disp != dy_disp and dx_disp > 0 and dy_disp > 0:
        # Choose the smaller of the two pixel distances as the base
        if dx_disp < dy_disp:
            # Adjust y offset proportionally so |dx| == |dy|
            scale = dx_disp / dy_disp
            start = (
                corner_x + (+offset if corner_x == 0 else -offset),
                corner_y + ((+offset if corner_y == 0 else -offset) * scale),
            )
        else:
            # Adjust x offset proportionally so |dx| == |dy|
            scale = dy_disp / dx_disp
            start = (
                corner_x + ((+offset if corner_x == 0 else -offset) * scale),
                corner_y + (+offset if corner_y == 0 else -offset),
            )

    # Create the arrow (in axes-fraction coordinates)
    arrow = ax.annotate(
        "",
        xy=end, xytext=start,
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle=f"Fancy,head_length={ar_head_length},head_width={ar_head_width},tail_width={ar_tail_width}",
            facecolor="forestgreen",
            edgecolor="forestgreen",
            linewidth=0,
            mutation_scale=100,
        ),
    )
    start_data = ax.transAxes.transform(start)
    end_data = ax.transAxes.transform(end)
    angle = np.degrees(np.arctan2(
        end_data[1] - start_data[1],
        end_data[0] - start_data[0]
    ))

    if angle < -90 or angle > 90:
        angle += 180
    mid = (np.array(start) + np.array(end)) / 2
    text = ax.text(
        mid[0], mid[1], "Optimal",
        transform=ax.transAxes,
        rotation=angle, rotation_mode="anchor",
        ha="center", va="center",
        fontsize=ar_text_size, fontweight="bold",
        color="white",
    )
    return arrow, text



def plot_pareto_aggregated(
    data: pd.DataFrame,
    x_name: str,
    y_name: str,
    data_x: pd.DataFrame = None,
    x_method: str = "median",
    y_method: str = "mean",
    max_X: bool = False,
    max_Y: bool = True,
    ylim=(None, 1),
    hue: str = "Method",
    title: str = None,
    save_path: str = None,
    show: bool = True,
    include_method_in_axis_name: bool = True,
    sort_y: bool = False,
):
    if data_x is None:
        data_x = data
    if x_name not in data_x:
        raise AssertionError(f"Missing x_name='{x_name}' column in data_x")
    elif not is_numeric_dtype(data_x[x_name]):
        raise AssertionError(f"x_name='{x_name}' must be a numeric dtype")
    elif data_x[x_name].isnull().values.any():
        raise AssertionError(f"x_name='{x_name}' cannot contain NaN values")
    if y_name not in data:
        raise AssertionError(f"Missing y_name='{y_name}' column in data")
    elif not is_numeric_dtype(data[y_name]):
        raise AssertionError(f"y_name='{y_name}' must be a numeric dtype")
    elif data[y_name].isnull().values.any():
        raise AssertionError(f"y_name='{y_name}' cannot contain NaN values")
    y_vals = aggregate_stats(df=data, on=y_name, method=[y_method])[y_method]
    x_vals = aggregate_stats(df=data_x, on=x_name, method=[x_method])[x_method]
    if include_method_in_axis_name:
        x_name = f'{x_name} ({x_method})'
        y_name = f'{y_name} ({y_method})'
    df_aggregated = y_vals.to_frame(name=y_name)
    df_aggregated[x_name] = x_vals
    df_aggregated[hue] = df_aggregated.index

    plot_pareto(
        data=df_aggregated,
        x_name=x_name,
        y_name=y_name,
        title=title,
        save_path=save_path,
        max_X=max_X,
        max_Y=max_Y,
        ylim=ylim,
        hue=hue,
        sort_y=sort_y,
        show=show,
    )
