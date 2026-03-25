from __future__ import annotations

import os

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
import matplotlib.patches as mpatches
import seaborn as sns

try:
    from adjustText import adjust_text

    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False


def aggregate_stats(df, on: str, groupby="method", method=["mean", "median", "std"]):
    return df[[groupby, on]].groupby(groupby).agg(method)[on]


def get_pareto_frontier(
    Xs,
    Ys,
    names=None,
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
    palette="tab20",
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
    legend_in_plot: bool = False,
    legend_maintain_plot_dims: bool = False,
    tick_fontsize: int = 12,
    label_fontsize: int = 10,
    axis_fontsize: int = 17,
    legend_fontsize: int = 9,
    legend_borderpad: float = 0.1,
    legend_handletextpad: float = 0,
    highlight_prefixes: dict[str, str] | None = None,
    top_left: bool = True,
    smart_pos: bool = True,
    pad_xy: bool = False,
    force_labels: list[str] | None = None,
    y_percent_format: bool = False,
    legend_first: list[str] | None = None,
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
        y_per_hue = plot_df.groupby(hue)[y_name].agg(agg_fun).sort_values(ascending=False)
        hue_order = list(y_per_hue.index)
    else:
        plot_df = data.copy()
        hue_order = None

    # ------------------------------
    # helper for "specified first, then rest"
    # ------------------------------
    def _apply_legend_first(
        levels: list[str],
        first: list[str] | None,
        *,
        place_first_at_end: bool = False,
    ) -> list[str]:
        if not first:
            return levels
        first_in = [h for h in first if h in levels]
        rest = [h for h in levels if h not in first_in]
        if place_first_at_end:
            first_in.reverse()
            return rest + first_in
        else:
            return first_in + rest

    # Build stable color mapping per hue category (here: method_type)
    hue_levels = list(pd.unique(plot_df[hue]))

    # If we computed a hue_order (sort_y=True), make sure its order respects legend_first too.
    if hue_order is not None:
        hue_order = _apply_legend_first(hue_order, legend_first, place_first_at_end=not max_Y)
    else:
        # Otherwise keep seaborn default ordering for plotting, but we *still* want the legend order
        # to respect legend_first later (we do that below on ordered_visible).
        pass

    if hue_levels is not None:
        hue_levels = _apply_legend_first(hue_levels, legend_first)

    # Ensure ≥20 visually distinct colors for many method types
    base_palette = sns.color_palette(palette, 20)
    if len(hue_levels) > 20:
        # Extend deterministically by combining tab20 + tab20b + tab20c if needed
        extended_palette = (
            sns.color_palette("tab20", 20) + sns.color_palette("tab20b", 20) + sns.color_palette("tab20c", 20)
        )
        colors = extended_palette[:len(hue_levels)]
    else:
        colors = base_palette[:len(hue_levels)]
    colors = [colors[i % len(colors)] for i in range(len(hue_levels))]
    palette_map = dict(zip(hue_levels, colors))

    label_to_hue_dict = data.set_index(label_col)[hue].to_dict()
    label_to_color_dict = {l: palette_map[h] for l, h in label_to_hue_dict.items()}

    # Build highlight prefix mapping and update colors for highlighted methods
    if highlight_prefixes is None:
        highlight_prefixes = {}

    # Create a mapping of labels to highlight colors
    label_to_highlight = {}
    for label in label_to_color_dict.keys():
        for prefix, color in highlight_prefixes.items():
            if label.startswith(prefix):
                label_to_highlight[label] = color
                break

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

    # Split data into highlighted and non-highlighted for separate plotting
    highlighted_labels = set(label_to_highlight.keys())
    plot_df_normal = plot_df[~plot_df[label_col].isin(highlighted_labels)]
    plot_df_highlighted = plot_df[plot_df[label_col].isin(highlighted_labels)]

    g = sns.relplot(
        x=x_name,
        y=y_name,
        data=plot_df_normal,
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

    # Plot highlighted points with their specified colors (larger size)
    if len(plot_df_highlighted) > 0:
        for _, row in plot_df_highlighted.iterrows():
            label = row[label_col]
            highlight_color = label_to_highlight.get(label, "black")
            # Get marker from style column if available
            if style_col is not None and isinstance(markers_arg, dict):
                marker = markers_arg.get(row[style_col], "o")
            else:
                marker = "o"
            ax.scatter(
                row[x_name],
                row[y_name],
                c=highlight_color,
                marker=marker,
                s=220,  # Larger size for highlighted points
                alpha=0.9,
                linewidth=1.5,
                edgecolor="white",
                zorder=3,
            )

    ax.set_xlabel(x_name, fontsize=axis_fontsize)
    ax.set_ylabel(y_name, fontsize=axis_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    # Compute Pareto frontier (use the plotted order)
    Xs = list(plot_df[x_name])
    Ys = list(plot_df[y_name])
    labels_for_front = list(plot_df[label_col])  # annotate with full method names
    pareto_front, pareto_names = get_pareto_frontier(Xs=Xs, Ys=Ys, names=labels_for_front, max_X=max_X, max_Y=max_Y)

    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]

    g.set(xscale="log")

    if ylim is not None:
        ax.set_ylim(ylim)

    # Draw Pareto frontier
    if pad_xy:
        # Get current axis limits and add padding for text labels
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # Add padding to keep labels inside (in log space for x)
        x_log_range = np.log10(x_max) - np.log10(x_min)
        y_range = y_max - y_min
        x_padding = 0.08 * x_log_range  # 8% padding
        y_padding = 0.06 * y_range  # 6% padding

        # Apply padding
        x_min_padded = 10 ** (np.log10(x_min) - x_padding)
        x_max_padded = 10 ** (np.log10(x_max) + x_padding)
        y_min_padded = y_min - y_padding
        y_max_padded = y_max + y_padding

        ax.set_xlim(x_min_padded, x_max_padded)
        ax.set_ylim(y_min_padded, y_max_padded)

    # Update limits for Pareto frontier calculation
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

    ax.plot(pf_X, pf_Y, linewidth=2 * fig_size_ratio, zorder=1, color="black", linestyle="--")

    ax.grid(True, zorder=-2)
    grid_color = ax.xaxis.get_gridlines()[0].get_color()
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_color(grid_color)
    ax.spines["right"].set_color(grid_color)
    ax.spines["bottom"].set_color(grid_color)
    ax.spines["left"].set_color(grid_color)
    # Make major and minor tick lines gray, but labels stay black
    ax.tick_params(axis="both", which="both", color=grid_color, labelcolor="black")
    ax.set_axisbelow(True)

    # ------------------------------------------------------------------
    # Label every real vertex on the Pareto frontier
    # Smart positioning: ABOVE symbol for top points, LEFT for others
    # ------------------------------------------------------------------
    texts = []
    text_positions = []  # Store (x, y) for adjustText

    # Compute Y range for positioning decisions
    y_range = y_max - y_min
    y_offset = y_range * 0.025  # 2.5% vertical offset for "above" placement

    # Get all real Pareto points (with labels) for analysis
    real_pareto_points = [(x, y, lbl) for (x, y), lbl in zip(pareto_front, pareto_names) if lbl is not None]
    pareto_labels_set = {lbl for _, _, lbl in real_pareto_points}

    # Build the list of points to label: Pareto frontier + force_labels (if not already on frontier)
    points_to_label = list(zip(pareto_front, pareto_names))
    if force_labels is not None:
        # Add forced labels that are not on the Pareto frontier
        for force_label in force_labels:
            if force_label not in pareto_labels_set:
                # Find the coordinates for this label in the data
                label_data = plot_df[plot_df[label_col] == force_label]
                if len(label_data) > 0:
                    x_val = label_data[x_name].values[0]
                    y_val = label_data[y_name].values[0]
                    points_to_label.append(((x_val, y_val), force_label))

    # Sort by Y to find top points
    sorted_by_y = sorted(real_pareto_points, key=lambda p: p[1], reverse=True)
    top_y_threshold = y_max - y_range * 0.25  # Top 25% of Y range

    for (x, y), label in points_to_label:
        if label is None:
            continue

        # Determine if this label should be highlighted
        highlight_color = label_to_highlight.get(label)

        # Use highlight color if specified, otherwise use the default color
        text_color = highlight_color if highlight_color else label_to_color_dict[label]

        # Create bbox for highlighted labels (black border box)
        if highlight_color:
            bbox_props = dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="black",
                linewidth=1.5,
                alpha=0.9,
            )
        else:
            bbox_props = None

        # Compute position relative to plot area
        x_log = np.log10(x)
        x_log_min, x_log_max = np.log10(x_min), np.log10(x_max)
        x_frac = (x_log - x_log_min) / (x_log_max - x_log_min) if x_log_max != x_log_min else 0.5
        y_frac = (y - y_min) / y_range if y_range != 0 else 0.5

        # Calculate offsets
        x_offset_log = (x_log_max - x_log_min) * 0.015  # 1.5% of x-axis range

        # Smart positioning logic:
        # 1. Specific label overrides (by name)
        # 2. Points in upper region (high Y): place ABOVE-LEFT or ABOVE-RIGHT based on top_left
        # 3. Points near left edge: place to the RIGHT
        # 4. Default: place to the LEFT

        is_top_region = y >= top_y_threshold

        is_fixed_placement = False  # Track if this label should skip adjustText

        if smart_pos:
            # Smart positioning logic:
            # 1. Specific label overrides (by name)
            # 2. Points near left edge: place to the RIGHT (check FIRST to avoid text going outside)
            # 3. Points in upper region (high Y): place ABOVE-LEFT or ABOVE-RIGHT based on top_left
            # 4. Default: place to the LEFT

            # Specific label overrides
            if x_frac < 0.2:
                # Near left edge - place to the RIGHT to avoid text going outside
                ha = "left"
                va = "bottom" if is_top_region else "center"
                text_x = 10 ** (x_log + x_offset_log)
                text_y = y + y_offset if is_top_region else y
                is_fixed_placement = is_top_region
            elif is_top_region:
                # Top region: place ABOVE and to the LEFT or RIGHT based on top_left
                if top_left:
                    ha = "right"
                    text_x = 10 ** (x_log - x_offset_log * 0.5)
                else:
                    ha = "left"
                    text_x = 10 ** (x_log + x_offset_log * 0.5)
                va = "bottom"
                text_y = y + y_offset
                is_fixed_placement = False
            else:
                # Default - place to the LEFT
                ha = "right"
                va = "center"
                text_x = 10 ** (x_log - x_offset_log)
                text_y = y
        else:
            # Non-smart positioning: all labels use adjustText
            # Default placement to the LEFT for all labels
            if x_frac < 0.15:
                # Near left edge - place to the RIGHT
                ha = "left"
                va = "center"
                text_x = 10 ** (x_log + x_offset_log)
                text_y = y
            else:
                # Default - place to the LEFT
                ha = "right"
                va = "center"
                text_x = 10 ** (x_log - x_offset_log)
                text_y = y
            # All labels will be added to adjustText
            is_fixed_placement = False

        # Create text at calculated position
        txt = ax.text(
            text_x,
            text_y,
            label,
            fontsize=label_fontsize,
            color=text_color,
            fontweight="extra bold" if highlight_color else "bold",
            bbox=bbox_props,
            ha=ha,
            va=va,
        )

        # Add path effects for text styling
        if highlight_color:
            # Highlighted labels: shadow effect for artistic emphasis
            txt.set_path_effects(
                [
                    PathEffects.withSimplePatchShadow(
                        offset=(1.5, -1.5),
                        shadow_rgbFace="gray",
                        alpha=0.5,
                    ),
                    PathEffects.withStroke(linewidth=2, foreground="white"),
                ]
            )
        else:
            # Non-highlighted labels: white stroke for readability
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="white")])

        # Only add non-fixed texts to adjustText (keep top labels fixed)
        if not is_fixed_placement:
            texts.append(txt)
            text_positions.append((x, y))

    # Use adjustText for collision avoidance (only for non-top-region labels)
    if HAS_ADJUST_TEXT and texts:
        adjust_text(
            texts,
            x=[p[0] for p in text_positions],
            y=[p[1] for p in text_positions],
            ax=ax,
            force_text=(0.3, 0.3),
            force_points=(0.8, 0.8),  # Strong repulsion from points
            expand_text=(1.1, 1.1),
            expand_points=(1.8, 1.8),  # Large buffer around points
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5),
            only_move={"points": "y", "text": "xy"},
            lim=50,
        )

    # Restore original limits (prevents Matplotlib from auto-expanding them)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # --------------------------------------------------
    # Add unified two-block legend (color + marker + line)
    # --------------------------------------------------
    y_bottom = ax.get_ylim()[0]
    visible_hue_levels = list(plot_df.loc[plot_df[y_name] >= y_bottom, hue].dropna().unique())
    if hue_order is not None:
        ordered_visible = [h for h in hue_order if h in visible_hue_levels] + [
            h for h in visible_hue_levels if h not in hue_order
        ]
    else:
        ordered_visible = visible_hue_levels

    color_handles = []
    color_labels = []
    for base_label in ordered_visible:
        # Check if any method with this hue is highlighted
        # Find labels that have this hue and check if highlighted
        is_highlighted = False
        highlight_color = None
        for lbl, hue_val in label_to_hue_dict.items():
            if hue_val == base_label and lbl in label_to_highlight:
                is_highlighted = True
                highlight_color = label_to_highlight[lbl]
                break

        if is_highlighted and highlight_color:
            color = highlight_color
            marker_size = 9  # Larger for highlighted
            edge_width = 1.0
        else:
            color = palette_map.get(base_label, (0.33, 0.33, 0.33))
            marker_size = 7
            edge_width = 0.0

        handle = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=edge_width,
            markersize=marker_size,
        )
        color_handles.append(handle)
        color_labels.append(str(base_label))

    marker_handles = []
    marker_labels = []
    if style_col is not None:
        if isinstance(markers_arg, dict):
            marker_map = markers_arg
        elif markers_arg is True:
            default_cycle = ["o", "D", "^", "s", "P", "X", "*"]
            marker_map = {lvl: default_cycle[i % len(default_cycle)] for i, lvl in enumerate(style_order or [])}
        else:
            marker_map = {}
        for lvl in style_order or []:
            m = marker_map.get(lvl, "o")
            h = Line2D(
                [0],
                [0],
                marker=m,
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.8,
                markersize=6,
            )
            marker_handles.append(h)
            marker_labels.append(str(lvl))

    frontier_proxy = Line2D([0], [0], linewidth=1.2, color="black", linestyle="--")
    marker_handles.append(frontier_proxy)
    marker_labels.append("Pareto Front")

    # Legend positioning: outside plot (upper right corner, tightly attached) or inside
    if legend_in_plot:
        # Inside plot positioning (original behavior)
        legend_in_plot_right = 0.98
        legend1_loc = "lower right" if max_Y else "upper right"
        legend1_bbox = (legend_in_plot_right, 0.085) if max_Y else (legend_in_plot_right, 0.977)
        legend1 = g.fig.legend(
            color_handles,
            color_labels,
            loc=legend1_loc,
            bbox_to_anchor=legend1_bbox,
            frameon=True,
            fontsize=legend_fontsize,
            ncol=1,
            labelspacing=0.15,
            handletextpad=0.3,
            borderpad=0.2,
            borderaxespad=0.05,
            columnspacing=0.4,
        )
        # Retrieve the bbox of the first legend (in figure coordinates)
        g.fig.canvas.draw()
        bbox1 = legend1.get_window_extent()
        bbox1_fig = bbox1.transformed(g.fig.transFigure.inverted())
        left_edge_legend1 = bbox1_fig.x0
        legend2_loc = "lower right" if max_Y else "upper right"
        legend2_bbox = (left_edge_legend1, 0.085) if max_Y else (left_edge_legend1, 0.977)
        g.fig.legend(
            marker_handles,
            marker_labels,
            loc=legend2_loc,
            bbox_to_anchor=legend2_bbox,
            frameon=True,
            fontsize=legend_fontsize,
            ncol=1,
            labelspacing=0.15,
            handletextpad=0.3,
            borderpad=0.2,
            borderaxespad=0.05,
            columnspacing=0.4,
        )
    else:
        # Outside plot: position legends using axes transform
        # Add small gap (0.01) between plot and legend
        # First legend: method colors (on top) - this is typically the wider one
        legend1 = ax.legend(
            color_handles,
            color_labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),  # Small gap from axes right edge
            frameon=True,
            fontsize=legend_fontsize,
            ncol=1,
            labelspacing=0.15,
            handletextpad=legend_handletextpad,
            borderpad=legend_borderpad,
            borderaxespad=0.0,
            columnspacing=0.4,
        )

        # Need to draw to get legend bbox
        g.fig.canvas.draw()
        bbox1 = legend1.get_window_extent()
        bbox1_axes = bbox1.transformed(ax.transAxes.inverted())

        # Second legend: markers (Default, Tuned, etc.) - INSIDE plot at lower right
        legend2 = ax.legend(
            marker_handles,
            marker_labels,
            loc="best",
            frameon=True,
            fontsize=legend_fontsize,
            ncol=1,
            labelspacing=0.15,
            handletextpad=0.3,
            borderpad=0.3,
            borderaxespad=0.5,
            columnspacing=0.4,
        )

        # Add first legend back (since ax.legend replaces previous legend)
        ax.add_artist(legend1)

        # IMPORTANT: bbox_extra_artists ensures the outside legend is included when saving
        g._legend_extra_artists = [legend1]

    # Format y-axis ticks with percentage symbol if requested (right before save)
    if y_percent_format:
        # Get current tick locations within visible range and set labels with %
        y_lim = ax.get_ylim()
        yticks = [t for t in ax.get_yticks() if y_lim[0] <= t <= y_lim[1]]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{int(y)}\\%" for y in yticks])

    # Title + save/show
    # g.fig.suptitle(title, fontsize=14)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Include extra artists (legends) in bounding box calculation
        extra_artists = getattr(g, "_legend_extra_artists", None)
        if extra_artists:
            if legend_maintain_plot_dims:
                # Adjust figure to make room for legend on the right
                g.fig.subplots_adjust(right=0.8)
            plt.savefig(save_path, bbox_inches="tight", bbox_extra_artists=extra_artists, dpi=600)
        else:
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
        xy=end,
        xytext=start,
        xycoords="axes fraction",
        textcoords="axes fraction",
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
    angle = np.degrees(np.arctan2(end_data[1] - start_data[1], end_data[0] - start_data[0]))

    if angle < -90 or angle > 90:
        angle += 180
    mid = (np.array(start) + np.array(end)) / 2
    text = ax.text(
        mid[0],
        mid[1],
        "Optimal",
        transform=ax.transAxes,
        rotation=angle,
        rotation_mode="anchor",
        ha="center",
        va="center",
        fontsize=ar_text_size,
        fontweight="bold",
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
        x_name = f"{x_name} ({x_method})"
        y_name = f"{y_name} ({y_method})"
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
