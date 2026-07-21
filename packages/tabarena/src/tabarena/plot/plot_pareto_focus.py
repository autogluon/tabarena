"""Focus-style Pareto-front scatter used by the website figures.

Design: color encodes the *model family* (few hues) instead of one hue per
method; only Pareto-front members plus an explicit ``focus_methods`` list are
emphasized (family color, direct label, full opacity) while every other method
recedes into a grey field. Thin connectors tie a method's variants (default /
tuned / tuned + ensembled) together, and the legend collapses to a single strip
above the axes.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

from tabarena.plot.plot_pareto_frontier import get_pareto_frontier, plot_optimal_arrow

if TYPE_CHECKING:
    import pandas as pd

#: Family display name (see ``tabarena.website.website_format.Constants``) -> color.
#: Palette validated for color-vision deficiency; family identity is always backed
#: by a direct text label on emphasized methods, never color alone.
FAMILY_COLORS: dict[str, str] = {
    "Foundation Model": "#2a78d6",  # blue
    "Tree-based": "#eb6834",  # orange
    "Neural Network": "#1baf7a",  # aqua
    "Other": "#4a3aa7",  # violet
    "Reference Pipeline": "#e87ba4",  # magenta
    "Baseline": "#898781",  # grey
}

#: Grey used for non-emphasized ("field") methods.
MUTED_COLOR = "#b9b8b1"

#: Variant ("Type") -> matplotlib marker; mirrors ``LeaderboardReporter.style_markers``.
DEFAULT_VARIANT_MARKERS: dict[str, str] = {
    "Default": "o",
    "Tuned": "s",
    "Tuned + Ens.": "X",
    "Baseline": "D",
    "Best": "*",
    "Default, Holdout": "^",
    "Tuned, Holdout": "<",
    "Tuned + Ens., Holdout": ">",
}


def compute_front_methods(
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    method_col: str,
    max_X: bool = False,
    max_Y: bool = False,
) -> tuple[list[tuple[float, float]], set[str]]:
    """Pareto-front vertices over all points + the set of methods that define the front."""
    front, names = get_pareto_frontier(
        Xs=list(data[x_col]),
        Ys=list(data[y_col]),
        names=list(data[method_col]),
        max_X=max_X,
        max_Y=max_Y,
        include_boundary_edges=False,
    )
    return front, {n for n in names if n is not None}


def plot_pareto_focus(
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    method_col: str = "Method",
    variant_col: str = "Type",
    family_col: str = "Family",
    max_X: bool = False,
    max_Y: bool = False,
    xlog: bool = True,
    focus_methods: list[str] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    add_optimal_arrow: bool = True,
    variant_markers: dict[str, str] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    figsize: tuple[float, float] = (10.5, 6.2),
):
    """Render the focus-style Pareto scatter.

    Parameters
    ----------
    data
        One row per (method, variant) point. Must contain ``x_col``, ``y_col``,
        ``method_col`` (method display name, e.g. "CatBoost"), ``variant_col``
        (e.g. "Default"/"Tuned"/"Tuned + Ens.") and ``family_col`` (a key of
        :data:`FAMILY_COLORS`).
    focus_methods
        Methods to emphasize *in addition to* the Pareto-front members. Focused
        methods get a boxed label; front members a plain colored label.
    max_X, max_Y
        Direction of "better" per axis (both False = lower-left is optimal).
    """
    import matplotlib.patheffects as PathEffects
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    try:
        from adjustText import adjust_text

        has_adjust_text = True
    except ImportError:
        has_adjust_text = False

    if variant_markers is None:
        variant_markers = DEFAULT_VARIANT_MARKERS
    if focus_methods is None:
        focus_methods = []

    data = data.dropna(subset=[x_col, y_col]).copy()
    variant_rank = {v: i for i, v in enumerate(variant_markers)}
    data["_variant_rank"] = data[variant_col].map(variant_rank).fillna(len(variant_rank))

    front, front_methods = compute_front_methods(
        data, x_col=x_col, y_col=y_col, method_col=method_col, max_X=max_X, max_Y=max_Y
    )
    emphasized = front_methods | {m for m in focus_methods if m in set(data[method_col])}

    fig, ax = plt.subplots(figsize=figsize)
    if xlog:
        ax.set_xscale("log")

    method_family = data.groupby(method_col)[family_col].first().to_dict()

    def _color(method: str) -> str:
        if method in emphasized:
            return FAMILY_COLORS.get(method_family.get(method), FAMILY_COLORS["Other"])
        return MUTED_COLOR

    # Connectors linking a method's variants (default -> tuned -> tuned + ensembled).
    for method, g in data.groupby(method_col):
        g = g.sort_values("_variant_rank")
        if len(g) < 2:
            continue
        emph = method in emphasized
        ax.plot(
            g[x_col],
            g[y_col],
            "-",
            color=_color(method),
            lw=1.4 if emph else 0.9,
            alpha=0.65 if emph else 0.45,
            zorder=2 if emph else 1,
        )

    # Points.
    for _, p in data.iterrows():
        method = p[method_col]
        emph = method in emphasized
        ax.scatter(
            p[x_col],
            p[y_col],
            marker=variant_markers.get(p[variant_col], "o"),
            s=130 if emph else 70,
            c=_color(method),
            alpha=0.95 if emph else 0.55,
            edgecolor="#0b0b0b" if emph else "white",
            linewidth=0.8 if emph else 0.4,
            zorder=4 if emph else 3,
        )

    if ylim is not None:
        ax.set_ylim(ylim)

    # Pareto-front staircase, extended to the axis edges. ``get_pareto_frontier``
    # orders vertices starting at the better-X end, so the vertical extension
    # (toward the worst Y) attaches to the first vertex and the horizontal
    # extension (toward the worst X) to the last one.
    ax.autoscale_view()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    y_worst_edge = y_min if max_Y else y_max
    x_worst_edge = x_min if max_X else x_max
    fx = [front[0][0], *[v[0] for v in front], x_worst_edge]
    fy = [y_worst_edge, *[v[1] for v in front], front[-1][1]]
    ax.plot(fx, fy, "--", color="#0b0b0b", lw=1.5, zorder=2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if add_optimal_arrow:
        plot_optimal_arrow(ax=ax, max_X=max_X, max_Y=max_Y, size=0.45, scale=1.2)

    # Direct labels: one per emphasized method, anchored at its best-Y point.
    texts = []
    for method in sorted(emphasized):
        g = data[data[method_col] == method]
        if g.empty:
            continue
        p = g.loc[g[y_col].idxmax() if max_Y else g[y_col].idxmin()]
        is_focus = method in focus_methods
        txt = ax.text(
            p[x_col],
            p[y_col],
            method,
            fontsize=10.5 if is_focus else 9.5,
            color=FAMILY_COLORS.get(method_family.get(method), FAMILY_COLORS["Other"]),
            fontweight="bold",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#0b0b0b", linewidth=1.2, alpha=0.9)
            if is_focus
            else None,
            zorder=6,
        )
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="white")])
        texts.append(txt)
    if has_adjust_text and texts:
        adjust_text(
            texts,
            x=data[x_col].tolist(),
            y=data[y_col].tolist(),
            ax=ax,
            force_text=(0.35, 0.5),
            force_static=(0.6, 0.8),
            expand=(1.15, 1.3),
            arrowprops=dict(arrowstyle="-", color="#898781", lw=0.6, alpha=0.7),
            iter_lim=120,
        )

    # Single legend strip above the axes: family swatches, variant markers,
    # front line, grey "other methods" swatch.
    families_present = [f for f in FAMILY_COLORS if f in {method_family[m] for m in emphasized}]
    variants_present = [v for v in variant_markers if v in set(data[variant_col])]
    handles = [Patch(facecolor=FAMILY_COLORS[f], edgecolor="none", label=f) for f in families_present]
    handles += [
        Line2D(
            [0],
            [0],
            marker=variant_markers[v],
            ls="None",
            markerfacecolor="white",
            markeredgecolor="#0b0b0b",
            markersize=8,
            label=v,
        )
        for v in variants_present
    ]
    handles.append(Line2D([0], [0], ls="--", color="#0b0b0b", lw=1.5, label="Pareto front"))
    if len(emphasized) < data[method_col].nunique():
        handles.append(Patch(facecolor=MUTED_COLOR, edgecolor="none", label="Other methods"))
    legend = ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=4,
        frameon=False,
        fontsize=10,
        handletextpad=0.5,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    ax.set_xlabel(x_label if x_label is not None else x_col, fontsize=13)
    ax.set_ylabel(y_label if y_label is not None else y_col, fontsize=13)
    ax.grid(True, color="#e1e0d9", lw=0.7)
    ax.tick_params(labelsize=10, color="#c3c2b7")
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    if title is not None:
        # The legend strip sits just above the axes; pad the title past it
        # (one legend row ≈ 22pt at this font size).
        legend_rows = math.ceil(len(handles) / 4)
        ax.set_title(title, fontsize=15, pad=14 + 22 * legend_rows)

    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight", bbox_extra_artists=[legend])
    if show:
        plt.show()
    plt.close(fig)
