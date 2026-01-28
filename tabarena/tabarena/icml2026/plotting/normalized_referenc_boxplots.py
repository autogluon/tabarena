import pandas as pd

def normalize_against_references(
    df: pd.DataFrame,
    dataset_col: str,
    lower_ref_col: str,
    upper_ref_col: str,
    method_cols: list[str],
    suffix: str = "_norm",
):
    """
    Normalize method scores between lower and upper reference columns
    and prepare data for boxplots of improvement over the lower reference.

    Normalization:
        (method - lower_ref) / (upper_ref - lower_ref)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    dataset_col : str
        Column identifying datasets
    lower_ref_col : str
        Lower reference column name
    upper_ref_col : str
        Upper reference column name
    method_cols : list[str]
        Columns to normalize
    suffix : str
        Suffix for normalized columns

    Returns
    -------
    df_out : pd.DataFrame
        Original dataframe with added normalized columns
    plot_df : pd.DataFrame
        Long-format dataframe suitable for boxplots
    """

    df_out = df.copy()

    denom = df_out[upper_ref_col] - df_out[lower_ref_col]
    denom = denom.replace(0, pd.NA)

    # Normalize
    for col in method_cols:
        df_out[f"{col}{suffix}"] = (df_out[col] - df_out[lower_ref_col]) / denom

    # Prepare long format for plotting
    norm_cols = [f"{c}{suffix}" for c in method_cols]

    plot_df = df_out.melt(
        id_vars=dataset_col,
        value_vars=norm_cols,
        var_name="method",
        value_name="improvement",
    )

    plot_df["method"] = plot_df["method"].str.replace(suffix, "", regex=False)

    return df_out, plot_df



# df_norm, plot_df = normalize_against_references(
#     df=df_components,
#     dataset_col="dataset",
#     lower_ref_col="BaseLGB",
#     upper_ref_col="TunedEnsembleLGB",
#     method_cols=["+Arithmetic", "+OOF-TE", "+CombineThenTE", "+GroupBy", "+RSFC", "TunedPrepLightGBM"][::-1],
# )
# boxplot_plotdf_pubready(plot_df, cap=(-2,5))

########################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def boxplot_plotdf_pubready(
    plot_df: pd.DataFrame,
    *,
    method_col: str = "method",
    value_col: str = "improvement",
    dataset_col: str | None = None,  # optional (only used to pivot to wide if provided)
    methods: list[str] | None = None,
    labels: list[str] | None = None,
    cap: float | tuple[float, float] | None = None,
    title: str | None = None,
    horizontal: bool = True,
    figsize: tuple[float, float] = (3.4, 2.6),
    jitter: float = 0.11,
    point_size: float = 12.0,
    point_alpha: float = 0.75,
    # publication styling
    font_size: float = 8.0,
    title_size: float | None = None,
    tick_size: float | None = None,
    spine_linewidth: float = 0.8,
    box_linewidth: float = 0.9,
    pad: float = 0.02,
    # reference lines
    add_zero_line: bool = True,
    zero_line_style: str = "--",
    zero_line_width: float = 0.8,
    # saving
    save_path: str | None = None,
    dpi: int = 300,
    transparent: bool = True,
    show: bool = True,
    dropna: bool = True,
):
    """
    Publication-ready, single-panel boxplot+points from a long-format plot_df.

    Expected input:
      - plot_df[method_col] : method name (categorical)
      - plot_df[value_col]  : improvement values (e.g., normalized improvement vs lower ref)

    Notes:
      - No score computation inside; this ONLY plots what you give it.
      - If cap is provided, values are clipped before plotting.
      - Returns (fig, ax, scores_wide) where scores_wide is a wide table
        (rows = datasets if dataset_col provided else row index) and columns = methods.
    """
    if title_size is None:
        title_size = font_size
    if tick_size is None:
        tick_size = font_size

    df = plot_df.copy()

    # Drop NaNs
    if dropna:
        df = df.dropna(subset=[method_col, value_col])

    # Cap/clip values
    if cap is not None:
        if isinstance(cap, tuple):
            lo, hi = cap
        else:
            lo, hi = -float(cap), float(cap)
        df[value_col] = df[value_col].clip(lower=lo, upper=hi)

    # Decide method order
    if methods is None:
        # stable-ish: order by appearance, fallback to sorted uniques
        methods = list(pd.unique(df[method_col]))
        if len(methods) == 0:
            methods = sorted(df[method_col].dropna().unique().tolist())

    # Labels
    if labels is None:
        labels = methods
    if len(labels) != len(methods):
        raise ValueError("labels must have same length as methods.")

    # Build per-method arrays for boxplot and points
    data_per_method = []
    for m in methods:
        vals = df.loc[df[method_col] == m, value_col].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        data_per_method.append(vals)

    # Also return a "scores" wide df for convenience
    if dataset_col is not None and dataset_col in df.columns:
        scores_wide = (
            df.pivot_table(index=dataset_col, columns=method_col, values=value_col, aggfunc="first")
            .reindex(columns=methods)
        )
    else:
        # no dataset id -> wide table by original row index (not as meaningful, but consistent)
        tmp = df[[method_col, value_col]].copy()
        tmp["_row"] = np.arange(len(tmp))
        scores_wide = tmp.pivot(index="_row", columns=method_col, values=value_col).reindex(columns=methods)

    axis_label = value_col.replace("_", " ").strip().title()

    with plt.rc_context(
        {
            "font.size": font_size,
            "axes.titlesize": title_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "axes.linewidth": spine_linewidth,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Boxplot (matplotlib expects list of arrays)
        bp = ax.boxplot(
            data_per_method,
            vert=not horizontal,
            widths=0.6,
            showfliers=False,
            patch_artist=False,
            boxprops=dict(linewidth=box_linewidth),
            whiskerprops=dict(linewidth=box_linewidth),
            capprops=dict(linewidth=box_linewidth),
            medianprops=dict(linewidth=box_linewidth),
        )

        # Jittered points
        rng = np.random.default_rng(0)
        for i, vals in enumerate(data_per_method, start=1):
            if vals.size == 0:
                continue
            # positions are 1..N in matplotlib boxplot
            if horizontal:
                y = i + rng.uniform(-jitter, jitter, size=vals.size)
                x = vals
                ax.scatter(x, y, s=point_size, alpha=point_alpha, linewidths=0)
            else:
                x = i + rng.uniform(-jitter, jitter, size=vals.size)
                y = vals
                ax.scatter(x, y, s=point_size, alpha=point_alpha, linewidths=0)

        # Axis labels / ticks
        if horizontal:
            ax.set_yticks(range(1, len(methods) + 1))
            ax.set_yticklabels(labels)
            ax.set_xlabel(axis_label)
        else:
            ax.set_xticks(range(1, len(methods) + 1))
            ax.set_xticklabels(labels, rotation=0)
            ax.set_ylabel(axis_label)

        if title:
            ax.set_title(title)

        # Optional zero line (improvement = 0)
        if add_zero_line:
            if horizontal:
                ax.axvline(0.0, linestyle=zero_line_style, linewidth=zero_line_width)
            else:
                ax.axhline(0.0, linestyle=zero_line_style, linewidth=zero_line_width)

        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(spine_linewidth)
        ax.spines["bottom"].set_linewidth(spine_linewidth)
        ax.tick_params(axis="both", width=spine_linewidth, length=3)

        # Pad limits
        if horizontal:
            lo, hi = ax.get_xlim()
            pad_amt = pad * (hi - lo) if hi > lo else 0.1
            ax.set_xlim(lo - pad_amt, hi + pad_amt)
        else:
            lo, hi = ax.get_ylim()
            pad_amt = pad * (hi - lo) if hi > lo else 0.1
            ax.set_ylim(lo - pad_amt, hi + pad_amt)

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(
                save_path,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.01,
                transparent=transparent,
            )

        if show:
            plt.show()

    return fig, ax, scores_wide


# fig, ax, df_norm, plot_df, scores_wide = compare_methods_boxplot(
#     df=df_components,
#     dataset_col="dataset",
#     lower_ref_col="BaseLGB",
#     upper_ref_col="BestOnTabArena",
#     method_cols=["+Arithmetic", "+OOF-TE", "+CombineThenTE", "+GroupBy", "+RSFC", "TunedPrepLightGBM"][::-1],
#     title="Normalized improvement vs lower reference",
#     horizontal=True,
#     clip_norm=None,          # or (0, 1) if you want to clamp
#     cap=(-1.5,2),                # or cap=3 to clip extreme outliers in the plot only
#     dpi=300,
#     figsize=(8, 4),
# )

