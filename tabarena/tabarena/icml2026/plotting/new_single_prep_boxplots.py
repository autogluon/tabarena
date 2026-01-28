import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_normalized_df(
    df: pd.DataFrame,
    *,
    dataset_col: str,
    lower_bound_col: str,
    upper_bound_col: str,
    cols_to_normalize: list[str],
    suffix: str = "_norm",
    drop_invalid_bounds: bool = True,
) -> pd.DataFrame:
    """
    Add normalized columns for cols_to_normalize using per-row bounds:
        x_norm = (x - lower) / (upper - lower)

    If upper==lower, normalization is undefined.
      - drop_invalid_bounds=True -> drop those rows
      - else -> produce NaNs for normalized values
    """
    out = df.copy()
    denom = out[upper_bound_col] - out[lower_bound_col]

    invalid = denom == 0
    if invalid.any():
        if drop_invalid_bounds:
            out = out.loc[~invalid].copy()
            denom = denom.loc[~invalid]
        else:
            denom = denom.replace(0, pd.NA)

    denom = denom.replace(0, pd.NA)

    for c in cols_to_normalize:
        out[f"{c}{suffix}"] = (out[c] - out[lower_bound_col]) / denom

    return out


def make_comparison_plot_df(
    df: pd.DataFrame,
    *,
    dataset_col: str,
    lower_bound_col: str,
    upper_bound_col: str,
    reference_col: str,
    method_cols: list[str],
    normalize_suffix: str = "_norm",
    drop_invalid_bounds: bool = True,
    dropna: bool = True,
    # optional clipping
    clip_norm: tuple[float, float] | None = None,
    clip_delta: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end data prep:
      1) normalize (reference + methods) between lower/upper bounds
      2) compute delta vs reference on normalized scale
      3) return plot_df in long format with columns:
         [dataset_col, 'method', 'improvement']

    improvement here is:
        norm(method) - norm(reference)
    """
    cols_to_norm = [reference_col] + list(method_cols)
    df_norm = make_normalized_df(
        df,
        dataset_col=dataset_col,
        lower_bound_col=lower_bound_col,
        upper_bound_col=upper_bound_col,
        cols_to_normalize=cols_to_norm,
        suffix=normalize_suffix,
        drop_invalid_bounds=drop_invalid_bounds,
    )

    ref_norm_col = f"{reference_col}{normalize_suffix}"

    # Optional clip in normalized space (often you may NOT want this)
    if clip_norm is not None:
        lo, hi = clip_norm
        for c in cols_to_norm:
            df_norm[f"{c}{normalize_suffix}"] = df_norm[f"{c}{normalize_suffix}"].clip(lo, hi)

    # Build long delta dataframe
    rows = []
    ref_vals = df_norm[ref_norm_col]

    for m in method_cols:
        m_norm_col = f"{m}{normalize_suffix}"
        delta = df_norm[m_norm_col] - ref_vals
        rows.append(
            pd.DataFrame(
                {
                    dataset_col: df_norm[dataset_col].values,
                    "method": m,
                    "improvement": delta.values,
                }
            )
        )

    plot_df = pd.concat(rows, ignore_index=True)

    if clip_delta is not None:
        lo, hi = clip_delta
        plot_df["improvement"] = plot_df["improvement"].clip(lo, hi)

    if dropna:
        plot_df = plot_df.dropna(subset=["method", "improvement", dataset_col])

    return df_norm, plot_df


# --- Your plotting function (kept, but I'd recommend optionally adding a line at 0 and maybe also at 1 if desired) ---
def boxplot_plotdf_pubready(
    plot_df: pd.DataFrame,
    *,
    method_col: str = "method",
    value_col: str = "improvement",
    dataset_col: str | None = None,
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
    if title_size is None:
        title_size = font_size
    if tick_size is None:
        tick_size = font_size

    df = plot_df.copy()

    if dropna:
        df = df.dropna(subset=[method_col, value_col])

    if cap is not None:
        if isinstance(cap, tuple):
            lo, hi = cap
        else:
            lo, hi = -float(cap), float(cap)
        df[value_col] = df[value_col].clip(lower=lo, upper=hi)

    if methods is None:
        methods = list(pd.unique(df[method_col]))
        if len(methods) == 0:
            methods = sorted(df[method_col].dropna().unique().tolist())

    if labels is None:
        labels = methods
    if len(labels) != len(methods):
        raise ValueError("labels must have same length as methods.")

    data_per_method = []
    for m in methods:
        vals = df.loc[df[method_col] == m, value_col].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        data_per_method.append(vals)

    if dataset_col is not None and dataset_col in df.columns:
        scores_wide = (
            df.pivot_table(index=dataset_col, columns=method_col, values=value_col, aggfunc="first")
            .reindex(columns=methods)
        )
    else:
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

        ax.boxplot(
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

        rng = np.random.default_rng(0)
        for i, vals in enumerate(data_per_method, start=1):
            if vals.size == 0:
                continue
            if horizontal:
                y = i + rng.uniform(-jitter, jitter, size=vals.size)
                x = vals
                ax.scatter(x, y, s=point_size, alpha=point_alpha, linewidths=0)
            else:
                x = i + rng.uniform(-jitter, jitter, size=vals.size)
                y = vals
                ax.scatter(x, y, s=point_size, alpha=point_alpha, linewidths=0)

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

        if add_zero_line:
            if horizontal:
                ax.axvline(0.0, linestyle=zero_line_style, linewidth=zero_line_width)
            else:
                ax.axhline(0.0, linestyle=zero_line_style, linewidth=zero_line_width)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(spine_linewidth)
        ax.spines["bottom"].set_linewidth(spine_linewidth)
        ax.tick_params(axis="both", width=spine_linewidth, length=3)

        if horizontal:
            lo, hi = ax.get_xlim()
            pad_amt = pad * (hi - lo) if hi > lo else 0.1
            ax.set_xlim(lo - pad_amt, hi + pad_amt)
        else:
            lo, hi = ax.get_ylim()
            pad_amt = pad * (hi - lo) if hi > lo else 0.1
            ax.set_ylim(lo - pad_amt, hi + pad_amt)
        ax.invert_xaxis()
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


def compare_methods_via_boxplots(
    df: pd.DataFrame,
    *,
    dataset_col: str,
    lower_bound_col: str,
    upper_bound_col: str,
    reference_col: str,
    method_cols: list[str],
    methods: list[str] | None = None,
    labels: list[str] | None = None,
    title: str | None = None,
    **plot_kwargs,
):
    """
    One-call convenience wrapper:
      df -> normalized -> delta vs reference -> pub-ready boxplot.
    """
    df_norm, plot_df = make_comparison_plot_df(
        df,
        dataset_col=dataset_col,
        lower_bound_col=lower_bound_col,
        upper_bound_col=upper_bound_col,
        reference_col=reference_col,
        method_cols=method_cols,
    )

    fig, ax, scores_wide = boxplot_plotdf_pubready(
        plot_df,
        dataset_col=dataset_col,
        methods=methods or method_cols,
        labels=labels,
        title=title or f"Δ normalized vs {reference_col}",
        **plot_kwargs,
    )

    return fig, ax, df_norm, plot_df, scores_wide