import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _compute_scores_generic(
    df: pd.DataFrame,
    baseline_col: str,
    competitor_cols: list[str],
    lower_is_better: bool,
    mode: str,
    eps: float,
    cap: float | tuple[float, float] | None,
) -> tuple[pd.DataFrame, str]:
    for c in [baseline_col, *competitor_cols]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in dataframe.")

    d = df[[baseline_col, *competitor_cols]].dropna().copy()
    base = d[baseline_col].to_numpy(dtype=float)

    scores = {}
    if mode == "log_ratio":
        base_safe = np.maximum(base, eps)
        for c in competitor_cols:
            m = np.maximum(d[c].to_numpy(dtype=float), eps)
            lr = np.log(m / base_safe)
            scores[c] = (-lr) if lower_is_better else (lr)
        axis_label = f"Improvement vs {baseline_col}"
    elif mode == "relative":
        denom = np.maximum(np.abs(base), eps)
        for c in competitor_cols:
            m = d[c].to_numpy(dtype=float)
            scores[c] = ((base - m) / denom) if lower_is_better else ((m - base) / denom)
        axis_label = f"Relative improvement vs {baseline_col}"
    else:
        raise ValueError("mode must be 'log_ratio' or 'relative'")

    score_df = pd.DataFrame(scores)[competitor_cols]

    if cap is not None:
        if isinstance(cap, (int, float)):
            low, high = -float(cap), float(cap)
        else:
            low, high = cap
        score_df = score_df.clip(lower=low, upper=high)

    return score_df, axis_label


def _boxplot_with_points_on_ax(
    ax: plt.Axes,
    score_df: pd.DataFrame,
    labels: list[str],
    axis_label: str,
    panel_title: str | None,
    jitter: float,
    point_size: float,
    point_alpha: float,
    horizontal: bool,
    box_linewidth: float,
):
    data = [score_df[col].to_numpy() for col in score_df.columns]

    common_props = dict(linewidth=box_linewidth)
    boxprops = dict(**common_props)
    whiskerprops = dict(**common_props)
    capprops = dict(**common_props)
    medianprops = dict(**common_props)
    # no explicit colors (good for grayscale & journal styles)

    if horizontal:
        ax.boxplot(
            data,
            labels=labels,
            showfliers=False,
            widths=0.6,
            vert=False,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops,
        )
        rng = np.random.default_rng(0)
        for i, col in enumerate(score_df.columns, start=1):
            x = score_df[col].to_numpy()
            y = i + rng.uniform(-jitter, jitter, size=len(x))
            ax.scatter(x, y, s=point_size, alpha=point_alpha, linewidths=0, zorder=3)

        ax.axvline(0.0, color="black", linewidth=box_linewidth)
        ax.set_xlabel(axis_label)
        ax.set_ylabel("")
        ax.set_yticklabels(labels, rotation=30, va="center")


    else:
        ax.boxplot(
            data,
            labels=labels,
            showfliers=False,
            widths=0.6,
            vert=True,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops,
        )
        rng = np.random.default_rng(0)
        for i, col in enumerate(score_df.columns, start=1):
            y = score_df[col].to_numpy()
            x = i + rng.uniform(-jitter, jitter, size=len(y))
            ax.scatter(x, y, s=point_size, alpha=point_alpha, linewidths=0, zorder=3)

        ax.axhline(0.0, color="black", linewidth=box_linewidth)
        ax.set_ylabel(axis_label)
        ax.set_xlabel("")
        ax.set_xticklabels(labels, rotation=30, ha="right")

    if panel_title:
        ax.set_title(panel_title)


def boxplot_two_dataframes_pubready(
    df_left: pd.DataFrame,
    left_baseline_col: str,
    left_competitor_cols: list[str],
    df_right: pd.DataFrame,
    right_baseline_col: str,
    right_competitor_cols: list[str],
    *,
    left_labels: list[str] | None = None,
    right_labels: list[str] | None = None,
    lower_is_better: bool = True,
    mode: str = "log_ratio",  # "log_ratio" or "relative"
    eps: float = 1e-12,
    cap_left: float | tuple[float, float] | None = None,
    cap_right: float | tuple[float, float] | None = None,
    titles: tuple[str | None, str | None] = (None, None),
    horizontal: bool = True,          # flipped (categories on y, values on x)
    share_scale: bool = False,        # different x-limits per panel when horizontal=True
    figsize: tuple[float, float] = (6.8, 2.6),
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
    # saving
    save_path: str | None = None,
    dpi: int = 300,
    transparent: bool = True,
    show: bool = True,
):
    """
    Publication-ready, two-panel boxplot+points (different baseline/competitors per dataframe),
    shown side-by-side. By default: flipped/horizontal boxes and independent scales per panel.

    If save_path is provided, saves the figure (PNG/PDF/SVG depending on extension).
    """
    if left_labels is None:
        left_labels = left_competitor_cols
    if right_labels is None:
        right_labels = right_competitor_cols
    if len(left_labels) != len(left_competitor_cols):
        raise ValueError("left_labels must have same length as left_competitor_cols.")
    if len(right_labels) != len(right_competitor_cols):
        raise ValueError("right_labels must have same length as right_competitor_cols.")

    if title_size is None:
        title_size = font_size
    if tick_size is None:
        tick_size = font_size

    # Compute scores (positive = better)
    left_scores, axis_label = _compute_scores_generic(
        df_left, left_baseline_col, left_competitor_cols, lower_is_better, mode, eps, cap_left
    )
    right_scores, axis_label_r = _compute_scores_generic(
        df_right, right_baseline_col, right_competitor_cols, lower_is_better, mode, eps, cap_right
    )
    # if axis_label_r != axis_label:
    #     axis_label = "Improvement vs baseline"

    # Local rcParams for consistent, paper-friendly styling
    with plt.rc_context({
        "font.size": font_size,
        "axes.titlesize": title_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "axes.linewidth": spine_linewidth,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(
            1, 2,
            figsize=figsize,
            sharex=(share_scale if horizontal else False),
            sharey=(share_scale if not horizontal else False),
        )

        _boxplot_with_points_on_ax(
            ax=axes[0],
            score_df=left_scores,
            labels=left_labels,
            axis_label=axis_label,
            panel_title=titles[0],
            jitter=jitter,
            point_size=point_size,
            point_alpha=point_alpha,
            horizontal=horizontal,
            box_linewidth=box_linewidth,
        )
        _boxplot_with_points_on_ax(
            ax=axes[1],
            score_df=right_scores,
            labels=right_labels,
            axis_label=axis_label_r,
            panel_title=titles[1],
            jitter=jitter,
            point_size=point_size,
            point_alpha=point_alpha,
            horizontal=horizontal,
            box_linewidth=box_linewidth,
        )

        # Clean look: remove top/right spines, keep left/bottom
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(spine_linewidth)
            ax.spines["bottom"].set_linewidth(spine_linewidth)
            ax.tick_params(axis="both", width=spine_linewidth, length=3)

        # Independent scales (recommended per your request)
        if not share_scale:
            for ax in axes:
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
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01, transparent=transparent)

        if show:
            plt.show()

    return fig, (axes[0], axes[1]), (left_scores, right_scores)



def _scores_by_model(
    df: pd.DataFrame,
    models: list[str],
    *,
    competitor_cfg: str,   # "combined" or "prepModel"
    baseline_cfg: str = "baseModel",
    lower_is_better: bool = True,
    mode: str = "log_ratio",
    eps: float = 1e-12,
    cap: float | tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Returns score_df with columns == models (each column is improvements for that model),
    and an axis label.
    """
    cols_needed = []
    for m in models:
        cols_needed += [f"{m}_{baseline_cfg}", f"{m}_{competitor_cfg}"]

    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    scores = {}
    axis_label_out = None

    # compute each model's improvement (competitor over baseline) as its own Series
    for m in models:
        baseline_col = f"{m}_{baseline_cfg}"
        competitor_col = f"{m}_{competitor_cfg}"

        s_df, axis_label = _compute_scores_generic(
            df=df,
            baseline_col=baseline_col,
            competitor_cols=[competitor_col],
            lower_is_better=lower_is_better,
            mode=mode,
            eps=eps,
            cap=cap,
        )
        # s_df has one column: competitor_col
        scores[m] = s_df[competitor_col].reset_index(drop=True)
        axis_label_out = axis_label  # same for all models

    score_df = pd.DataFrame(scores)[models]  # preserve order
    return score_df, axis_label_out

def boxplot_models_combined_vs_tabprep(
    df: pd.DataFrame,
    *,
    models: list[str] = ["RealTabPFN-v2.5", "LightGBM", "TabM"],
    model_labels: list[str] | None = None,
    baseline_cfg: str = "baseModel",
    left_cfg: str = "combined",
    right_cfg: str = "prepModel",
    lower_is_better: bool = True,
    mode: str = "log_ratio",
    eps: float = 1e-12,
    cap_left: float | tuple[float, float] | None = None,
    cap_right: float | tuple[float, float] | None = None,
    titles: tuple[str | None, str | None] = ("Combined vs Base", "Prep vs Base"),
    horizontal: bool = True,
    share_scale: bool = False,
    figsize: tuple[float, float] = (6.8, 2.6),
    jitter: float = 0.11,
    point_size: float = 12.0,
    point_alpha: float = 0.75,
    font_size: float = 8.0,
    title_size: float | None = None,
    tick_size: float | None = None,
    spine_linewidth: float = 0.8,
    box_linewidth: float = 0.9,
    pad: float = 0.02,
    save_path: str | None = None,
    dpi: int = 300,
    transparent: bool = True,
    show: bool = True,
):
    if model_labels is None:
        model_labels = models
    if len(model_labels) != len(models):
        raise ValueError("model_labels must have same length as models.")

    if title_size is None:
        title_size = font_size
    if tick_size is None:
        tick_size = font_size

    left_scores, left_axis_label = _scores_by_model(
        df,
        models=models,
        competitor_cfg=left_cfg,
        baseline_cfg=baseline_cfg,
        lower_is_better=lower_is_better,
        mode=mode,
        eps=eps,
        cap=cap_left,
    )
    right_scores, right_axis_label = _scores_by_model(
        df,
        models=models,
        competitor_cfg=right_cfg,
        baseline_cfg=baseline_cfg,
        lower_is_better=lower_is_better,
        mode=mode,
        eps=eps,
        cap=cap_right,
    )

    with plt.rc_context({
        "font.size": font_size,
        "axes.titlesize": title_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "axes.linewidth": spine_linewidth,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(
            1, 2,
            figsize=figsize,
            sharex=(share_scale if horizontal else False),
            sharey=(share_scale if not horizontal else False),
        )

        _boxplot_with_points_on_ax(
            ax=axes[0],
            score_df=left_scores,
            labels=model_labels,
            axis_label=left_axis_label,
            panel_title=titles[0],
            jitter=jitter,
            point_size=point_size,
            point_alpha=point_alpha,
            horizontal=horizontal,
            box_linewidth=box_linewidth,
        )
        _boxplot_with_points_on_ax(
            ax=axes[1],
            score_df=right_scores,
            labels=model_labels,
            axis_label=right_axis_label,
            panel_title=titles[1],
            jitter=jitter,
            point_size=point_size,
            point_alpha=point_alpha,
            horizontal=horizontal,
            box_linewidth=box_linewidth,
        )

        # Clean look: remove top/right spines, keep left/bottom
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(spine_linewidth)
            ax.spines["bottom"].set_linewidth(spine_linewidth)
            ax.tick_params(axis="both", width=spine_linewidth, length=3)

        for ax, lbl in zip(axes, ["Improvement of adding Prep trials", "Improvement of standalone Prep models"]):
            ax.set_xlabel(lbl, fontsize=9)

        # independent scales (like your current function)
        if not share_scale:
            for ax in axes:
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
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01, transparent=transparent)

        if show:
            plt.show()

    return fig, (axes[0], axes[1]), (left_scores, right_scores)
