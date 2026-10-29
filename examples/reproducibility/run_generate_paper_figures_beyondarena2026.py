"""Reproduce the result figures and tables of the BeyondArena paper (arXiv 2606.30410).

Evaluates the paper's 11 methods (``beyond_method_metadata_2026_camera_ready_collection``; swap in
``beyond_method_metadata_2026_06_paper_collection`` for the results of arXiv v1) on every
split of the 12 sub-benchmarks and writes, under ``save_path``:

* ``subsets/<subset>/``: each sub-benchmark's leaderboard. ``subsets/full/`` also holds the
  figures for all datasets: ``tuning-impact-elo.pdf`` (paper: ``full_tuning-impact-elo.pdf``),
  ``pareto_front_improvability_vs_time_infer.pdf`` and the LaTeX leaderboard ``leaderboard.tex``.
* ``result_plots/``: Elo and improvability per sub-benchmark, per model family
  (``per_family_elo.pdf``, paper: ``result_plot_per_family_plot.pdf``) and per model
  (``per_model_elo.pdf``, paper: ``result_plot_per_model.pdf``).
* ``tuning_trajectories/pareto_n_configs_imp.pdf``: improvability vs. train time over the
  number of tuned configurations.
* ``runtimes/``: the training-runtime distribution across configurations
  (paper: ``data-analysis-runtime-per-method.pdf``).

Two views beyond the paper's figures:

* ``result_plots_le100k/``: the overview figures on the datasets with at most 100k training rows,
  where TabPFN-2.6 and TabDPT ran on every split (no imputed results).
* ``subsets_large_with_tabpfn3/``: the leaderboard on the datasets above 100k training rows with
  TabPFN-3 added. TabPFN-2.6 and TabDPT did not run there, so they are left out.
* ``metafeature_regression/``: which dataset properties explain how well the TFMs (with TabPFN-3 as a fourth)
  do, holding the others fixed (``coefficients.csv``, see ``tfm_metafeature_regression``), and how often each
  property comes with more than 100k training rows (``large_shares.csv``).
* ``paper_figures/``: the last two views at their printed size, side by side in one figure:
  ``per_family_elo_le100k.pdf`` and ``tuning-impact-elo-horizontal_tabpfn_3_large.pdf``.

Results download to ``~/.cache/tabarena/`` on first run.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from tabarena.contexts import BeyondArenaContext
from tabarena.contexts.beyondarena.methods import (
    beyond_method_metadata_2026_camera_ready_collection,
    beyond_tabpfn3_metadata,
)
from tabarena.plot.subset_results import (
    DEFAULT_FAMILY_GROUPS,
    DEFAULT_SUBSET_ORDER,
    SUBSET_LABELS,
    SUBSET_REGION,
    SUBTYPE_STYLES,
    plot_subset_results,
)

#: The paper's sub-benchmarks: label -> subset expressions, each on every split.
PAPER_SUBSETS: dict[str, list[str]] = {
    "random": ["random"],
    "grouped": ["grouped"],
    "temporal": ["temporal"],
    "tiny": ["tiny"],
    "small": ["small"],
    "medium": ["medium"],
    "large": ["large"],
    "low-dim": ["low-dim"],
    "high-dim": ["high-dim"],
    "text": ["text"],
    "high-cardinality": ["high-cardinality"],
    "full": ["all"],
}

#: Regressors of the meta-feature regression: subset-membership dummies, read against IID splits, the medium
#: size bucket (10k to 100k training rows) and low-dim data (at most 100 columns after preprocessing).
META_FEATURES = ["temporal", "grouped", "tiny", "small", "large", "high-dim", "text", "high-cardinality"]
TFMS = ["TabICLv2", "TabPFN-2.6", "TabDPT", "TabPFN-3"]

#: The paper's figures are drawn at their printed size (NeurIPS text width 5.5 in), so LaTeX does not rescale fonts.
PAPER_FIGURE_HEIGHT = 2.2
PAPER_RC = {
    "font.family": "serif",
    "font.size": 6,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
}


def subset_leaderboards(context: BeyondArenaContext, output_dir: Path, subsets: dict[str, list[str]]) -> dict:
    """Each subset's leaderboard, ready for ``plot_subset_results``; figures only for ``"full"``."""
    rename = context.get_method_rename_map()
    leaderboards = {}
    for label, subset in subsets.items():
        leaderboard, results = context.compare(
            output_dir=output_dir / label,
            subset=subset,
            return_results=True,
            plot=label == "full",
        )
        # The overview figures group methods by display name (e.g. "RealMLP", not "TA-REALMLP").
        leaderboard["method"] = leaderboard["config_type"].map(rename).fillna(leaderboard["method"])
        leaderboard["n_datasets_total"] = results["dataset"].nunique()  # the "N=" under each subset
        leaderboards[label] = leaderboard
    return leaderboards


def ols_hc3(y: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
    """OLS of ``y`` on ``features`` plus an intercept, with HC3 heteroskedasticity-robust standard errors."""
    X = np.column_stack([np.ones(len(features)), features.to_numpy(float)])
    y = y.to_numpy(float)
    xtx_inv = np.linalg.inv(X.T @ X)
    coef = xtx_inv @ X.T @ y
    resid = y - X @ coef
    leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    cov = xtx_inv @ (X.T @ (X * (resid**2 / (1 - leverage) ** 2)[:, None])) @ xtx_inv
    se = np.sqrt(np.diag(cov))
    return pd.DataFrame(
        {
            "coef": coef,
            "ci_low": coef - 1.96 * se,
            "ci_high": coef + 1.96 * se,
            "p_value": 2 * stats.norm.sf(np.abs(coef / se)),
            "r2": 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean())),
            "n_datasets": len(y),
        },
        index=pd.Index(["const", *features.columns], name="feature"),
    )


def dataset_properties(context: BeyondArenaContext) -> pd.DataFrame:
    """One row per dataset with a 0/1 column per ``META_FEATURES`` subset."""
    grid = context.task_metadata_collection.task_grid().groupby("dataset").first()
    return pd.DataFrame(
        {name: context.subset_predicates[name].evaluate(grid, name=name).astype(float) for name in META_FEATURES}
    )


def large_shares(features: pd.DataFrame) -> pd.DataFrame:
    """The share of large datasets (more than 100k training rows) with and without each other property."""
    return pd.DataFrame(
        {
            name: {
                "n_with": int(features[name].sum()),
                "large_share_with": features.loc[features[name] == 1, "large"].mean(),
                "large_share_without": features.loc[features[name] == 0, "large"].mean(),
            }
            for name in META_FEATURES
            if name != "large"
        }
    ).T.rename_axis("property")


def tfm_metafeature_regression(context: BeyondArenaContext, results_per_split: pd.DataFrame) -> pd.DataFrame:
    """Regress per-dataset TFM performance on dataset properties, each effect holding the others fixed.

    Two targets per dataset, averaged over its splits: ``best_tfm_rank``, the best TFM's rank among all methods
    (1 = best; a positive coefficient means worse for TFMs), and ``tfm_margin``, the best non-TFM's minus the best
    TFM's min-max normalized error (positive means a TFM beats every other method). Imputed results are left out:
    a stand-in Random Forest result is not a TFM's, so on the datasets above 100k rows, where TabPFN-2.6 and TabDPT
    did not run, the TFMs are TabICLv2 and TabPFN-3 and the ranking has two entries fewer. ``univariate_*`` is each
    property's effect fit alone.
    """
    features = dataset_properties(context)

    df = results_per_split
    if "imputed" in df:
        df = df[~df["imputed"].fillna(False).astype(bool)]
    df = df.copy()
    df["is_tfm"] = df["config_type"].map(context.get_method_rename_map()).isin(TFMS)
    errors = df.groupby(["dataset", "fold"])["metric_error"]
    df["rank"] = errors.rank()
    lowest = errors.transform("min")
    df["norm_error"] = ((df["metric_error"] - lowest) / (errors.transform("max") - lowest)).fillna(0.0)
    tfm = df[df["is_tfm"]].groupby(["dataset", "fold"])
    targets = (
        pd.DataFrame(
            {
                "best_tfm_rank": tfm["rank"].min(),
                "tfm_margin": df[~df["is_tfm"]].groupby(["dataset", "fold"])["norm_error"].min()
                - tfm["norm_error"].min(),
            }
        )
        .groupby("dataset")
        .mean()
    )

    tables = {}
    for target in targets:
        y, X = targets[target], features.loc[targets.index]
        table = ols_hc3(y, X)
        univariate = pd.DataFrame([ols_hc3(y, X[[name]]).loc[name] for name in META_FEATURES])
        table[["univariate_coef", "univariate_p_value"]] = univariate[["coef", "p_value"]]
        tables[target] = table
    return pd.concat(tables, names=["target"])


def plot_per_family_compact(leaderboards: dict[str, pd.DataFrame], path: Path, width: float = 3.46) -> None:
    """Figure 1's per-family Elo lines (best member per family and pipeline, its CI as error bar) at print size."""
    order = [label for label in DEFAULT_SUBSET_ORDER if label in leaderboards]
    families = list(dict.fromkeys(cfg["family"] for cfg in DEFAULT_FAMILY_GROUPS.values()))
    colors = dict(zip(families, sns.color_palette("bright", n_colors=len(families)), strict=True))
    lows, highs = [], []
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(width, PAPER_FIGURE_HEIGHT), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.01, w_pad=0.01)
        for cfg in DEFAULT_FAMILY_GROUPS.values():
            members = {m if isinstance(m, tuple) else (m, "default") for m in cfg["methods"]}
            best = []
            for label in order:
                lb = leaderboards[label]
                rows = lb[[(m, st) in members for m, st in zip(lb["method"], lb["method_subtype"], strict=True)]]
                best.append(rows.loc[rows["elo"].idxmax()] if len(rows) else None)
            if all(row is None for row in best):
                continue
            x = [i for i, row in enumerate(best) if row is not None]
            rows = pd.DataFrame([row for row in best if row is not None])
            marker, linestyle, _ = SUBTYPE_STYLES[cfg["subtype"]]
            color = colors[cfg["family"]]
            ax.errorbar(
                x,
                rows["elo"],
                yerr=[rows["elo-"], rows["elo+"]],
                color=color,
                linestyle=linestyle,
                linewidth=0.9,
                marker=marker,
                markersize=3.5,
                markerfacecolor="white",
                markeredgewidth=0.8,
                elinewidth=0.6,
                capsize=1.5,
                alpha=0.9,
            )
            lows.append((rows["elo"] - rows["elo-"]).min())
            highs.append((rows["elo"] + rows["elo+"]).max())
        # Dashed dividers between the task, scale, dimensionality and feature groups, and before "All".
        for i in range(1, len(order)):
            if SUBSET_REGION.get(order[i]) != SUBSET_REGION.get(order[i - 1]):
                ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=0.6)
        # The dataset counts on a row of their own (5.5 pt) under the axis, the subset names rotated below it: at
        # print size the names do not fit side by side.
        n_datasets = [int(leaderboards[label]["n_datasets_total"].iloc[0]) for label in order]
        ax.xaxis.remove_overlapping_locs = False
        ax.set_xticks(range(len(order)), labels=[SUBSET_LABELS[label] for label in order])
        ax.set_xticks(range(len(order)), labels=[f"N={n}" for n in n_datasets], minor=True)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", which="major", length=0, pad=8)
        ax.tick_params(axis="x", which="minor", length=0, pad=1.5, labelsize=5.5, labelcolor="dimgray")
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.grid(axis="x", visible=False)
        bottom, top = np.floor(min(lows) / 200) * 200, np.ceil(max(highs) / 200) * 200
        ax.set_ylim(bottom, top)
        ax.set_yticks(np.arange(bottom, top + 1, 200))
        ax.set_ylabel("Elo (Best Model)")
        # Two legend rows, filled column by column: families on top, pipelines below.
        family_handles = [Patch(color=colors[f], label=f) for f in families]
        pipeline_handles = [
            Line2D(
                [],
                [],
                color="black",
                linestyle=SUBTYPE_STYLES[st][1],
                linewidth=0.9,
                marker=SUBTYPE_STYLES[st][0],
                markersize=3.5,
                markerfacecolor="white",
                markeredgewidth=0.8,
                label=label,
            )
            for st, label in [("ICL", "ICL"), ("D", "Default"), ("T+E", "Tuned + Ens.")]
        ]
        blank = Line2D([], [], linestyle="none", label=" ")
        handles = [h for pair in zip(family_handles, [*pipeline_handles, blank], strict=True) for h in pair]
        fig.legend(
            handles=handles,
            loc="outside upper center",
            ncol=4,
            frameon=False,
            handlelength=1.8,
            handleheight=0.8,
            columnspacing=1.0,
            handletextpad=0.4,
            borderaxespad=0.1,
        )
        fig.savefig(path)
        plt.close(fig)


def plot_tuning_impact_compact(
    leaderboard: pd.DataFrame, path: Path, width: float = 1.93, xlim: tuple[float, float] = (400, 1480)
) -> None:
    """The leaderboard's Elo per model as nested horizontal bars per pipeline (the tuning-impact plot) at print size."""
    pastel, deep = sns.color_palette("pastel"), sns.color_palette("deep")
    pipelines = [("tuned_ensemble", "Tuned + Ens.", 0.6, 2), ("tuned", "Tuned", 0.5, 1), ("default", "Default", 0.4, 0)]
    models = leaderboard.groupby("method")["elo"].max().sort_values().index  # the best model on top
    y = np.arange(len(models))
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(width, PAPER_FIGURE_HEIGHT), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.01, w_pad=0.01)
        for subtype, label, height, c in pipelines:
            rows = leaderboard[leaderboard["method_subtype"] == subtype].set_index("method").reindex(models)
            has = rows["elo"].notna().to_numpy()
            ax.barh(y[has], rows["elo"][has], height=height, color=pastel[c], label=label, zorder=2)
            ax.errorbar(
                rows["elo"][has],
                y[has],
                xerr=[rows["elo-"][has], rows["elo+"][has]],
                fmt="none",
                ecolor=deep[c],
                elinewidth=0.6,
                capsize=1.5,
                zorder=3,
            )
        ax.set_yticks(y, labels=models)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(*xlim)
        ax.set_xticks([400, 800, 1200])
        ax.set_xlabel("Elo")
        ax.grid(axis="y", visible=False)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles[::-1],
            labels[::-1],
            loc="outside upper center",
            ncol=3,
            frameon=False,
            fontsize=5.5,
            handlelength=1.0,
            handleheight=0.8,
            columnspacing=0.6,
            handletextpad=0.3,
            borderaxespad=0.1,
        )
        fig.savefig(path)
        plt.close(fig)


if __name__ == "__main__":
    save_path = Path("output_paper_results_beyondarena2026")  # folder to save all figures and tables

    methods = beyond_method_metadata_2026_camera_ready_collection.method_metadata_lst
    context = BeyondArenaContext(methods=methods)

    leaderboards = subset_leaderboards(context, save_path / "subsets", PAPER_SUBSETS)
    plot_subset_results(leaderboards, save_path / "result_plots", family_legend_members="all")

    leaderboards_le100k = subset_leaderboards(
        context,
        save_path / "subsets_le100k",
        {label: [*subset, "!large"] for label, subset in PAPER_SUBSETS.items() if label != "large"},
    )
    plot_subset_results(leaderboards_le100k, save_path / "result_plots_le100k", family_legend_members="all")
    (save_path / "paper_figures").mkdir(parents=True, exist_ok=True)
    plot_per_family_compact(leaderboards_le100k, save_path / "paper_figures" / "per_family_elo_le100k.pdf")

    context_tabpfn3 = BeyondArenaContext(methods=[*methods, beyond_tabpfn3_metadata])
    leaderboard_large = context_tabpfn3.compare(
        output_dir=save_path / "subsets_large_with_tabpfn3",
        subset=["large"],
        remove_imputed=True,  # drops TabPFN-2.6 and TabDPT, imputed on the large datasets
        elo_ymin=400,  # the baselines fall well below the default axis floor of 800 here
    )
    leaderboard_large["method"] = (
        leaderboard_large["config_type"]
        .map(context_tabpfn3.get_method_rename_map())
        .fillna(leaderboard_large["method"])
    )
    plot_tuning_impact_compact(
        leaderboard_large, save_path / "paper_figures" / "tuning-impact-elo-horizontal_tabpfn_3_large.pdf"
    )

    # The regression over every split of all datasets, with TabPFN-3 among the TFMs.
    _, results_tabpfn3 = context_tabpfn3.compare(output_dir=None, subset=["all"], return_results=True)
    regression = tfm_metafeature_regression(context_tabpfn3, results_tabpfn3)
    (save_path / "metafeature_regression").mkdir(exist_ok=True)
    regression.to_csv(save_path / "metafeature_regression" / "coefficients.csv")
    regression.round(3).to_latex(save_path / "metafeature_regression" / "coefficients.tex")
    large_shares(dataset_properties(context_tabpfn3)).to_csv(save_path / "metafeature_regression" / "large_shares.csv")
    print(regression.round(3).to_markdown())

    context.plot_tuning_trajectories(
        save_path=save_path / "tuning_trajectories",
        exclude_imputed=False,
        # Color methods by each figure's own y-axis (improvability in pareto_n_configs_imp.pdf).
        use_elo_method_order=False,
        plot_kwargs={"ylim_imp": (None, None), "legend_in_plot": True, "figsize": (6, 4.5)},
    )

    context.plot_runtime_per_method(
        save_path=save_path / "runtimes",
        deep_dive_kwargs={"time_limit_soft": 10800, "time_limit_hard": 14400},  # the paper's 4 h limit
    )
