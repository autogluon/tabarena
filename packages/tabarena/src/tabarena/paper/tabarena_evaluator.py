from __future__ import annotations

import copy
import functools
import itertools
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.common.savers import save_pd

from bencheval.tabarena import TabArena
from tabarena.benchmark.task.metadata import TaskMetadataCollection, default_task_metadata_collection
from tabarena.paper.paper_utils import get_f_map_suffix_plots, get_framework_type_method_names, get_method_rename_map
from tabarena.plot.dataset_analysis import plot_train_time_deep_dive
from tabarena.plot.plot_ens_weights import create_heatmap
from tabarena.plot.plot_pareto_frontier import (
    plot_pareto as _plot_pareto,
)
from tabarena.utils.normalized_scorer import NormalizedScorer

MethodLabelStyle = str | Mapping[str, object]


@dataclass
class TuneMethodOverride:
    """Extra bar style for `TabArenaEvaluator.plot_tuning_impact`.

    Rows whose method name is in `methods` get retagged with `tune_method`
    (replacing whatever value `f_map_inverse` would have produced) and are
    rendered as a separate bar in the plot. Use this for fixed-API methods
    (e.g. TabPFN-3, TabPFN-3-Thinking) — or any future variant — that need to
    appear alongside the standard default/tuned/tuned_ensembled bars.

    Multiple overrides can coexist in the same plot; each contributes one bar
    dict and one entry to the elo err-color loop.
    """

    tune_method: str  # value written to df["tune_method"]; must be unique per plot
    methods: list[str]  # method names (in `method_col`) to retag
    bar_label: str  # legend label for this bar
    bar_color: str  # bar fill color
    bar_width: float = 0.4
    err_color: str | None = None  # whisker color; defaults to errcolors[0] (default-blue)
    err_linewidth_key: str = "tuned_ensembled"  # which entry of err_linewidths to use
    rename_map: dict[str, str] = field(default_factory=dict)  # applied to method_col before f_map_inverse
    promote_from_baselines: bool = False  # move `methods` out of `baselines` into `framework_types`


@functools.cache
def _init_global_rcparams() -> None:
    """Apply TabArena's global matplotlib style (once).

    Cached + lazy so that importing this module does not import ``matplotlib``/``tueplots``
    — this keeps the plotting stack off the ``TabArenaContext`` import path. Called from
    ``TabArenaEvaluator.__init__`` (previously ran at module import time).
    """
    import matplotlib
    from tueplots import fontsizes

    matplotlib.rcParams.update(fontsizes.neurips2024())
    matplotlib.rcParams.update(
        {
            "text.latex.preamble": r"\usepackage{times} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{xcolor}",
        }
    )


def darken_color(color_str, amount=0.5):
    import matplotlib.colors as mcolors

    # Convert color string to RGB tuple (values between 0 and 1)
    rgb = mcolors.to_rgb(color_str)
    # Interpolate with black (0, 0, 0)
    return tuple((1 - amount) * c for c in rgb)


# FIXME: ensemble weights can get if including `config_hyperparameters` as input
class TabArenaEvaluator:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        task_metadata: TaskMetadataCollection | None = None,
        config_types: dict[str, str] | None = None,
        method_col: str = "method",
        error_col: str = "metric_error",
        methods: list[str] | None = None,
        folds: list[int] | None = None,
        datasets: list[str] | None = None,
        problem_types: list[str] | None = None,
        method_rename_map: dict[str, str] | None = None,
        banned_model_types: list[str] | None = None,
        banned_pareto_methods: list[str] | None = None,
        elo_bootstrap_rounds: int = 200,
        elo_ymin: float = 800,
        keep_best: bool = False,
        figure_file_type: str = "pdf",
        use_latex: bool = False,
        tabarena_context=None,  # FIXME: Remove this and refactor after leaderboard v0.2 upload, this is purely to get things working fast
    ):
        """Parameters
        ----------

        Methods:
            filter methods
        folds
            filter folds
        datasets
            filter datasets
        problem_types
            filter problem_types
        elo_bootstrap_rounds
            10 = toy
            100 = paper
        kwargs
        """
        if task_metadata is None:
            task_metadata = default_task_metadata_collection()
        if banned_pareto_methods is None:
            banned_pareto_methods = []
        if method_rename_map is None:
            method_rename_map = {}
        self.output_dir: Path = Path(output_dir)
        self.task_metadata = task_metadata
        self.method_col = method_col
        self.error_col = error_col
        self.config_types = config_types
        self.figure_file_type = figure_file_type
        self.banned_pareto_methods = banned_pareto_methods
        self._method_rename_map = method_rename_map

        self.datasets = datasets
        self.problem_types = problem_types
        self.methods = methods
        self.folds = folds
        self.elo_bootstrap_rounds = elo_bootstrap_rounds
        self.elo_ymin = elo_ymin
        self.banned_model_types = banned_model_types
        self.keep_best = keep_best

        _init_global_rcparams()
        self.use_latex = use_latex
        if self.use_latex:
            import matplotlib
            from tueplots import bundles, fonts, fontsizes

            matplotlib.rcParams.update(bundles.neurips2024())
            matplotlib.rcParams.update(fonts.neurips2024_tex())
            self.rc_context_params = {
                "font.family": "serif",
                "text.usetex": True,
            } | fontsizes.neurips2024(default_smaller=0)
        else:
            self.rc_context_params = {}

        self.style_order = [
            "Default",
            "Tuned",
            "Tuned + Ens.",
            "Baseline",
            "Best",
            "Default, Holdout",
            "Tuned, Holdout",
            "Tuned + Ens., Holdout",
        ]

        self.style_markers = {
            "Default": "o",
            "Tuned": "s",
            "Tuned + Ens.": "X",
            "Baseline": "D",
            "Best": "*",
            "Default, Holdout": "^",
            "Tuned, Holdout": "<",
            "Tuned + Ens., Holdout": ">",
        }

        if tabarena_context is not None:
            self.method_metadata_info = tabarena_context.method_metadata_collection.info()
            self.method_metadata_info = self.method_metadata_info.rename(
                columns={
                    "method": "ta_name",
                    "artifact_name": "ta_suite",
                }
            )
            self.method_metadata_info = self.method_metadata_info.drop(columns=["method_type"])
        else:
            self.method_metadata_info = None

    def compute_normalized_error_dynamic(self, df_results: pd.DataFrame) -> pd.DataFrame:
        df_results = df_results.copy(deep=True)
        df_results_og = df_results.copy(deep=True)

        df_results = df_results.drop(columns=["normalized-error-dataset", "normalized-error-task"], errors="ignore")

        method_col = self.method_col

        df_results_per_dataset = (
            df_results.groupby([method_col, "dataset"])[self.error_col].mean().reset_index(drop=False)
        )

        # Alternative, this also incorporates Portfolios and HPO into the normalized scoring. This makes normalized-error dependent on what simulations we run.
        # This is unbiased against very strong simulation results because the best method defines what is `0.0` on a dataset.
        normalized_scorer_dataset = NormalizedScorer(
            df_results_per_dataset,
            tasks=list(df_results_per_dataset["dataset"].unique()),
            baseline=None,
            task_col="dataset",
            framework_col=method_col,
            metric_error_col=self.error_col,
        )

        all_tasks = df_results[["dataset", "fold"]].drop_duplicates().values.tolist()
        all_tasks = [tuple(task) for task in all_tasks]

        normalized_scorer_task = NormalizedScorer(
            df_results,
            tasks=all_tasks,
            baseline=None,
            task_col=["dataset", "fold"],
            framework_col=method_col,
        )

        df_results["normalized-error-task"] = [
            normalized_scorer_task.rank(task=(dataset, fold), error=error)
            for (dataset, fold, error) in zip(
                df_results["dataset"], df_results["fold"], df_results[self.error_col], strict=False
            )
        ]

        df_results_per_dataset["normalized-error-dataset"] = [
            normalized_scorer_dataset.rank(task=dataset, error=error)
            for (dataset, error) in zip(
                df_results_per_dataset["dataset"], df_results_per_dataset[self.error_col], strict=False
            )
        ]

        df_results_per_dataset = df_results_per_dataset.set_index(["dataset", method_col], drop=True)[
            "normalized-error-dataset"
        ]
        df_results = df_results.merge(df_results_per_dataset, left_on=["dataset", method_col], right_index=True)

        df_results_og["normalized-error-dataset"] = df_results["normalized-error-dataset"]
        df_results_og["normalized-error-task"] = df_results["normalized-error-task"]
        return df_results_og

    @classmethod
    def _get_config_types(cls, df_results: pd.DataFrame) -> list[str]:
        return sorted(
            [
                config_type
                for config_type in df_results["config_type"].unique()
                if config_type is not None and isinstance(config_type, str)
            ]
        )

    # TODO: Remove the need for this, have the original results have the correct name to begin with
    @classmethod
    def _rename_dict(cls) -> dict:
        return {}

    def compare_methods_per_dataset(
        self,
        df: pd.DataFrame,
        *,
        method_a: str,
        method_b: str,
        dataset_col: str = "dataset",
        method_col: str = "method",
        split_col: str = "split",
        error_col: str = "metric_error",
        require_paired_splits: bool = True,
        zscore_ddof: int = 1,  # 1 = sample std (recommended), 0 = population std
    ) -> pd.DataFrame:
        """Compare two methods on metric_error (lower is better; 0 is perfect), per dataset.

        Adds a per-dataset z-score of the paired mean difference across splits:
            z = mean(diff) / (std(diff) / sqrt(n))
        where diff = A - B (negative => A better).

        Notes:
          - With very few splits, z can be unstable.
          - If std(diff)==0 or n<2, z is returned as NaN.
        """
        needed = {dataset_col, method_col, split_col, error_col}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"df missing columns: {sorted(missing)}")

        sub = df[df[method_col].isin([method_a, method_b])][[dataset_col, split_col, method_col, error_col]].copy()

        # Align A/B on the same (dataset, split)
        wide = sub.pivot_table(
            index=[dataset_col, split_col],
            columns=method_col,
            values=error_col,
            aggfunc="mean",  # handle duplicates
        ).rename(columns={method_a: "A", method_b: "B"})

        if require_paired_splits:
            wide = wide.dropna(subset=["A", "B"])

        # Paired quantities
        wide["diff_A_minus_B"] = wide["A"] - wide["B"]  # <0 => A better
        wide["A_wins"] = (wide["A"] < wide["B"]).astype(float)
        wide["ties"] = (wide["A"] == wide["B"]).astype(float)
        wide["A_losses"] = (wide["A"] > wide["B"]).astype(float)
        wide["rel_impr_A_vs_B"] = np.where(wide["B"] > 0, (wide["B"] - wide["A"]) / wide["B"], np.nan)

        def _z_from_diffs(diffs: pd.Series) -> float:
            diffs = diffs.dropna()
            n = int(diffs.shape[0])
            if n < 2:
                return np.nan
            s = diffs.std(ddof=zscore_ddof)
            if not np.isfinite(s) or s == 0:
                return np.nan
            return float(diffs.mean() / (s / np.sqrt(n)))

        def _agg(g: pd.DataFrame) -> pd.Series:
            paired = g.dropna(subset=["A", "B"])
            diffs = paired["diff_A_minus_B"]

            out = {
                "n_splits": len(paired),
                "mean_error_A": paired["A"].mean(),
                "mean_error_B": paired["B"].mean(),
                "median_error_A": paired["A"].median(),
                "median_error_B": paired["B"].median(),
                "std_error_A": paired["A"].std(ddof=1),
                "std_error_B": paired["B"].std(ddof=1),
                "mean_diff_A_minus_B": diffs.mean(),
                "median_diff_A_minus_B": diffs.median(),
                "std_diff_A_minus_B": diffs.std(ddof=1),
                "win_rate_A": paired["A_wins"].mean(),
                "tie_rate": paired["ties"].mean(),
                "loss_rate_A": paired["A_losses"].mean(),
                "mean_rel_impr_A_vs_B": paired["rel_impr_A_vs_B"].mean(),
                # z-score based on variability across splits ("seeds")
                "z_score_mean_diff": _z_from_diffs(diffs),
            }
            return pd.Series(out)

        stats = wide.groupby(level=0, sort=True).apply(_agg)

        stats["winner_by_mean"] = np.where(
            stats["mean_error_A"] < stats["mean_error_B"],
            method_a,
            np.where(stats["mean_error_A"] > stats["mean_error_B"], method_b, "tie"),
        )

        return stats.reset_index(names=[dataset_col])

    def dataset_pairwise_split_agreement(
        self,
        df_dataset: pd.DataFrame,
        *,
        method_col: str = "method",
        split_col: str = "split",
        error_col: str = "metric_error",
        tie_policy: str = "half",  # {"ignore", "half", "count_as_disagree"}
        min_splits: int = 2,
        tie_decimals: int | None = None,
    ) -> dict[str, float]:
        """Compute average split agreement across all method pairs within ONE dataset.

        Agreement for pair (i,j):
          p_ij = fraction of splits where error_i < error_j
          agreement_ij = max(p_ij, 1 - p_ij)

        tie_policy:
          - "ignore": exclude split ties from the denominator for that pair
          - "half": count ties as half-win for both sides (so they push p toward 0.5)
          - "count_as_disagree": treat ties as neither side winning (pushes agreement down)

        Returns a dict with:
          - n_methods, n_splits
          - avg_pairwise_agreement (float in [0.5, 1])
          - avg_pairwise_agreement_weighted (optional nuance: pairs weighted by usable splits)
        """
        # Pivot to a dense matrix: rows=splits, cols=methods, values=error
        wide = df_dataset.pivot_table(
            index=split_col,
            columns=method_col,
            values=error_col,
            aggfunc="mean",
        )
        # Keep only methods that appear on >=1 split (already true) and splits with >=2 methods
        if wide.shape[0] < min_splits or wide.shape[1] < 2:
            return {
                "n_methods": float(wide.shape[1]),
                "n_splits": float(wide.shape[0]),
                "avg_pairwise_agreement": np.nan,
                "avg_pairwise_agreement_weighted": np.nan,
            }

        X = wide.to_numpy(dtype=float)  # shape (S, M)
        S, M = X.shape

        # Optional: round to tie_decimals so two methods that hit the same
        # metric via different code paths (floating-point noise in the last
        # few bits) still count as tied. Default None preserves the original
        # exact-equality behavior.
        X_cmp = np.round(X, tie_decimals) if tie_decimals is not None else X

        # Broadcast comparisons: for each split s, compare all method pairs (i,j)
        # win[s,i,j] = 1 if X[s,i] < X[s,j]
        # tie[s,i,j] = 1 if X[s,i] == X[s,j]
        win = X_cmp[:, :, None] < X_cmp[:, None, :]
        tie = X_cmp[:, :, None] == X_cmp[:, None, :]

        # Handle missing values: comparisons where either side is NaN should not count
        valid = np.isfinite(X)
        valid_pair = valid[:, :, None] & valid[:, None, :]  # shape (S,M,M)
        win = win & valid_pair
        tie = tie & valid_pair & tie

        wins = win.sum(axis=0).astype(float)  # (M,M): #splits i beats j
        ties = tie.sum(axis=0).astype(float)  # (M,M): #split ties
        usable = valid_pair.sum(axis=0).astype(float)  # (M,M): #splits with both present

        # We only want i<j pairs (upper triangle) to avoid double counting.
        iu = np.triu_indices(M, k=1)

        if tie_policy == "ignore":
            denom = usable - ties
            # avoid divide-by-zero (all ties or no usable splits for that pair)
            p = np.divide(wins, denom, out=np.full_like(wins, np.nan), where=denom > 0)
            agree = np.maximum(p, 1.0 - p)

            agree_pairs = agree[iu]
            denom_pairs = denom[iu]

            avg = float(np.nanmean(agree_pairs))
            # weighted by how many non-tie split comparisons each pair had
            wavg = (
                float(np.nansum(agree_pairs * denom_pairs) / np.nansum(denom_pairs))
                if np.nansum(denom_pairs) > 0
                else np.nan
            )

        elif tie_policy == "half":
            # treat ties as half-win for both sides
            denom = usable
            p = np.divide(wins + 0.5 * ties, denom, out=np.full_like(wins, np.nan), where=denom > 0)
            agree = np.maximum(p, 1.0 - p)

            agree_pairs = agree[iu]
            denom_pairs = denom[iu]

            avg = float(np.nanmean(agree_pairs))
            wavg = (
                float(np.nansum(agree_pairs * denom_pairs) / np.nansum(denom_pairs))
                if np.nansum(denom_pairs) > 0
                else np.nan
            )

        elif tie_policy == "count_as_disagree":
            # ties count in denom but not in wins => they push p toward 0 and agreement downward
            denom = usable
            p = np.divide(wins, denom, out=np.full_like(wins, np.nan), where=denom > 0)
            agree = np.maximum(p, 1.0 - p)

            agree_pairs = agree[iu]
            denom_pairs = denom[iu]

            avg = float(np.nanmean(agree_pairs))
            wavg = (
                float(np.nansum(agree_pairs * denom_pairs) / np.nansum(denom_pairs))
                if np.nansum(denom_pairs) > 0
                else np.nan
            )

        else:
            raise ValueError("tie_policy must be one of: {'ignore','half','count_as_disagree'}")

        return {
            "n_methods": float(M),
            "n_splits": float(S),
            "avg_pairwise_agreement": avg,
            "avg_pairwise_agreement_weighted": wavg,
        }

    def add_dataset_agreement_metric(
        self,
        df: pd.DataFrame,
        *,
        dataset_col: str = "dataset",
        method_col: str = "method",
        split_col: str = "split",
        error_col: str = "metric_error",
        tie_policy: str = "half",
    ) -> pd.DataFrame:
        """Compute the metric for every dataset and return a summary DataFrame."""
        # Example:
        # agreement_df = add_dataset_agreement_metric(df, tie_policy="ignore")
        # agreement_df.sort_values("avg_pairwise_agreement", ascending=False).head()
        rows = []
        for ds, g in df.groupby(dataset_col, sort=True):
            out = self.dataset_pairwise_split_agreement(
                g,
                method_col=method_col,
                split_col=split_col,
                error_col=error_col,
                tie_policy=tie_policy,
            )
            out[dataset_col] = ds
            rows.append(out)
        return pd.DataFrame(rows)

    def plot_imp(self, results_per_task: pd.DataFrame):
        # 1. Find the index of the lowest-rank row per dataset
        best_idx = results_per_task.groupby("dataset", sort=False)["rank"].idxmin()

        # 2. Extract method name and metric_error for those rows
        best_per_dataset = results_per_task.loc[best_idx, ["dataset", "method", "metric_error"]].rename(
            columns={
                "method": "best_method",
                "metric_error": "best_metric_error",
            },
        )

        # 3. Join back to the original DataFrame
        results_per_task = results_per_task.merge(
            best_per_dataset,
            on="dataset",
            how="left",
        )

        import matplotlib.pyplot as plt

        # --- dataset order (same as you already do) ---
        order = results_per_task.groupby("dataset")["improvability"].median().sort_values().index

        results_per_task = (
            results_per_task.assign(dataset=pd.Categorical(results_per_task["dataset"], categories=order, ordered=True))
            .sort_values("dataset")
            .reset_index(drop=True)
        )

        # --- helper for filenames ---
        def _safe_filename(s: str) -> str:
            return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)

        out_dir = Path("plots_improvability_by_method")
        out_dir.mkdir(parents=True, exist_ok=True)

        methods = results_per_task["method"].dropna().unique()

        for method in methods:
            fig, ax = plt.subplots(figsize=(max(10, 0.4 * len(order)), 6))

            # background boxplot (all methods)
            results_per_task.boxplot(
                column="improvability",
                by="dataset",
                grid=False,
                rot=90,
                ax=ax,
            )

            # compute ONE value per dataset for this method (median across folds / rows)
            per_dataset = (
                results_per_task.loc[results_per_task["method"] == method]
                .groupby("dataset", observed=True)["improvability"]
                .median()
                .reindex(order)  # align with x-axis order
            )

            # x positions used by pandas boxplot are 1..N
            x = range(1, len(order) + 1)
            y = per_dataset.to_numpy()

            # overlay in red (skip NaNs automatically by masking)
            mask = pd.notna(y)
            ax.scatter([xi for xi, ok in zip(x, mask, strict=False) if ok], y[mask], color="red", s=30, zorder=3)

            ax.set_title(f"Improvability per Dataset — highlight: {method}")
            fig.suptitle("")  # remove pandas default subtitle
            ax.set_xlabel("Dataset")
            ax.set_ylabel("Improvability")

            fig.tight_layout()
            out_path = out_dir / f"improvability_boxplot_highlight_{_safe_filename(str(method))}.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        # TODO: Get rank 1 method per task/split, include alongside as column w/ value for comparison

    def eval(
        self,
        df_results: pd.DataFrame,
        include_norm_score: bool = False,
        use_gmean: bool = False,
        imputed_names: list[str] | None = None,
        baselines: list[str] | str | None = "auto",
        baseline_colors: list[str] | None = None,
        plot_tune_types: list[str] | None = None,
        plot_times: bool = False,
        plot_extra_barplots: bool = False,
        plot_cdd: bool = True,
        plot_critical_diagrams: bool = False,
        plot_runtimes: bool = False,
        plot_pareto: bool = True,
        compute_fold_stability_curves: bool = False,
        compute_fold_similarity: bool = False,
        fold_similarity_kwargs: dict | None = None,
        calibration_framework: str | None = "auto",
        average_seeds: bool = False,
        leaderboard_kwargs: dict | None = None,
        plot_with_baselines: bool = True,
        plot_tuning_kwargs: dict | None = None,
        banned_methods: list[str] | None = None,
        verbose: bool = True,
        show_winrate_title: bool = False,
        show_tuning_impact_title: bool = False,
        subset_label: str | None = None,
        winrate_method_rename: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        if banned_methods is not None:
            df_results = df_results[~df_results["method"].isin(banned_methods)]
        if leaderboard_kwargs is None:
            leaderboard_kwargs = {}
        leaderboard_kwargs = leaderboard_kwargs.copy()
        if plot_tuning_kwargs is None:
            plot_tuning_kwargs = {}
        plot_tuning_kwargs = plot_tuning_kwargs.copy()

        # Auto-build the tuning-impact suptitle in the same
        # "TabArena-<subset> …" style as the winrate and tuning-trajectory
        # titles. Threaded via ``plot_tuning_kwargs["title"]`` (which every
        # ``self.plot_tuning_impact(...)`` call below picks up); an
        # explicitly caller-supplied title still wins. ``Leaderboard``
        # rather than ``Tuning Impact`` because in practice only the elo
        # variants are emitted (``include_norm_score`` defaults to False)
        # and the elo bar plot reads as a per-method leaderboard.
        # ``None`` and ``"all"`` collapse to the unsuffixed ``TabArena``
        # prefix (aggregate / no-meaningful-subset case).
        if show_tuning_impact_title and "title" not in plot_tuning_kwargs:
            if subset_label and subset_label != "all":
                plot_tuning_kwargs["title"] = f"TabArena-{subset_label} Leaderboard"
            else:
                plot_tuning_kwargs["title"] = "TabArena Leaderboard"

        if calibration_framework is not None and calibration_framework == "auto":
            calibration_framework = "RF (default)"
        df_results = df_results.copy(deep=True)
        if "method_metadata" in df_results.columns:
            # currently no need to use this column
            df_results = df_results.drop(columns=["method_metadata"])
        if "imputed" not in df_results.columns:
            df_results["imputed"] = False
        df_results["imputed"] = df_results["imputed"].astype(int).fillna(0).astype(bool)

        # rename methods
        _rename_dict = self._rename_dict()
        df_results[self.method_col] = df_results[self.method_col].map(_rename_dict).fillna(df_results[self.method_col])
        if calibration_framework is not None:
            calibration_framework = _rename_dict.get(calibration_framework, calibration_framework)

        if imputed_names is None:
            imputed_names = self.get_imputed_names(df_results=df_results)
        if verbose:
            print(f"Model for which results were imputed: {imputed_names}")

        self.assert_no_duplicates(df_results=df_results)
        self.assert_no_nan_methods(df_results=df_results)

        df_results = self.filter_results(df_results=df_results)

        if isinstance(baselines, list):
            baselines = [_rename_dict.get(b, b) for b in baselines]
        baselines, baseline_colors = self._process_baselines(
            df_results=df_results,
            baselines=baselines,
            baseline_colors=baseline_colors,
        )

        framework_types = self._get_config_types(df_results=df_results[~df_results["method"].isin(baselines)])

        df_results_rank_compare = copy.deepcopy(df_results)

        _f_map, f_map_type, _f_map_inverse, f_map_type_name = self.get_framework_type_method_names(
            framework_types=framework_types,
        )

        df_results_rank_compare = df_results_rank_compare[
            (~df_results_rank_compare[self.method_col].map(f_map_type).isna())
            | (df_results_rank_compare[self.method_col].isin(baselines))
        ]

        # ----- end removing unused methods -----

        method_info: pd.DataFrame = self.get_method_info(df=df_results_rank_compare)
        if self.method_metadata_info is not None:
            method_info_full = pd.merge(
                left=method_info.reset_index(), right=self.method_metadata_info, on=["ta_name", "ta_suite"]
            )
            save_pd.save(path=self.output_dir / "method_info.csv", df=method_info_full)
        save_pd.save(path=self.output_dir / "results_per_split.csv", df=df_results_rank_compare)

        # ----- add times per 1K samples -----
        # Per-dataset mean per-fold train/test sizes, derived natively from the collection's
        # splits (keyed by dataset == tabarena_task_name, matching df_results["dataset"]).
        train_sizes: dict[str, list[float]] = {}
        test_sizes: dict[str, list[float]] = {}
        for ttm in self.task_metadata:
            ds = ttm.tabarena_task_name
            for split in ttm.splits_metadata.values():
                train_sizes.setdefault(ds, []).append(split.num_instances_train)
                test_sizes.setdefault(ds, []).append(split.num_instances_test)
        dataset_to_n_samples_train = {ds: sum(v) / len(v) for ds, v in train_sizes.items()}
        dataset_to_n_samples_test = {ds: sum(v) / len(v) for ds, v in test_sizes.items()}

        df_results_rank_compare["time_train_s_per_1K"] = (
            df_results_rank_compare["time_train_s"]
            * 1000
            / df_results_rank_compare["dataset"].map(dataset_to_n_samples_train)
        )
        df_results_rank_compare["time_infer_s_per_1K"] = (
            df_results_rank_compare["time_infer_s"]
            * 1000
            / df_results_rank_compare["dataset"].map(dataset_to_n_samples_test)
        )

        if plot_times:
            self.plot_tabarena_times(df=df_results_rank_compare, output_dir=self.output_dir, show=False)

        # TODO: Move this into the `.leaderboard` call
        if "normalized-error-dataset" not in df_results_rank_compare.columns:
            df_results_rank_compare = self.compute_normalized_error_dynamic(df_results=df_results_rank_compare)
        assert "normalized-error-dataset" in df_results_rank_compare.columns, (
            "Run `self.compute_normalized_error_dynamic(df_results)` first to get normalized-error."
        )
        df_results_rank_compare["normalized-error"] = df_results_rank_compare["normalized-error-dataset"]

        if include_norm_score:
            self.plot_tuning_impact(
                df=df_results_rank_compare,
                framework_types=framework_types,
                save_prefix=f"{self.output_dir}",
                use_gmean=use_gmean,
                baselines=baselines,
                baseline_colors=baseline_colors,
                use_score=True,
                name_suffix="-normscore-dataset-horizontal",
                imputed_names=imputed_names,
                plot_tune_types=plot_tune_types,
                show=False,
                use_y=True,
                **plot_tuning_kwargs,
            )

            if plot_extra_barplots:
                self.plot_tuning_impact(
                    df=df_results_rank_compare,
                    framework_types=framework_types,
                    save_prefix=f"{self.output_dir}",
                    use_gmean=use_gmean,
                    baselines=baselines,
                    baseline_colors=baseline_colors,
                    use_score=True,
                    name_suffix="-normscore-dataset",
                    imputed_names=imputed_names,
                    plot_tune_types=plot_tune_types,
                    show=False,
                    **plot_tuning_kwargs,
                )

                self.plot_tuning_impact(
                    df=df_results_rank_compare,
                    framework_types=framework_types,
                    save_prefix=f"{self.output_dir}",
                    use_gmean=use_gmean,
                    baselines=baselines,
                    baseline_colors=baseline_colors,
                    use_score=True,
                    metric="normalized-error-task",
                    name_suffix="-normscore-task",
                    imputed_names=imputed_names,
                    plot_tune_types=plot_tune_types,
                    show=False,
                    **plot_tuning_kwargs,
                )

        elo_kwargs = dict(
            calibration_framework=calibration_framework,
            calibration_elo=1000,
            BOOTSTRAP_ROUNDS=self.elo_bootstrap_rounds,
        )

        tabarena = TabArena(
            method_col=self.method_col,
            task_col="dataset",
            seed_column="fold",
            error_col=self.error_col,
            columns_to_agg_extra=[
                "time_train_s",
                "time_infer_s",
                "time_train_s_per_1K",
                "time_infer_s_per_1K",
                "normalized-error",
                "normalized-error-task",
                "imputed",
            ],
            groupby_columns=[
                "metric",
                "problem_type",
            ],
        )

        leaderboard_kwargs.setdefault("include_elo", True)
        leaderboard_kwargs.setdefault("include_winrate", True)
        leaderboard_kwargs.setdefault("include_mrr", True)
        leaderboard_kwargs.setdefault("include_rank_counts", True)
        leaderboard_kwargs.setdefault("baseline_method", calibration_framework)
        leaderboard_kwargs.setdefault("elo_kwargs", elo_kwargs)
        leaderboard_kwargs.setdefault("average_seeds", average_seeds)

        leaderboard = tabarena.leaderboard(
            data=df_results_rank_compare,
            **leaderboard_kwargs,
        )
        leaderboard = leaderboard.join(method_info, on="method")
        leaderboard["elo"]
        leaderboard = leaderboard.reset_index(drop=False)
        save_pd.save(path=f"{self.output_dir}/tabarena_leaderboard.csv", df=leaderboard)

        self.create_leaderboard_latex(
            leaderboard,
            framework_types=framework_types,
            save_dir=self.output_dir,
            hidden_methods=plot_tuning_kwargs.get("hidden_methods"),
        )

        n_tasks = len(df_results_rank_compare[[tabarena.task_col, tabarena.seed_column]].drop_duplicates())

        if verbose:
            print(
                f"Evaluating with {len(df_results_rank_compare[tabarena.task_col].unique())} datasets... ({n_tasks} tasks)"
            )
            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
                print(leaderboard)

        # horizontal elo barplot
        self.plot_tuning_impact(
            df=df_results_rank_compare,
            df_elo=leaderboard,
            framework_types=framework_types,
            save_prefix=f"{self.output_dir}",
            use_gmean=use_gmean,
            baselines=baselines,
            baseline_colors=baseline_colors,
            name_suffix="-elo-horizontal",
            imputed_names=imputed_names,
            plot_tune_types=plot_tune_types,
            use_y=True,
            show=False,
            **plot_tuning_kwargs,
        )

        # vertical elo barplot
        self.plot_tuning_impact(
            df=df_results_rank_compare,
            df_elo=leaderboard,
            framework_types=framework_types,
            save_prefix=f"{self.output_dir}",
            use_gmean=use_gmean,
            baselines=baselines,
            baseline_colors=baseline_colors,
            name_suffix="-elo",
            imputed_names=imputed_names,
            plot_tune_types=plot_tune_types,
            show=False,
            **plot_tuning_kwargs,
        )

        results_per_task = tabarena.compute_results_per_task(data=df_results_rank_compare)
        results_per_split = tabarena.compute_results_per_task(data=df_results_rank_compare, include_seed_col=True)

        results_per_task = results_per_task.join(method_info, on="method")
        results_per_split = results_per_split.join(method_info, on="method")

        if compute_fold_stability_curves:
            fold_stability_curves = tabarena.jitter_bootstrap_curve_all_datasets(
                results_per_split,
                n_bootstrap=200,
            )
            save_pd.save(path=f"{self.output_dir}/fold_stability_curves.csv", df=fold_stability_curves)

        if compute_fold_similarity:
            fold_similarity = tabarena.rank_datasets_by_fold_similarity(
                results_per_task=results_per_split,
                **(fold_similarity_kwargs or {}),
            )
            df_jitter, _ = tabarena.jitter_all_datasets(results_per_split)
            df_jitter = df_jitter.set_index("dataset")
            fold_similarity_df = fold_similarity["dataset_ranking"]
            fold_similarity_df["jitter_mean"] = df_jitter["jitter_mean"]
            fold_similarity_df["pairwise_jitter_mean"] = df_jitter["pairwise_jitter_mean"]
            save_pd.save(path=f"{self.output_dir}/fold_similarity.csv", df=fold_similarity_df.reset_index(drop=False))

        # TODO: Consider adding the metadata to the saved `results_per_split.csv` file?
        # assert len(results_per_split) == len(df_results_rank_compare)
        # groupby_columns = tabarena._get_groupby_cols(results=results_per_split)
        # extra_cols = [c for c in df_results_rank_compare.columns if c not in results_per_split.columns]
        # results_per_split_w_metadata = results_per_split.merge(df_results_rank_compare[[*groupby_columns, *extra_cols]], on=groupby_columns)
        # assert len(results_per_split) == len(results_per_split_w_metadata)

        # FIXME: Is critical diagram incorrect?
        if plot_cdd:

            def rename_model(name: str):
                parts = name.split(" ")
                if parts[0] in f_map_type_name:
                    parts[0] = f_map_type_name[parts[0]]
                name = " ".join(parts)
                return name.replace("(tuned + ensemble)", "(T+E)")

            # use tuned+ensembled version if available, and default otherwise
            tune_methods = results_per_task[self.method_col].map(method_info["method_subtype"])
            method_types = (
                results_per_task[self.method_col]
                .map(method_info["config_type"])
                .fillna(results_per_task[self.method_col])
            )

            tuned_ens_types = method_types[tune_methods == "tuned_ensemble"]
            per_task_filter = (tune_methods == "tuned_ensemble") | (
                (tune_methods == "default") & ~method_types.isin(tuned_ens_types)
            )

            tune_methods_split = results_per_split[self.method_col].map(method_info["method_subtype"])
            method_types_split = (
                results_per_split[self.method_col]
                .map(method_info["config_type"])
                .fillna(results_per_split[self.method_col])
            )
            tuned_ens_types_split = method_types_split[tune_methods_split == "tuned_ensemble"]
            per_split_filter = (tune_methods_split == "tuned_ensemble") | (
                (tune_methods_split == "default") & ~method_types_split.isin(tuned_ens_types_split)
            )

            if plot_with_baselines:
                per_task_filter = per_task_filter | results_per_task[self.method_col].isin(baselines)
                per_split_filter = per_split_filter | results_per_split[self.method_col].isin(baselines)

            results_te_per_task = results_per_task[per_task_filter]
            results_te_per_split = results_per_split[per_split_filter]

            results_te_per_task.loc[:, self.method_col] = results_te_per_task[self.method_col].map(rename_model)
            results_te_per_split.loc[:, self.method_col] = results_te_per_split[self.method_col].map(rename_model)

            if average_seeds:
                _results_to_use_winrate_matrix = results_te_per_task.copy()
            else:
                _results_to_use_winrate_matrix = results_te_per_split.copy()

            # Drop hidden_methods from the winrate matrix. Two surfaces to
            # cover:
            #   - Configs: ``config_type`` on the raw per-task/split
            #     dataframes is the *short* framework name ("RF", "XT",
            #     "GBM"); callers supply ``hidden_methods`` in the *long*
            #     display-name form ("RandomForest", "ExtraTrees",
            #     "LightGBM"). ``f_map_type_name`` is exactly that
            #     short → long rename map, so route the column through it
            #     before filtering and fall back to the raw value whenever
            #     the map has no entry for that short name.
            #   - Baselines: ``config_type`` is typically NaN for these
            #     rows; their long display name lives in ``self.method_col``
            #     (already passed through ``rename_model`` above), so match
            #     hidden_methods against that column directly.
            hidden_methods_winrate = plot_tuning_kwargs.get("hidden_methods") or []
            if hidden_methods_winrate:
                hidden_mask = _results_to_use_winrate_matrix[self.method_col].isin(
                    hidden_methods_winrate,
                )
                if "config_type" in _results_to_use_winrate_matrix.columns:
                    mapped_names = (
                        _results_to_use_winrate_matrix["config_type"]
                        .map(f_map_type_name)
                        .fillna(_results_to_use_winrate_matrix["config_type"])
                    )
                    hidden_mask = hidden_mask | mapped_names.isin(hidden_methods_winrate)
                _results_to_use_winrate_matrix = _results_to_use_winrate_matrix.loc[~hidden_mask]

            if len(_results_to_use_winrate_matrix) != 0:
                winrate_matrix = tabarena.compute_winrate_matrix(results_per_task=_results_to_use_winrate_matrix)
                # Display-only rename of winrate-matrix row/column labels.
                # Applied after `compute_winrate_matrix` so the rename only
                # touches the axis tick labels — pairwise comparisons are
                # unaffected. Unknown keys are ignored.
                if winrate_method_rename:
                    winrate_matrix = winrate_matrix.rename(
                        index=winrate_method_rename,
                        columns=winrate_method_rename,
                    )
                # Build the suptitle in the same "TabArena-<subset> …"
                # style used by the tuning-trajectory and tuning-impact bar
                # plots so the three surfaces read as one report. ``None``
                # and ``"all"`` collapse to the unsuffixed ``TabArena``
                # prefix (aggregate / no-meaningful-subset case).
                winrate_title: str | None = None
                if show_winrate_title:
                    if subset_label and subset_label != "all":
                        winrate_title = f"TabArena-{subset_label} Win-rate Matrix"
                    else:
                        winrate_title = "TabArena Win-rate Matrix"
                try:
                    tabarena.plot_winrate_matrix(
                        winrate_matrix=winrate_matrix,
                        save_path=str(Path(self.output_dir / f"winrate_matrix.{self.figure_file_type}")),
                        title=winrate_title,
                    )
                except (RuntimeError, ValueError) as e:
                    print(
                        f"Warning: Error encountered during winrate matrix plotting. {e}"
                        "This likely means the CLI does not have access to the correct Chromium version...",
                    )

            # Off by default while the FIXME above (diagram possibly incorrect) stands.
            if plot_critical_diagrams and len(results_te_per_task) != 0:
                try:
                    tabarena.plot_critical_diagrams(
                        results_per_task=results_te_per_task,
                        save_path=f"{self.output_dir}/figures/critical-diagram.{self.figure_file_type}",
                        show=False,
                    )
                except ValueError:
                    print(
                        "Warning: ValueError encountered during critical diagram plotting. "
                        "This likely means there is too little data to compute critical diagrams. Skipping ...",
                    )

        if plot_runtimes:
            self.generate_runtime_plot(df_results=df_results_rank_compare)

        if plot_pareto and (framework_types or plot_with_baselines):
            self.plot_pareto(
                leaderboard=leaderboard,
                framework_types=framework_types,
                with_baselines=plot_with_baselines,
                plot_tuning_kwargs=plot_tuning_kwargs,
            )

        return leaderboard

    def assert_no_duplicates(self, df_results: pd.DataFrame):
        # don't allow duplicate results
        dupes = df_results[
            df_results.duplicated(
                subset=["dataset", "fold", self.method_col],
                keep=False,
            )
        ]
        if not dupes.empty:
            dupes = dupes.sort_values(by=[self.method_col, "dataset", "fold"])
            duplicated_methods = dupes["method"].value_counts()
            raise ValueError(
                "Duplicate rows detected on keys [dataset, fold, "
                f"{self.method_col}].\n"
                f"The following {len(duplicated_methods)} methods were duplicated (w/ counts):\n"
                f"{duplicated_methods.to_string()}\n"
                f"The following {len(dupes)} rows are duplicates:\n"
                f"{dupes.to_string(index=False)}",
            )

    def assert_no_nan_methods(self, df_results: pd.DataFrame):
        if df_results[self.method_col].isna().any():
            missing_count = df_results[self.method_col].isna().sum()
            missing_percent = missing_count / len(df_results)
            raise AssertionError(
                f"Found NaN values in '{self.method_col}' column: "
                f"{missing_count}/{len(df_results)} ({missing_percent * 100:.1f}%) were NaN.",
            )

    def filter_results(self, df_results: pd.DataFrame):
        if self.datasets is not None:
            df_results = df_results[df_results["dataset"].isin(self.datasets)]
        if self.folds is not None:
            df_results = df_results[df_results["fold"].isin(self.folds)]
        if self.methods is not None:
            df_results = df_results[df_results[self.method_col].isin(self.methods)]
        if self.problem_types is not None:
            df_results = df_results[df_results["problem_type"].isin(self.problem_types)]
        if not self.keep_best:
            df_results = df_results[df_results["method_subtype"] != "best"]
        if self.banned_model_types:
            df_results = df_results[~df_results["config_type"].isin(self.banned_model_types)]
        return df_results

    def get_method_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Verify that each method has exactly one unique value for method_type,
        method_subtype, config_type, ta_name, and ta_suite. Return a mapping dataframe.

        Raises:
        ------
        ValueError
            If any method has non-unique values in any of the checked columns.

        Returns:
        -------
        pd.DataFrame
            A dataframe indexed by "method" with
            the unique values for each of method_type, method_subtype, config_type, ta_name, ta_suite.
        """
        group_cols = self.method_col
        value_cols = ["method_type", "method_subtype", "config_type", "ta_name", "ta_suite"]

        grouped = df.groupby(group_cols)[value_cols]

        # Compute sets of unique values per group
        unique_vals = grouped.nunique(dropna=False)

        # Identify problematic groups
        bad = (unique_vals > 1).any(axis=1)
        if bad.any():
            msg = (
                "Found groups with multiple unique values in one or more columns.\n"
                f"Columns checked: {value_cols}\n\n"
                "Offending groups:\n"
                f"{unique_vals[bad]}"
            )
            raise ValueError(msg)

        # Safe: each value is unique → extract the scalar values
        return grouped.first()  # identical to .agg("first") but faster

    def plot_pareto(
        self,
        leaderboard: pd.DataFrame,
        framework_types: list[str],
        with_baselines: bool = True,
        plot_tuning_kwargs: dict | None = None,
    ):
        _f_map, f_map_type, f_map_inverse, f_map_type_name = self.get_framework_type_method_names(
            framework_types=framework_types,
        )
        leaderboard_pareto = leaderboard.copy()
        leaderboard_pareto["Method"] = (
            leaderboard_pareto[self.method_col].map(f_map_type).fillna(leaderboard_pareto[self.method_col])
        )

        if self.banned_pareto_methods:
            leaderboard_pareto = leaderboard_pareto[~leaderboard_pareto["Method"].isin(self.banned_pareto_methods)]

        leaderboard_pareto["Type"] = leaderboard_pareto[self.method_col].map(f_map_inverse).fillna("baseline")
        leaderboard_pareto["Method"] = (
            leaderboard_pareto["Method"].map(f_map_type_name).fillna(leaderboard_pareto["Method"])
        )
        f_map_suffix = get_f_map_suffix_plots()
        leaderboard_pareto["suffix"] = leaderboard_pareto["Type"].map(f_map_suffix).fillna("")
        method_order = None
        plot_pareto_kwargs = {}
        if plot_tuning_kwargs is not None:
            if "hidden_methods" in plot_tuning_kwargs:
                leaderboard_pareto = leaderboard_pareto[
                    ~leaderboard_pareto["Method"].isin(plot_tuning_kwargs["hidden_methods"])
                ]
            if "pareto_order" in plot_tuning_kwargs:
                method_order = plot_tuning_kwargs["pareto_order"]
            if "method_style_map" in plot_tuning_kwargs:
                method_style_map = plot_tuning_kwargs["method_style_map"]
                display_name_map = {m: v["display_name"] for m, v in method_style_map.items() if "display_name" in v}
                leaderboard_pareto["Method"] = (
                    leaderboard_pareto["Method"].map(display_name_map).fillna(leaderboard_pareto["Method"])
                )
                if method_order is not None:
                    method_order = [display_name_map.get(m, m) for m in method_order]
            if "title" in plot_tuning_kwargs:
                plot_pareto_kwargs["title"] = plot_tuning_kwargs["title"]

        leaderboard_pareto[self.method_col] = leaderboard_pareto["Method"] + leaderboard_pareto["suffix"]
        fig_rename_dict = {
            "baseline": "Baseline",
            "default": "Default",
            "tuned": "Tuned",
            "tuned_ensembled": "Tuned + Ens.",
            "best": "Best",
            "holdout": "Default, Holdout",
            "holdout_tuned": "Tuned, Holdout",
            "holdout_tuned_ensembled": "Tuned + Ens., Holdout",
        }
        leaderboard_pareto["Type"] = (
            leaderboard_pareto["Type"]
            .map(fig_rename_dict)
            .fillna(
                leaderboard_pareto["Type"],
            )
        )

        if not with_baselines:
            leaderboard_pareto = leaderboard_pareto[leaderboard_pareto["Type"] != "Baseline"]

        self.plot_pareto_elo_vs_time_infer(
            leaderboard=leaderboard_pareto, method_order=method_order, **plot_pareto_kwargs
        )
        self.plot_pareto_elo_vs_time_train(
            leaderboard=leaderboard_pareto, method_order=method_order, **plot_pareto_kwargs
        )
        self.plot_pareto_improvability_vs_time_infer(
            leaderboard=leaderboard_pareto, method_order=method_order, **plot_pareto_kwargs
        )
        self.plot_pareto_improvability_vs_time_train(
            leaderboard=leaderboard_pareto, method_order=method_order, **plot_pareto_kwargs
        )

    def plot_pareto_elo_vs_time_train(
        self,
        leaderboard: pd.DataFrame,
        method_order: list[str] | None = None,
        title: str | None = "auto",
    ):
        save_prefix = Path(self.output_dir)
        save_path = str(save_prefix / f"pareto_front_elo_vs_time_train.{self.figure_file_type}")
        y_name = "Elo"
        x_name = "Train time per 1K samples (s) (median)"
        if title == "auto":
            title = "Elo vs Train Time"

        data = leaderboard.copy()
        data[x_name] = data["median_time_train_s_per_1K"]
        data[y_name] = data["elo"]

        _plot_pareto(
            data=data,
            x_name=x_name,
            y_name=y_name,
            max_X=False,
            max_Y=True,
            sort_y=True,
            hue="Method",  # color by family
            style_col="Type",  # marker by run type
            style_order=self.style_order,
            style_markers=self.style_markers,
            label_col=self.method_col,  # annotate with full method name
            title=title,
            save_path=save_path,
            show=False,
            aspect=4 / 3,
            legend_first=method_order,
        )

    def plot_pareto_elo_vs_time_infer(
        self,
        leaderboard: pd.DataFrame,
        method_order: list[str] | None = None,
        title: str | None = "auto",
    ):
        save_prefix = Path(self.output_dir)
        save_path = str(save_prefix / f"pareto_front_elo_vs_time_infer.{self.figure_file_type}")
        y_name = "Elo"
        x_name = "Inference time per 1K samples (s) (median)"
        if title == "auto":
            title = "Elo vs Inference Time"

        data = leaderboard.copy()
        data[x_name] = data["median_time_infer_s_per_1K"]
        data[y_name] = data["elo"]

        _plot_pareto(
            data=data,
            x_name=x_name,
            y_name=y_name,
            max_X=False,
            max_Y=True,
            sort_y=True,
            hue="Method",  # <-- same color for same method_type
            style_col="Type",  # <-- different marker per run_type
            style_order=self.style_order,
            style_markers=self.style_markers,
            label_col=self.method_col,  # <-- annotate with full method name
            title=title,
            save_path=save_path,
            show=False,
            aspect=4 / 3,
            legend_first=method_order,
        )

    def plot_pareto_improvability_vs_time_infer(
        self,
        leaderboard: pd.DataFrame,
        method_order: list[str] | None = None,
        title: str | None = "auto",
    ):
        save_prefix = Path(self.output_dir)
        save_path = str(save_prefix / f"pareto_front_improvability_vs_time_infer.{self.figure_file_type}")
        y_name = "Improvability (%)"
        x_name = "Inference time per 1K samples (s) (median)"
        if title == "auto":
            title = "Improvability vs Inference Time"

        data = leaderboard.copy()
        data[x_name] = data["median_time_infer_s_per_1K"]
        data[y_name] = data["improvability"] * 100

        _plot_pareto(
            data=data,
            x_name=x_name,
            y_name=y_name,
            max_X=False,
            max_Y=False,
            sort_y=True,
            hue="Method",  # color by family
            style_col="Type",  # marker by run type
            style_order=self.style_order,
            style_markers=self.style_markers,
            label_col=self.method_col,  # annotate with full method name
            # ylim=(0, None),
            title=title,
            save_path=save_path,
            show=False,
            aspect=4 / 3,
            legend_first=method_order,
        )

    def plot_pareto_improvability_vs_time_train(
        self,
        leaderboard: pd.DataFrame,
        method_order: list[str] | None = None,
        title: str | None = "auto",
    ):
        save_prefix = Path(self.output_dir)
        save_path = str(save_prefix / f"pareto_front_improvability_vs_time_train.{self.figure_file_type}")
        y_name = "Improvability (%)"
        x_name = "Train time per 1K samples (s) (median)"
        if title == "auto":
            title = "Improvability vs Train Time"

        data = leaderboard.copy()
        data[x_name] = data["median_time_train_s_per_1K"]
        data[y_name] = data["improvability"] * 100

        _plot_pareto(
            data=data,
            x_name=x_name,
            y_name=y_name,
            max_X=False,
            max_Y=False,
            sort_y=True,
            hue="Method",  # color by family
            style_col="Type",  # marker by run type
            style_order=self.style_order,
            style_markers=self.style_markers,
            label_col=self.method_col,  # annotate with full method name
            # ylim=(0, None),
            title=title,
            save_path=save_path,
            show=False,
            aspect=4 / 3,
            legend_first=method_order,
        )

    def get_method_rename_map(self) -> dict[str, str]:
        method_rename_map = get_method_rename_map()  # FIXME: Avoid hardcoding
        method_rename_map.update(self._method_rename_map)
        return method_rename_map

    def get_imputed_names(self, df_results: pd.DataFrame) -> list[str]:
        # Handle imputation of names
        imputed_names = list(df_results[self.method_col][df_results["imputed"] > 0].unique())
        if len(imputed_names) == 0:
            return []

        method_rename_map = self.get_method_rename_map()

        # remove suffix
        imputed_names = [n.split(" (")[0] for n in imputed_names]
        imputed_names = [method_rename_map.get(n, n) for n in imputed_names]
        return list(set(imputed_names))

    def plot_portfolio_ensemble_weights_barplot(self, df_ensemble_weights: pd.DataFrame):
        from pathlib import Path

        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        _fig, ax = plt.subplots(
            1,
            1,
            figsize=(3.5, 3),
        )

        df_ensemble_weights = df_ensemble_weights.copy(deep=True)
        _method_rename_map = self.get_method_rename_map()
        columns_new = [_method_rename_map.get(c, c) for c in df_ensemble_weights.columns]
        df_ensemble_weights.columns = columns_new

        df_long = df_ensemble_weights.melt(var_name="Model", value_name="Weight")
        model_order = list(df_ensemble_weights.columns)

        pastel_palette = sns.color_palette("pastel")
        deep_palette = sns.color_palette("deep")

        # Define gradient from pastel and deep separately
        pastel_start = mcolors.to_rgb(pastel_palette[2])
        pastel_end = mcolors.to_rgb(pastel_palette[0])
        deep_start = mcolors.to_rgb(deep_palette[2])
        deep_end = mcolors.to_rgb(deep_palette[0])

        # Create pastel gradient for bars
        bar_colors = [mcolors.to_hex(c) for c in np.linspace(pastel_start, pastel_end, len(model_order))]

        # Create deep gradient for error bars
        error_colors = [mcolors.to_hex(c) for c in np.linspace(deep_start, deep_end, len(model_order))]

        # Alphas for bars
        alphas = np.linspace(1.0, 1.0, len(model_order))[::-1]  # Keep at 1.0 for now

        # Create barplot
        barplot = sns.barplot(
            data=df_long,
            x="Weight",
            y="Model",
            hue="Model",
            legend=False,
            ax=ax,
            order=model_order,
            palette=bar_colors,
        )

        # Apply alpha to bar colors
        for patch, alpha in zip(barplot.patches, alphas, strict=False):
            r, g, b = patch.get_facecolor()[:3]
            patch.set_facecolor((r, g, b, alpha))

        # Update error bar colors manually
        for i, line in enumerate(ax.lines):
            # Seaborn/matplotlib adds error bar lines in a certain order.
            # Each bar usually has 2 lines: one vertical bar and one cap on top.
            # Here, we assume two lines per error bar, so divide i by 2.
            color_index = i // 2
            if color_index < len(error_colors):
                line.set_color(error_colors[color_index])

        barplot.set_xlabel("Average weight in TabArena ensemble")
        barplot.set_ylabel("")

        fig_name = f"portfolio-weight-barplot.{self.figure_file_type}"
        fig_prefix = Path(self.output_dir) / "figures"
        fig_prefix.mkdir(parents=True, exist_ok=True)

        fig_save_path = fig_prefix / fig_name
        plt.savefig(fig_save_path)

    def create_leaderboard_latex(
        self,
        df: pd.DataFrame,
        framework_types,
        save_dir,
        hidden_methods: list[str] | None = None,
    ):
        df = df.copy(deep=True)
        _f_map, _f_map_type, _f_map_inverse, f_map_type_name = self.get_framework_type_method_names(
            framework_types=framework_types,
        )

        # Drop hidden methods. ``df["config_type"]`` is the *short* framework
        # name ("RF", "XT", "GBM"); ``hidden_methods`` is supplied in the
        # *long* display form ("RandomForest", "ExtraTrees", "LightGBM").
        # ``f_map_type_name`` is exactly that short → long rename map, so
        # route the column through it before filtering — same pattern used
        # by the winrate-matrix filter elsewhere in this file.
        if hidden_methods and "config_type" in df.columns:
            mapped_names = df["config_type"].map(f_map_type_name).fillna(df["config_type"])
            df = df.loc[~mapped_names.isin(hidden_methods)]

        def rename_model(name: str):
            parts = name.split(" ")
            if parts[0] in f_map_type_name:
                parts[0] = f_map_type_name[parts[0]]
            name = " ".join(parts)
            name = name.replace("(default)", "(D)")
            name = name.replace("(tuned)", "(T)")
            return name.replace("(tuned + ensemble)", "(T+E)")

        df = df.sort_values(by="elo", ascending=False)

        df_new = pd.DataFrame()

        df_new[r"Model"] = df[self.method_col].map(rename_model)
        # do the more annoying way {}_{...} so that \textbf{} affects the main number
        df_new[r"Elo ($\uparrow$)"] = [
            f"{round(elo)}" + r"${}_{" + f"-{math.ceil(elom)},+{math.ceil(elop)}" + r"}$"
            for elo, elom, elop in zip(df["elo"], df["elo-"], df["elo+"], strict=False)
        ]
        df_new[r"\#wins ($\uparrow$)"] = [f"{cnt:.1f}" for cnt in df["rank=1_count"]]
        df_new["Improva-\n" + r"bility ($\downarrow$)"] = [f"{100 * val:.1f}\\%" for val in df["improvability"]]
        df_new[r"Train time" + "\n" + r"per 1K [s]"] = [f"{t:.2f}" for t in df["median_time_train_s_per_1K"]]
        df_new[r"Predict time" + "\n" + r"per 1K [s]"] = [f"{t:.2f}" for t in df["median_time_infer_s_per_1K"]]

        # ----- highlight best and second-best numbers per column -----

        # first, convert the strings back to floats
        def extract_first_float(s):
            """Extracts the first sequence of digits (including decimal point) from the input string
            and returns it as a float. Returns None if no valid number is found.
            """
            match = re.search(r"\d+(\.\d+)?", s)
            if match:
                return float(match.group())
            return None

        def find_smallest_and_second_smallest_indices(numbers):
            if len(numbers) < 2:
                return [], []

            # Find the smallest value
            min_val = min(numbers)
            min_indices = [i for i, x in enumerate(numbers) if x == min_val]

            # Exclude the smallest values and find the second smallest
            remaining = [x for x in numbers if x != min_val]
            if not remaining:
                return min_indices, []  # No second smallest

            second_min_val = min(remaining)
            second_min_indices = [i for i, x in enumerate(numbers) if x == second_min_val]

            return min_indices, second_min_indices

        # then, add textbf or underline to the correct rows
        for _col_idx, col in enumerate(df_new.columns):
            if r"\uparrow" in col or r"\downarrow" in col:
                # factor = 1 if r'\downarrow' in col else -1
                # numbers = [factor * extract_first_float(s) for s in df_new[col]]
                ranks = df_new[col].map(extract_first_float).rank(method="min", ascending=r"\downarrow" in col)
                for rank, color in [(1, "gold"), (2, "silver"), (3, "bronze")]:
                    df_new.loc[ranks == rank, col] = df_new.loc[ranks == rank, col].apply(
                        lambda x: f"\\textcolor{{{color}}}{{\\textbf{{{x}}}}}",  # noqa: B023
                    )

                # min_indices, second_min_indices = find_smallest_and_second_smallest_indices(numbers)
                # for idx in min_indices:
                #     df_new.iloc[idx, col_idx] = r'\textbf{' + df_new.iloc[idx, col_idx] + r'}'
                # for idx in second_min_indices:
                #     df_new.iloc[idx, col_idx] = r'\underline{' + df_new.iloc[idx, col_idx] + r'}'

        # ----- create latex table -----

        rows = []
        rows.append(r"\begin{tabular}{" + "llccrr" + r"}")
        rows.append(r"\toprule")
        # rows.append(' & '.join(df_new.columns) + r' \\')

        col_names_split = [col.split("\n") for col in df_new.columns]
        n_rows_header = max([len(rows) for rows in col_names_split])
        for row_idx in range(n_rows_header):
            rows.append(
                " & ".join([r"\textbf{" + lst[row_idx] + r"}" if row_idx < len(lst) else "" for lst in col_names_split])
                + r" \\"
            )
        rows.append(r"\midrule")

        for _row_index, row in df_new.iterrows():
            rows.append(" & ".join([row[col_name] for col_name in df_new.columns]) + r" \\")

        rows.append(r"\bottomrule")
        rows.append(r"\end{tabular}")

        table = "\n".join(rows)

        with open(Path(save_dir) / "leaderboard.tex", "w") as f:
            f.write(table)

    # FIXME: Avoid hardcoding
    def plot_tuning_impact(
        self,
        df: pd.DataFrame,
        framework_types: list,
        save_prefix: str,
        baselines: list[str] | None = None,
        baseline_colors: list[str] | None = None,
        show: bool = False,
        use_gmean=False,
        use_score: bool = True,
        df_elo: pd.DataFrame = None,
        name_suffix: str | None = None,
        imputed_names: list[str] | None = None,
        use_y: bool = False,
        metric: str = "normalized-error",
        plot_tune_types: list[str] | None = None,
        method_style_map: dict[str, MethodLabelStyle] | None = None,
        default_method_style: MethodLabelStyle | None = None,
        hidden_methods: list[str] | None = None,
        baseline_text_y_gap: float = 1.0,
        figsize: tuple[int, int] | None = None,
        figheight: float | None = None,
        figheight_horizontal: float | None = None,
        bar_width: float | None = None,
        title: str | None = None,
        tune_method_overrides: list[TuneMethodOverride] | None = None,
    ):
        import matplotlib.patheffects as PathEffects
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Patch

        if method_style_map is None:
            method_style_map = {}
        if default_method_style is None:
            default_method_style = {}
        same_width = use_y
        use_lim = True
        use_elo = df_elo is not None
        lower_is_better = True
        lim = None
        xlim = None
        ylim = None

        if imputed_names is None:
            imputed_names = []

        df = df.copy(deep=True)

        framework_col = "framework_type"

        groupby_columns_extra = ["dataset"]

        if use_elo:
            metric = "elo"
            use_lim = True
            lim = [self.elo_ymin, None]
            lower_is_better = False
            df = df_elo.copy(deep=True)
            df = df[[self.method_col, "elo", "elo+", "elo-"]]
            groupby_columns_extra = []
        elif use_score:
            lower_is_better = False
            df["normalized-score"] = 1 - df[metric]
            metric = "normalized-score"
        else:
            pass  # keep the caller-provided metric

        if tune_method_overrides is None:
            tune_method_overrides = []

        # Phase 1: apply rename_map and promote-from-baselines BEFORE building f_map_*,
        # so f_map_inverse sees the post-rename method names.
        for ov in tune_method_overrides:
            if ov.rename_map:
                df[self.method_col] = df[self.method_col].map(ov.rename_map).fillna(df[self.method_col])
                framework_types = [ov.rename_map.get(f, f) for f in framework_types]

        if baselines is None:
            baselines = []
        if baseline_colors is not None:
            assert len(baselines) == len(baseline_colors), (
                "A color must be specified for each baseline via the `baseline_colors` argument."
            )

        for ov in tune_method_overrides:
            if not ov.promote_from_baselines:
                continue
            for m in ov.methods:
                if m in baselines:
                    idx = baselines.index(m)
                    baselines = baselines[:idx] + baselines[idx + 1 :]
                    if baseline_colors is not None:
                        baseline_colors = baseline_colors[:idx] + baseline_colors[idx + 1 :]
                    if m not in framework_types:
                        framework_types.append(m)

        _f_map, f_map_type, f_map_inverse, f_map_type_name = self.get_framework_type_method_names(
            framework_types=framework_types,
        )

        df = df.copy()
        df.loc[:, "framework_type"] = df[self.method_col].map(f_map_type).fillna(df[self.method_col])
        df.loc[:, "tune_method"] = df[self.method_col].map(f_map_inverse).fillna("default")

        # Phase 2: retag tune_method for override-matched rows, and pin framework_type for
        # promoted rows whose f_map_type lookup wouldn't naturally produce the right value.
        for ov in tune_method_overrides:
            mask = df[self.method_col].isin(ov.methods)
            df.loc[mask, "tune_method"] = ov.tune_method
            if ov.promote_from_baselines:
                df.loc[mask, "framework_type"] = df.loc[mask, self.method_col]

        tick_methods = framework_types
        has_non_baselines = len(tick_methods) != 0

        df["framework_type"] = df["framework_type"].map(f_map_type_name).fillna(df["framework_type"])

        baselines = [f_map_type_name.get(m, m) for m in baselines]
        tick_methods = [f_map_type_name.get(m, m) for m in tick_methods]
        if hidden_methods:
            hidden_methods = [f_map_type_name.get(m, m) for m in hidden_methods]
            baselines = [m for m in baselines if m not in hidden_methods]
            tick_methods = [m for m in tick_methods if m not in hidden_methods]
        framework_types = baselines + tick_methods

        if plot_tune_types:
            df = df[df["tune_method"].isin(plot_tune_types) | df[self.method_col].isin(baselines)]

        df_plot = df[df["framework_type"].isin(framework_types)]

        df_plot_w_mean_per_dataset = (
            df_plot.groupby(["framework_type", "tune_method", *groupby_columns_extra])[metric].mean().reset_index()
        )

        if use_gmean:
            # FIXME: Doesn't plot correctly, need to figure out error bars for geometric mean
            df_plot_eps = df_plot.copy(deep=True)
            df_plot_eps[metric] += 0.01
            from scipy.stats import gmean

            df_plot_w_gmean_per_dataset = (
                df_plot.groupby(["framework_type", "tune_method", *groupby_columns_extra])[metric]
                .apply(gmean)
                .reset_index()
            )
            df_plot_w_mean_per_dataset = df_plot_w_gmean_per_dataset

        df_plot_w_mean_2 = (
            df_plot_w_mean_per_dataset.groupby(["framework_type", "tune_method"])[metric].mean().reset_index()
        )

        df_plot_w_mean_2 = df_plot_w_mean_2.sort_values(by=metric, ascending=lower_is_better)
        baseline_means = {}
        for baseline in baselines:
            baseline_means[baseline] = df_plot_w_mean_2[df_plot_w_mean_2["framework_type"] == baseline][metric].iloc[0]

        df_plot_w_mean_2 = df_plot_w_mean_2[~df_plot_w_mean_2["framework_type"].isin(baselines)]

        df_plot_mean_dedupe = df_plot_w_mean_2.drop_duplicates(subset=["framework_type"], keep="first")

        framework_type_order_orig = list(df_plot_mean_dedupe["framework_type"].to_list())
        framework_type_order_orig.reverse()

        framework_type_order = copy.deepcopy(framework_type_order_orig)

        display_name_map = {
            k: v["display_name"]
            for k, v in method_style_map.items()
            if isinstance(v, dict) and "display_name" in v and k in tick_methods
        }
        display_name_inverse_map = {v: k for k, v in display_name_map.items()}
        if display_name_map:
            df_plot = df_plot.copy()
            df_plot_mean_dedupe = df_plot_mean_dedupe.copy()
            df_plot_w_mean_per_dataset = df_plot_w_mean_per_dataset.copy()
            df_plot["framework_type"] = df_plot["framework_type"].replace(display_name_map)
            df_plot_mean_dedupe["framework_type"] = df_plot_mean_dedupe["framework_type"].replace(display_name_map)
            df_plot_w_mean_per_dataset["framework_type"] = df_plot_w_mean_per_dataset["framework_type"].replace(
                display_name_map
            )

        if display_name_map:
            framework_type_order = [display_name_map.get(m, m) for m in framework_type_order_orig]

        with sns.axes_style("whitegrid"):
            with plt.rc_context(self.rc_context_params):
                colors = sns.color_palette("pastel").as_hex()
                errcolors = sns.color_palette("deep").as_hex()

                if use_lim and not lim:
                    lim = [0, None]
                if use_y:
                    pos = metric
                    y = framework_col
                    if figsize is None:
                        figsize = (4, figheight_horizontal if figheight_horizontal is not None else 5)
                    xlim = lim

                    framework_type_order.reverse()
                    framework_type_order_orig.reverse()

                else:
                    pos = framework_col
                    y = metric
                    ylim = lim
                    if figsize is None:
                        # Floor the width: with few framework types (e.g. a small demo run),
                        # 0.5"/type leaves no room for the legend above the axes and
                        # constrained_layout collapses the axes to zero size.
                        figsize = (max(3.5, 0.5 * len(framework_types)), figheight if figheight is not None else 2.7)

                fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

                baseline_func = ax.axvline if use_y else ax.axhline

                linewidth = 0.0 if use_y else 0.3
                err_linewidth = 1.6
                err_linewidths = {
                    "tuned_ensembled": err_linewidth,
                    "tuned": err_linewidth * 0.8,
                    "default": err_linewidth * 0.6,
                    "holdout_tuned_ensembled": err_linewidth * 0.6,
                }
                # Ensure every override has an entry in err_linewidths so the
                # elo err-bar loop below can look it up.
                for ov in tune_method_overrides:
                    err_linewidths.setdefault(
                        ov.tune_method,
                        err_linewidths.get(ov.err_linewidth_key, err_linewidth),
                    )
                err_alpha = 0.6

                # Generated bar dicts, one per *unique* `tune_method` value across
                # all overrides. Two overrides sharing a tune_method (e.g. several
                # methods grouped under "api") share a single bar dict and inherit
                # styling from the first override that declares that tune_method.
                override_bars = []
                seen_tune_methods: set[str] = set()
                for ov in tune_method_overrides:
                    if ov.tune_method in seen_tune_methods:
                        continue
                    seen_tune_methods.add(ov.tune_method)
                    override_bars.append(
                        dict(
                            x=pos,
                            y=y,
                            label=ov.bar_label,
                            data=df_plot_w_mean_per_dataset[
                                df_plot_w_mean_per_dataset["tune_method"] == ov.tune_method
                            ],
                            ax=ax,
                            order=framework_type_order,
                            color=ov.bar_color,
                            width=ov.bar_width,
                            linewidth=linewidth,
                            err_kws={
                                "color": ov.err_color if ov.err_color is not None else errcolors[0],
                                "linewidth": err_linewidths[ov.tune_method],
                                "alpha": err_alpha,
                            },
                        )
                    )
                to_plot = [
                    *override_bars,
                    dict(
                        x=pos,
                        y=y,
                        label="Tuned + Ensembled",
                        data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned_ensembled"],
                        ax=ax,
                        order=framework_type_order,
                        color=colors[2],
                        width=0.6,
                        linewidth=linewidth,
                        err_kws={
                            "color": errcolors[2],
                            "linewidth": err_linewidths["tuned_ensembled"],
                            "alpha": err_alpha,
                        },
                    ),
                    # dict(
                    #     x=x, y=y,
                    #     label="Default (Holdout)",
                    #     data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "holdout"], ax=ax,
                    #     order=framework_type_order,
                    #     color=colors[4],
                    #     width=0.7, linewidth=linewidth,
                    #     err_kws={"color": errcolors[4]},
                    # ),
                    # dict(
                    #     x=x, y=y,
                    #     label="Tuned (Holdout)",
                    #     data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "holdout_tuned"], ax=ax,
                    #     order=framework_type_order,
                    #     color=colors[5],
                    #     width=0.65, linewidth=linewidth,
                    #     err_kws={"color": errcolors[5]},
                    # ),
                    dict(
                        x=pos,
                        y=y,
                        label="Tuned",
                        data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned"],
                        ax=ax,
                        order=framework_type_order,
                        color=colors[1],
                        width=0.5,
                        linewidth=linewidth,
                        err_kws={"color": errcolors[1], "linewidth": err_linewidths["tuned"], "alpha": err_alpha},
                    ),
                    dict(
                        x=pos,
                        y=y,
                        label="Default",
                        data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "default"],
                        ax=ax,
                        order=framework_type_order,
                        color=colors[0],
                        width=0.4,
                        linewidth=linewidth,
                        err_kws={"color": errcolors[0], "linewidth": err_linewidths["default"], "alpha": err_alpha},
                        alpha=1.0,
                    ),
                    dict(
                        x=pos,
                        y=y,
                        label="Tuned + Ensembled (Holdout)",
                        data=df_plot_w_mean_per_dataset[
                            df_plot_w_mean_per_dataset["tune_method"] == "holdout_tuned_ensembled"
                        ],
                        ax=ax,
                        order=framework_type_order,
                        color=colors[3],
                        width=0.3,
                        linewidth=linewidth,
                        err_kws={
                            "color": errcolors[3],
                            "linewidth": err_linewidths["holdout_tuned_ensembled"],
                            "alpha": err_alpha,
                        },
                    ),
                    # dict(
                    #     x=x, y=y,
                    #     label="Best",
                    #     data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "best"], ax=ax,
                    #     order=framework_type_order, color=colors[3],
                    #     width=0.55, linewidth=linewidth,
                    #     err_kws={"color": errcolors[3]},
                    #     alpha=1.0,
                    # ),
                ]

                if use_score:
                    widths = [plot_line["width"] for plot_line in to_plot]
                    colors = [plot_line["color"] for plot_line in to_plot]
                    err_kws_lst = [plot_line["err_kws"] for plot_line in to_plot]

                    # to_plot.reverse()
                    same_width_value = bar_width if bar_width is not None else 0.6 * 1.3
                    for plot_line, width, color, _err_kws in zip(to_plot, widths, colors, err_kws_lst, strict=False):
                        if same_width:
                            plot_line["width"] = same_width_value
                        else:
                            plot_line["width"] = width * 1.3

                for plot_line in to_plot:
                    boxplot = sns.barplot(**plot_line)

                if use_y:
                    boxplot.set(xlabel="Elo" if metric == "elo" else "Normalized score", ylabel=None)
                else:
                    boxplot.set(
                        xlabel=None, ylabel="Elo" if metric == "elo" else "Normalized score"
                    )  # remove method in the x-axis
                if title is not None:
                    # ``set_title`` places text just above the axes spine,
                    # which is where the legend sits (`bbox_to_anchor=[..., 1.01]`,
                    # `loc="lower center"` below). Use ``suptitle`` so the
                    # title floats above the figure as a whole and
                    # ``constrained_layout`` reserves room for it.
                    fig.suptitle(title)

                # do this before setting x/y limits
                for baseline_idx, (baseline, color) in enumerate(zip(baselines, baseline_colors, strict=False)):
                    baseline_mean = baseline_means[baseline]

                    style_raw = method_style_map.get(baseline)
                    style = _normalize_style(style_raw)
                    baseline_label = style.get("display_name", baseline)

                    # color default fallback remains baseline_colors if style doesn't specify color
                    color_final = style.get("color", color)
                    alpha_final = style.get("alpha", 1.0)

                    # line kwargs
                    line_kwargs = dict(
                        color=color_final,
                        alpha=alpha_final,
                        linewidth=style.get("line_width", 2.0),
                        ls=style.get("line_ls", "--"),
                        zorder=style.get("line_zorder", -10),
                    )
                    baseline_func(baseline_mean, **line_kwargs)

                    # text kwargs (reuse text_* or plain text keys)
                    text_kwargs = dict(
                        color=style.get("text_color", color_final),
                        alpha=style.get("text_alpha", alpha_final),
                    )
                    fontsize = style.get("text_fontsize", style.get("fontsize"))
                    if fontsize is not None:
                        text_kwargs["fontsize"] = fontsize

                    for k_src, k_dst in [
                        ("fontweight", "fontweight"),
                        ("fontstyle", "fontstyle"),
                        ("fontsize", "fontsize"),
                    ]:
                        v = style.get(f"text_{k_src}", style.get(k_src))
                        if v is not None:
                            text_kwargs[k_dst] = v

                    # drop None values to avoid overriding matplotlib defaults
                    text_kwargs = {k: v for k, v in text_kwargs.items() if v is not None}

                    # Match the method-label (tick label) size by default so the
                    # baseline annotation doesn't render larger than the axis
                    # ticks.  Without this, baseline text picks up
                    # rcParams["font.size"] (body text), which is typically
                    # larger than rcParams["{x,y}tick.labelsize"].
                    text_kwargs.setdefault(
                        "fontsize",
                        plt.rcParams["ytick.labelsize" if use_y else "xtick.labelsize"],
                    )

                    if use_y:
                        txt = ax.text(
                            y=(1 - 0.035 * (1 + 2 * baseline_text_y_gap * (len(baselines) - 1 - baseline_idx)))
                            * ax.get_ylim()[0],
                            x=baseline_mean * 0.99,
                            s=baseline_label,
                            ha="right",
                            **text_kwargs,
                        )
                    else:
                        # va="top" anchors the top of the text at y, so y=baseline_mean
                        # places the label directly below the dashed line.  The previous
                        # `* 0.97` produced an offset that scaled with the elo magnitude
                        # (~36 units at elo=1200, ~45 at elo=1500), pushing the text far
                        # below the line.  Add a small fixed 3-point offset (≈ 3 px at
                        # 72 DPI, visually consistent regardless of elo / output DPI).
                        from matplotlib.transforms import offset_copy

                        text_transform = offset_copy(
                            ax.transData,
                            fig=fig,
                            x=-8,
                            y=-3,
                            units="points",
                        )
                        txt = ax.text(
                            x=0.5,
                            y=baseline_mean,
                            s=baseline_label,
                            va="top",
                            transform=text_transform,
                            **text_kwargs,
                        )
                    txt.set_path_effects(
                        [
                            PathEffects.withStroke(
                                linewidth=2,
                                foreground="white",
                                alpha=0.5,
                            )
                        ]
                    )

                if ylim is not None:
                    ax.set_ylim(ylim)
                if xlim is not None:
                    ax.set_xlim(xlim)

                ticks = boxplot.get_yticks() if use_y else boxplot.get_xticks()
                ticklabels = [
                    tick.get_text() for tick in (boxplot.get_yticklabels() if use_y else boxplot.get_xticklabels())
                ]

                if use_elo:
                    # ----- add elo error bars -----
                    # Get the bar positions

                    # Add asymmetric error bars manually
                    # Pair each tune_method with the same errcolor used by its
                    # `to_plot` entry so the manual whiskers visually match the
                    # bar fill. Overrides extend the list with their own
                    # (tune_method, err_color) pair.
                    tune_method_errcolors = [
                        ("default", errcolors[0]),
                        ("tuned", errcolors[1]),
                        ("tuned_ensembled", errcolors[2]),
                        ("holdout_tuned_ensembled", errcolors[3]),
                    ]
                    _seen_tune_methods: set[str] = set()
                    for ov in tune_method_overrides:
                        if ov.tune_method in _seen_tune_methods:
                            continue
                        _seen_tune_methods.add(ov.tune_method)
                        tune_method_errcolors.append(
                            (
                                ov.tune_method,
                                ov.err_color if ov.err_color is not None else errcolors[0],
                            )
                        )
                    for pos, framework_type in zip(ticks, ticklabels, strict=False):
                        for tune_method, errcolor in tune_method_errcolors:
                            row = df_plot.loc[
                                (df_plot["framework_type"] == framework_type) & (df_plot["tune_method"] == tune_method)
                            ]
                            if len(row) == 1:
                                # not all methods have tuned or tuned_ensembled
                                y = row["elo"].values
                                yerr_low = row["elo-"].values
                                yerr_high = row["elo+"].values
                                if use_y:
                                    plotline, caps, barlinecols = plt.errorbar(
                                        y,
                                        pos,
                                        xerr=[yerr_low, yerr_high],
                                        fmt="none",
                                        color=errcolor,
                                        alpha=err_alpha,
                                        linewidth=err_linewidths[tune_method],
                                    )
                                else:
                                    _plotline, _caps, _barlinecols = plt.errorbar(
                                        pos,
                                        y,
                                        yerr=[yerr_low, yerr_high],
                                        fmt="none",
                                        color=errcolor,
                                        alpha=err_alpha,
                                        linewidth=err_linewidths[tune_method],
                                    )
                                # don't round because it will make the lines longer
                                # plt.setp(barlinecols[0], capstyle="round")

                # ----- highlight bars that contain imputed results -----

                # Map x-tick positions to category labels
                label_lookup = dict(zip(ticks, ticklabels, strict=False))

                has_imputed = False

                for _i, bar in enumerate(boxplot.patches):
                    # Get position and convert to category label
                    pos = bar.get_y() + bar.get_height() / 2 if use_y else bar.get_x() + bar.get_width() / 2
                    category_index = round(pos)  # x-ticks are usually 0, 1, 2, ...
                    category = label_lookup.get(category_index)

                    if category in imputed_names:
                        has_imputed = True
                        bar.set_hatch("xx")

                if not use_y:
                    # ----- alternate rows of x tick labels -----
                    # Get current x tick labels
                    labels = [label.get_text() for label in boxplot.get_xticklabels()]

                    # Add newline to every second label
                    new_labels = [
                        label if i % 2 == 0 else r"$\uparrow$" + "\n" + label for i, label in enumerate(labels)
                    ]

                    if has_non_baselines:
                        # Apply modified labels
                        boxplot.set_xticks(labels)
                        boxplot.set_xticklabels(new_labels)

                # remove unnecessary extra space on the sides
                if use_y:
                    plt.ylim(len(boxplot.get_yticklabels()) - 0.35, -0.65)
                    ax.tick_params(axis="y", pad=0)
                else:
                    plt.xlim(-0.5, len(boxplot.get_xticklabels()) - 0.5)

                ax.legend(loc="upper center", bbox_to_anchor=[0.5, 1.02])

                # reordering the labels
                handles, labels = ax.get_legend_handles_labels()

                if has_imputed:
                    # Create a custom legend patch for "imputed"
                    imputed_patch = Patch(facecolor="gray", edgecolor="white", hatch="xx", label="Partially imputed")

                    # Add to existing legend
                    handles.append(imputed_patch)
                    labels.append("Partially imputed")

                # quick fix
                is_holdout_plot = "Tuned + Ensembled (Holdout)" in labels
                if is_holdout_plot:
                    valid_idxs = [i for i, label in enumerate(labels) if label != "Default"]
                    labels = [labels[i] for i in valid_idxs]
                    handles = [handles[i] for i in valid_idxs]

                order = list(range(len(labels)))
                order = list(reversed(order))

                if method_style_map:
                    tick_method_type_order_orig = [f for f in framework_type_order_orig if f in tick_methods]
                    _apply_ticklabel_styles(
                        ax=ax,
                        use_y=use_y,
                        style_map=method_style_map,
                        default_method_style=default_method_style,
                        display_name_inverse_map=display_name_inverse_map,
                        tick_method_keys=tick_method_type_order_orig,
                    )

                # pass handle & labels lists along with order as below
                # ax.legend(
                #     [handles[i] for i in order],
                #     [labels[i] for i in order],
                #     loc="lower center",
                #     ncol=(len(labels)+1)//2 if has_imputed and use_y else len(labels),
                #     bbox_to_anchor=[0.35 if use_y else 0.5, 1.0],
                # )

                ax.legend(
                    [handles[i] for i in order],
                    [labels[i] for i in order],
                    loc="lower center",
                    ncol=(len(labels) + 1) // 2 if has_imputed and use_y else len(labels),
                    bbox_to_anchor=[0.35 if use_y else 0.5, 1.01],
                    borderaxespad=0.0,
                    borderpad=0.2,
                    handletextpad=0.4,
                    labelspacing=0.3,
                    columnspacing=0.8,
                    # frameon=False,
                )

                # plt.tight_layout()

                if save_prefix:
                    if name_suffix is None:
                        name_suffix = ""
                    fig_path = Path(save_prefix)
                    fig_path.mkdir(parents=True, exist_ok=True)
                    if use_gmean:
                        fig_name = f"tuning-impact-gmean{name_suffix}.{self.figure_file_type}"
                    else:
                        fig_name = f"tuning-impact{name_suffix}.{self.figure_file_type}"
                    fig_save_path = fig_path / fig_name
                    plt.savefig(fig_save_path, dpi=300)
                if show:
                    plt.show()
                plt.close()

    def plot_tabarena_times(self, df: pd.DataFrame, output_dir: Path | str, show: bool = True):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib import ticker

        df = df.copy()

        datasets_impute_freq = df.groupby("dataset")["imputed"].mean()
        datasets_no_impute = list(datasets_impute_freq[datasets_impute_freq <= 0].index)

        df = df[df["dataset"].isin(datasets_no_impute)]

        framework_types = self._get_config_types(df_results=df)

        _f_map, f_map_type, f_map_inverse, f_map_type_name = self.get_framework_type_method_names(
            framework_types=framework_types,
        )

        df["framework_type"] = df[self.method_col].map(f_map_type).fillna(df[self.method_col])
        df["tune_method"] = df[self.method_col].map(f_map_inverse).fillna("default")
        df = df[df["tune_method"].isin(["default", "tuned_ensembled"])]
        df = df[df["framework_type"].isin(framework_types)]
        df.loc[:, "framework_type"] = df["framework_type"].map(f_map_type_name).fillna(df["framework_type"])

        gpu_methods = ["TabICL", "TabDPT", "TabPFNv2", "ModernNCA", "TabM"]

        # add device name
        framework_types = df["framework_type"].unique()
        device_map = {
            ft: f"{ft} " + r"(GPU)" if ft in gpu_methods else f"{ft} (CPU)" if not ft.endswith("(CPU)") else ft
            for ft in framework_types
        }
        device_map = {}
        for ft in framework_types:
            if ft in gpu_methods:
                ft_new = f"{ft} (GPU)"
            elif ft.endswith(("(CPU)", "(GPU)")):
                ft_new = ft
            else:
                ft_new = f"{ft} (CPU)"
            device_map[ft] = ft_new

        df["framework_type"] = df["framework_type"].map(device_map).fillna(df["framework_type"])

        # take mean times
        df = (
            df.groupby(["dataset", "framework_type", "tune_method"])[["time_train_s_per_1K", "time_infer_s_per_1K"]]
            .mean()
            .reset_index()
        )
        df = (
            df.groupby(["framework_type", "tune_method"])[["time_train_s_per_1K", "time_infer_s_per_1K"]]
            .median()
            .reset_index()
        )

        # ----- ChatGPT plotting code -----

        # Unique values for mapping
        # Sort frameworks by max train time
        sorted_frameworks = (
            df.groupby("framework_type")["time_train_s_per_1K"].min().sort_values(ascending=False).index.tolist()
        )
        frameworks = sorted_frameworks
        y_positions = np.arange(len(frameworks))

        # Maps for tuning method to color and marker
        tune_methods = df["tune_method"].unique()
        # color_map = {tm: c for tm, c in zip(tune_methods, plt.cm.tab10.colors)}
        sns_colors = sns.color_palette("muted").as_hex()
        # sns_colors = sns.color_palette("pastel").as_hex()
        color_map = {"default": sns_colors[0], "tuned": sns_colors[1], "tuned_ensembled": sns_colors[2]}
        marker_list = ["o", "s", "^", "D", "P", "*", "X", "v"]
        marker_map = dict(zip(tune_methods, marker_list, strict=False))

        # Create side-by-side subplots with shared y-axis
        fig, (ax_train, ax_infer) = plt.subplots(
            1,
            2,
            sharey=True,
            figsize=(5, 4),
        )

        # Alternate row background on both axes
        for i in range(0, len(frameworks), 2):
            for ax in [ax_train, ax_infer]:
                ax.axhspan(i - 0.5, i + 0.5, facecolor="lightgray", alpha=0.3, zorder=0)

        # Plot training and inference times
        for i, fw in enumerate(frameworks):
            df_fw = df[df["framework_type"] == fw]
            for _, row in df_fw.iterrows():
                color = color_map[row["tune_method"]]
                marker = marker_map[row["tune_method"]]
                ax_train.plot(row["time_train_s_per_1K"], i, marker=marker, color=color, linestyle="None")
                ax_infer.plot(row["time_infer_s_per_1K"], i, marker=marker, color=color, linestyle="None")

        # Train time axis
        ax_train.set_xscale("log")
        ax_train.set_xlabel("Median time per 1K samples [s]")
        ax_train.set_title(r"\textbf{Train+val time}", fontweight="bold")
        ax_train.set_yticks(y_positions)
        ax_train.set_yticklabels(frameworks, fontsize=10)
        ax_train.grid(True, axis="x", alpha=0.5)

        # Inference time axis
        ax_infer.set_xscale("log")
        ax_infer.set_xlabel("Median time per 1K samples [s]")
        ax_infer.set_title(r"\textbf{Inference time}", fontweight="bold")
        ax_infer.set_yticks(y_positions)
        ax_infer.tick_params(labelleft=False)  # Explicitly hide y-tick labels
        ax_infer.grid(True, axis="x", alpha=0.5)

        for ax in [ax_train, ax_infer]:
            ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
            # ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
            # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            # ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        tune_method_display_names = {
            "default": "Default",
            "tuned": "Tuned",
            "tuned_ensembled": "Tuned + Ensembled",
        }

        # Add legend above both plots
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker=marker_map[tm],
                color=color_map[tm],
                linestyle="None",
                label=tune_method_display_names[tm],
                markersize=8,
            )
            for tm in tune_methods
        ]
        fig.legend(
            handles=legend_elements,  # title='Tuning Method',
            loc="upper center",
            bbox_to_anchor=(0.65, 1.01),
            ncol=3,
            fontsize=10,
            title_fontsize=11,
        )

        # Layout adjustment (no clipping)
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_dir / f"time_plot.{self.figure_file_type}")
        if show:
            plt.show()
        plt.close(fig)

    def get_ensemble_weights(
        self,
        df_results: pd.DataFrame,
        method: str,
        excluded_families: list[str] | None = None,
        aggregate_folds: bool = False,
    ) -> pd.DataFrame:
        if self.datasets is not None:
            df_results = df_results[df_results["dataset"].isin(self.datasets)]
        if excluded_families is None:
            excluded_families = []

        df_results_method = df_results[df_results[self.method_col] == method]

        df_ensemble_weights = df_results_method[["dataset", "fold", "ensemble_weight"]]

        full_dict = []
        # available_configs = set()
        # for ensemble_weights in df_ensemble_weights["ensemble_weight"].values:
        #     for k in ensemble_weights.keys():
        #         if k not in available_configs:
        #             available_configs.add(k)
        #     ens_weights_w_dataset_fold = ensemble_weights.copy(deep=True)
        #     full_dict.append(ensemble_weights)
        #
        # df_ensemble_weights_2 = pd.DataFrame()

        for d, f, ensemble_weights in zip(
            df_ensemble_weights["dataset"],
            df_ensemble_weights["fold"],
            df_ensemble_weights["ensemble_weight"],
            strict=False,
        ):
            if isinstance(ensemble_weights, str):
                ensemble_weights = json.loads(ensemble_weights)
            assert isinstance(ensemble_weights, dict)
            ens_weights_w_dataset_fold = dict()
            ens_weights_w_dataset_fold["dataset"] = d
            ens_weights_w_dataset_fold["fold"] = f
            ens_weights_w_dataset_fold.update(ensemble_weights)
            full_dict.append(ens_weights_w_dataset_fold)

        model_to_families = self.config_types

        model_families = set()
        for _m, f in model_to_families.items():
            if f not in model_families:
                model_families.add(f)

        weight_per_family_dict = []
        for cur_dict in full_dict:
            new_dict = {}
            for k, v in cur_dict.items():
                if k == "dataset":
                    new_dict["dataset"] = v
                elif k == "fold":
                    new_dict["fold"] = v
                else:
                    model_family = model_to_families[k]
                    if v is not None:
                        if model_family not in new_dict:
                            new_dict[model_family] = 0
                        new_dict[model_family] += v
            weight_per_family_dict.append(new_dict)

        import pandas as pd

        df = pd.DataFrame(weight_per_family_dict)
        df = df.set_index(["dataset", "fold"])
        df = df.fillna(0)

        df_cols = df.columns
        f_to_add = []
        for f in model_families:
            if f not in df_cols:
                f_to_add.append(f)
        df[f_to_add] = 0

        if excluded_families:
            df = df.drop(columns=excluded_families)

        return self._get_ensemble_weights(
            df_ensemble_weights=df,
            aggregate_folds=aggregate_folds,
            sort_by_mean=True,
        )

    def _process_baselines(
        self,
        df_results: pd.DataFrame,
        baselines: list[str] | None | str,
        baseline_colors: list[str] | None,
    ) -> tuple[list[str], list[str]]:
        methods = df_results[self.method_col].unique()

        if baselines is None:
            baselines = []
        elif baselines == "auto":
            baselines = list(
                df_results[
                    df_results["method_type"].isin(["baseline", "portfolio"])
                    | ((df_results["method_type"] == "config") & df_results["method_subtype"].isna())
                ][self.method_col].unique(),
            )
        else:
            missing_baselines = [b for b in baselines if b not in methods]
            if missing_baselines:
                print(f"Missing specified baselines: {missing_baselines}")
        if baseline_colors is None:
            default_baseline_colors = [
                "black",
                "purple",
                "blue",
                "red",
                "darkgray",
            ]
            # Assign colors dynamically, cycling if baselines > baseline_colors
            baseline_colors = list(itertools.islice(itertools.cycle(default_baseline_colors), len(baselines)))
        assert len(baselines) == len(baseline_colors)
        assert len(baselines) == len(list(set(baselines))), f"Duplicates keys found in baselines: {baselines}"
        # Filter both baselines and baseline_colors using the same mask
        filtered = [(b, c) for b, c in zip(baselines, baseline_colors, strict=False) if b in methods]

        baselines = [b for b, _ in filtered]
        baseline_colors = [c for _, c in filtered]
        return baselines, baseline_colors

    @classmethod
    def _get_ensemble_weights(
        cls,
        df_ensemble_weights: pd.DataFrame,
        aggregate_folds: bool = True,
        sort_by_mean: bool = True,
    ) -> pd.DataFrame:
        df_ensemble_weights = copy.deepcopy(df_ensemble_weights)
        if aggregate_folds:
            df_ensemble_weights = df_ensemble_weights.groupby(level="dataset").mean()
        else:
            index_new = list(df_ensemble_weights.index.to_flat_index())
            index_new = [str(t[0]) + "_" + str(t[1]) for t in index_new]
            df_ensemble_weights.index = index_new

        if sort_by_mean:
            s = df_ensemble_weights.sum()
            df_ensemble_weights = df_ensemble_weights[s.sort_values(ascending=False).index]
        return df_ensemble_weights

    def get_framework_type_method_names(self, framework_types):
        return get_framework_type_method_names(
            framework_types=framework_types,
            max_runtimes=[
                (3600 * 4, "_4h"),
                (None, None),
            ],
            f_map_type_name=self.get_method_rename_map(),
        )

    # TODO: aggregate_config_family: bool
    # TODO: sort rows by size? color by problem type?
    def _plot_ensemble_weights_heatmap(
        self,
        df_ensemble_weights: pd.DataFrame,
        aggregate_folds: bool = True,
        sort_by_mean: bool = True,
        include_mean: bool = True,
        **kwargs,
    ):
        """Parameters
        ----------
        df_ensemble_weights : pd.DataFrame
            The 2nd output object of `repo.evaluate_ensembles(...)
        aggregate_folds : bool, default True
            If True, averages folds of datasets together into single rows representing a dataset.
            If False, each fold of each dataset will be its own row.
        sort_by_mean : bool, default True
            If True, will sort columns by the mean value of the column.
            If False, columns will remain in the original order.
        include_mean : bool, default True
            If True, will add a row at the bottom with label "mean" representing the mean of the config weights across all tasks.
            NaN values are considered 0 for the purposes of calculating the mean.
        **kwargs
            Passed to the `create_heatmap` function

        Returns:
        -------
        plt

        """
        # df_ensemble_weights = self.get_ensemble_weights(
        #     df_ensemble_weights=df_ensemble_weights,
        #     aggregate_folds=aggregate_folds,
        #     sort_by_mean=sort_by_mean,
        # )

        return create_heatmap(df=df_ensemble_weights, include_mean=include_mean, **kwargs)

    def plot_ensemble_weights_heatmap(self, df_ensemble_weights: pd.DataFrame, **kwargs):
        # FIXME: if family never present, then this won't work
        p = self._plot_ensemble_weights_heatmap(df_ensemble_weights=df_ensemble_weights, **kwargs)
        fig_path = Path(f"{self.output_dir}/figures")
        fig_path.mkdir(parents=True, exist_ok=True)
        p.savefig(fig_path / f"ens-weights-per-dataset.{self.figure_file_type}")

    # FIXME: clean this up
    def generate_runtime_plot(
        self,
        df_results: pd.DataFrame,
        deep_dive_kwargs: dict | None = None,
    ):
        if deep_dive_kwargs is None:
            deep_dive_kwargs = {}
        df_results_configs = df_results[df_results["method_type"] == "config"]
        df_results_configs = df_results_configs.copy(deep=True)

        framework_types = self._get_config_types(df_results=df_results_configs)
        df_results_configs = df_results_configs[df_results_configs["config_type"].isin(framework_types)]

        method_rename_map = self.get_method_rename_map()
        df_results_configs["config_type"] = (
            df_results_configs["config_type"].map(method_rename_map).fillna(df_results_configs["config_type"])
        )

        plot_train_time_deep_dive(
            df=df_results_configs,
            expname_outdir=self.output_dir,
            method_col=self.method_col,
            family_col="config_type",
            show=False,
            **deep_dive_kwargs,
        )


class TabArenaEvaluator_2025_06_12(TabArenaEvaluator):
    def get_method_rename_map(self) -> dict[str, str]:
        method_rename_map = super().get_method_rename_map()
        method_rename_map["REALMLP"] = "RealMLP"
        method_rename_map["REALMLP_GPU"] = "RealMLP (GPU)"
        return method_rename_map


def _apply_ticklabel_styles(
    ax,
    use_y: bool,
    style_map: dict[str, MethodLabelStyle] | None,
    default_method_style: MethodLabelStyle | None,
    display_name_inverse_map: dict[str, str] | None = None,
    tick_method_keys: list[str] | None = None,
):
    """Apply per-method ticklabel styling.

    Key feature: if `tick_method_keys` is provided, styles are resolved using the
    original method key for each tick (in axis order). This prevents collisions
    when display_name(A) == B and B also exists in the style_map.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    use_y : bool
        If True, style y tick labels; otherwise style x tick labels.
    style_map : dict[str, MethodLabelStyle] | None
        Mapping from method name -> style (str shorthand for color, or dict of Text style kwargs).
    display_name_inverse_map : dict[str, str] | None
        Optional mapping {display_name -> original_name}. Used only as a fallback when
        `tick_method_keys` is not provided.
    tick_method_keys : list[str] | None
        Optional list of original method names corresponding 1:1 with tick labels,
        in the exact order they appear on the axis.
    """
    if default_method_style is None:
        default_method_style = {}
    if style_map is None:
        style_map = {}

    if display_name_inverse_map is None:
        display_name_inverse_map = {}

    tick_texts = ax.get_yticklabels() if use_y else ax.get_xticklabels()

    if tick_method_keys is not None and len(tick_method_keys) != len(tick_texts):
        raise ValueError(
            f"tick_method_keys length ({len(tick_method_keys)}) must match number of tick labels ({len(tick_texts)}).",
        )

    for i, t in enumerate(tick_texts):
        raw_name = t.get_text()

        # Normalize label text if you added arrows / newlines earlier
        base_name = raw_name.replace("\n", "").replace(r"$\uparrow$", "").strip()

        # --- Resolve the style lookup key ---
        # 1) If provided, use underlying original method key by position (robust to display_name collisions)
        key = tick_method_keys[i] if tick_method_keys is not None else None

        # 2) Otherwise try exact tick text / normalized tick text
        if key is None:
            if raw_name in style_map:
                key = raw_name
            elif base_name in style_map:
                key = base_name

        # 3) Fallback: if tick text is a display_name, map back to original
        if key is None:
            inv_raw = display_name_inverse_map.get(raw_name)
            inv_base = display_name_inverse_map.get(base_name)
            if inv_raw in style_map:
                key = inv_raw
            elif inv_base in style_map:
                key = inv_base
            else:
                key = base_name  # last resort

        style_spec = style_map.get(key)
        if style_spec is None:
            style_spec = {}

        # Normalize style: allow shorthand color string
        style = _normalize_style(style_spec)

        style = _resolve_method_style(
            raw_style=style,
            default_style=default_method_style,
        )

        # Apply supported Text properties dynamically
        for prop, value in style.items():
            setter = getattr(t, f"set_{prop}", None)
            if setter is not None:
                setter(value)


def _resolve_method_style(
    *,
    raw_style: dict | None,
    default_style: dict | None,
) -> dict:
    style = {}
    if default_style:
        style.update(_normalize_style(default_style))
    if raw_style:
        style.update(_normalize_style(raw_style))
    return style


def _normalize_label_style(style: MethodLabelStyle) -> dict[str, object]:
    """Normalize label style spec into a dict usable by matplotlib Text.set_* APIs."""
    if isinstance(style, str):
        return {"color": style}
    return dict(style)


def _normalize_style(style: str | Mapping[str, object] | None) -> dict[str, object]:
    if style is None:
        return {}
    if isinstance(style, str):
        return {"color": style}
    return dict(style)
