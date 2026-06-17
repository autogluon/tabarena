from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.stats import gmean

from ._analysis import DatasetAnalysisMixin
from ._common import FRONTIER_ADVANTAGE, IMPROVABILITY, LOSS_RESCALED, RANK
from ._plotting import PlottingMixin
from ._validation import ResultsValidationMixin
from .elo_utils import EloHelper
from .mean_utils import compute_weighted_mean_by_task
from .winrate_utils import compute_winrate, compute_winrate_matrix

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

MetricDirection = Literal["min", "max"]
MetricAlignment = Literal["row", "method"]
InvalidSubsetPolicy = Literal["raise", "nan", "skip"]


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Defines how to (re)compute a metric from a subset of results_per_task and how to
    reduce it to a single scalar score for a given method.

    - compute(): returns either
        * row-aligned Series (index == results_per_task.index)  [alignment="row"]
        * method-aligned Series (index == method names)         [alignment="method"]
    - score(): returns a float score for method_1 given the computed metric result
    """

    name: str
    direction: MetricDirection
    alignment: MetricAlignment
    compute: Callable[[BenchmarkEvaluator, pd.DataFrame], pd.Series]
    score: Callable[[BenchmarkEvaluator, pd.DataFrame, pd.Series, str], float]
    # Methods that must be present in any subset (e.g., Elo calibration framework)
    required_methods: frozenset[str] = frozenset()
    # What to do if required methods are missing from a subset
    invalid_subset_policy: InvalidSubsetPolicy = "raise"


@dataclass
class _LeaderboardContext:
    """Mutable state threaded through the leaderboard metric producers.

    ``results_agg`` may be replaced by a producer (e.g. ``improvability`` pops its own
    column out of the aggregate and re-emits it with bootstrap CI bars), so the trailing
    aggregate block is read from the context *after* every producer has run.
    """

    evaluator: BenchmarkEvaluator
    results_per_task: pd.DataFrame
    results_agg: pd.DataFrame
    baseline_method: str | None
    elo_kwargs: dict
    relative_error_kwargs: dict


@dataclass(frozen=True)
class _LeaderboardMetric:
    """One column-group of a leaderboard, produced from a ``_LeaderboardContext``.

    ``produce`` returns the list of Series/DataFrames to concatenate (in order) for this
    metric. Adding a leaderboard column is a single registry entry plus a selector key —
    no new ``leaderboard()`` parameter or ``if`` branch.
    """

    key: str
    produce: Callable[[_LeaderboardContext], list]
    requires_baseline: bool = False
    always_on: bool = False


def _lb_elo(ctx: _LeaderboardContext) -> list:
    return [ctx.evaluator.compute_elo(results_per_task=ctx.results_per_task, **ctx.elo_kwargs)]


def _lb_rank(ctx: _LeaderboardContext) -> list:
    return [ctx.results_agg[RANK]]


def _lb_winrate(ctx: _LeaderboardContext) -> list:
    return [ctx.evaluator.compute_winrate(results_per_task=ctx.results_per_task).to_frame()]


def _lb_improvability(ctx: _LeaderboardContext) -> list:
    ev = ctx.evaluator
    tasks = list(ctx.results_per_task[ev.task_col].unique())
    results_per_task_avg = ctx.results_per_task.groupby(ev.groupby_columns)[IMPROVABILITY].mean().reset_index()
    improvability_bootstrap = get_bootstrap_result_lst(
        data=tasks,
        func_=ev._weighted_groupby_mean,
        func_kwargs={"data": results_per_task_avg, "agg_column": IMPROVABILITY},
        num_round=100,
    )
    improvability = ctx.results_agg[IMPROVABILITY]
    ctx.results_agg = ctx.results_agg.drop(columns=[IMPROVABILITY])
    improvability_quantiles = pd.DataFrame(
        {
            f"{IMPROVABILITY}+": improvability_bootstrap.quantile(0.975) - improvability,
            f"{IMPROVABILITY}-": improvability - improvability_bootstrap.quantile(0.025),
        }
    )
    return [improvability, improvability_quantiles]


def _lb_baseline_advantage(ctx: _LeaderboardContext) -> list:
    return [ctx.evaluator.compute_baseline_advantage(ctx.results_per_task, baseline_method=ctx.baseline_method)]


def _lb_frontier_advantage(ctx: _LeaderboardContext) -> list:
    return [ctx.evaluator.compute_frontier_advantage(results_per_task=ctx.results_per_task)]


def _lb_mrr(ctx: _LeaderboardContext) -> list:
    return [ctx.evaluator.compute_mrr(results_per_task=ctx.results_per_task).to_frame()]


def _lb_relative_error(ctx: _LeaderboardContext) -> list:
    return [
        ctx.evaluator.compute_relative_error(
            results_per_task=ctx.results_per_task,
            baseline_method=ctx.baseline_method,
            **ctx.relative_error_kwargs,
        ).to_frame()
    ]


def _lb_skill_score(ctx: _LeaderboardContext) -> list:
    return [
        ctx.evaluator.compute_skill_score(results_per_task=ctx.results_per_task, baseline_method=ctx.baseline_method)
    ]


def _lb_rank_counts(ctx: _LeaderboardContext) -> list:
    return [ctx.evaluator.compute_rank_counts(results_per_task=ctx.results_per_task)]


# Ordered registry: the leaderboard emits these column-groups in this order. ``always_on``
# metrics are always emitted; the rest are selected via ``leaderboard(metrics=...)`` (or the
# legacy ``include_*`` flags). ``requires_baseline`` metrics are skipped when no baseline is set.
_LEADERBOARD_METRICS: tuple[_LeaderboardMetric, ...] = (
    _LeaderboardMetric("elo", _lb_elo),
    _LeaderboardMetric("rank", _lb_rank, always_on=True),
    _LeaderboardMetric("winrate", _lb_winrate),
    _LeaderboardMetric("improvability", _lb_improvability),
    _LeaderboardMetric("baseline_advantage", _lb_baseline_advantage, requires_baseline=True),
    _LeaderboardMetric("frontier_advantage", _lb_frontier_advantage),
    _LeaderboardMetric("mrr", _lb_mrr),
    _LeaderboardMetric("relative_error", _lb_relative_error, requires_baseline=True),
    _LeaderboardMetric("skill_score", _lb_skill_score, requires_baseline=True),
    _LeaderboardMetric("rank_counts", _lb_rank_counts),
)

# Maps the legacy ``include_*`` flags onto registry keys (back-compat selector layer).
_LEADERBOARD_FLAG_TO_KEY = {
    "include_elo": "elo",
    "include_winrate": "winrate",
    "include_improvability": "improvability",
    "include_mrr": "mrr",
    "include_rank_counts": "rank_counts",
    "include_relative_error": "relative_error",
    "include_skill_score": "skill_score",
    "include_baseline_advantage": "baseline_advantage",
    "include_frontier_advantage": "frontier_advantage",
}


# TODO: Should "data" be an init arg? Probably not.
class BenchmarkEvaluator(ResultsValidationMixin, DatasetAnalysisMixin, PlottingMixin):
    def __init__(
        self,
        method_col: str = "method",
        task_col: str = "task",
        error_col: str = "metric_error",
        columns_to_agg_extra: list[str] | str | None = "auto",
        groupby_columns: list[str] | None = None,
        seed_column: str | None = None,
        negative_error_threshold: float = -1e-15,
    ):
        self.method_col = method_col
        self.task_col = task_col
        self.error_col = error_col
        if columns_to_agg_extra is None:
            columns_to_agg_extra = []
        elif columns_to_agg_extra == "auto":
            columns_to_agg_extra = ["time_train_s", "time_infer_s"]
        self.columns_to_agg_extra = columns_to_agg_extra
        self.columns_to_agg = [self.error_col, *self.columns_to_agg_extra]
        if groupby_columns is None:
            groupby_columns = []
        self.groupby_columns = [self.method_col, self.task_col, *groupby_columns]
        self.task_groupby_columns = [self.task_col, *groupby_columns]
        self.seed_column = seed_column
        self.negative_error_threshold = negative_error_threshold

        for c in self.columns_to_agg:
            assert c not in self.groupby_columns
        if self.seed_column is not None:
            assert self.seed_column not in self.columns_to_agg
            assert self.seed_column not in self.groupby_columns

    @property
    def required_input_columns(self) -> list[str]:
        required_input_columns = [
            *self.groupby_columns,
            *self.columns_to_agg,
        ]
        if self.seed_column is not None:
            required_input_columns.append(self.seed_column)
        return required_input_columns

    def _resolve_groupby_columns(self, df: pd.DataFrame, *, task_only: bool = False) -> list[str]:
        """Return the groupby columns, appending the seed column iff it is present in ``df``.

        ``task_only=True`` returns the task-level columns (excluding ``method_col``).
        """
        base = self.task_groupby_columns if task_only else self.groupby_columns
        seed_col = self._seed_col_if_present(df)
        return [*base, seed_col] if seed_col is not None else base

    def leaderboard(
        self,
        data: pd.DataFrame,
        average_seeds: bool = False,
        include_error: bool = False,
        include_elo: bool = True,
        include_winrate: bool = True,
        include_improvability: bool = True,
        include_mrr: bool = False,
        include_rescaled_loss: bool = False,
        include_rank_counts: bool = False,
        include_relative_error: bool = False,
        include_skill_score: bool = False,
        include_baseline_advantage: bool = False,
        include_frontier_advantage: bool = False,
        baseline_method: str | None = None,
        relative_error_kwargs: dict | None = None,
        elo_kwargs: dict | None = None,
        sort_by: str | list[str] | None = "rank",
        metrics: Sequence[str] | None = None,
    ):
        """Build a per-method leaderboard.

        Metric columns are produced by the ordered registry ``_LEADERBOARD_METRICS``.
        Select them either with the legacy ``include_*`` flags or, equivalently, by passing
        ``metrics`` — an iterable of metric keys (e.g. ``["elo", "winrate", "mrr"]``) which,
        when given, overrides the flags. ``rank`` is always included. ``baseline_method`` is
        required for ``baseline_advantage`` / ``relative_error`` / ``skill_score`` (they are
        silently skipped without it). ``include_error`` / ``include_rescaled_loss`` toggle
        columns carried over from the aggregate, and ``average_seeds`` controls per-seed
        handling.
        """
        if elo_kwargs is None:
            elo_kwargs = {}
        if relative_error_kwargs is None:
            relative_error_kwargs = {}
        if baseline_method is None:
            baseline_method = elo_kwargs.get("calibration_framework")

        enabled = self._resolve_leaderboard_metrics(
            metrics,
            {
                "include_elo": include_elo,
                "include_winrate": include_winrate,
                "include_improvability": include_improvability,
                "include_mrr": include_mrr,
                "include_rank_counts": include_rank_counts,
                "include_relative_error": include_relative_error,
                "include_skill_score": include_skill_score,
                "include_baseline_advantage": include_baseline_advantage,
                "include_frontier_advantage": include_frontier_advantage,
            },
        )

        self.verify_data(data=data)

        # average_seeds=True averages each method's per-task error across seeds first;
        # otherwise metrics are computed per seed and averaged afterwards.
        results_per_task = self.compute_results_per_task(data=data, include_seed_col=not average_seeds)

        ctx = _LeaderboardContext(
            evaluator=self,
            results_per_task=results_per_task,
            results_agg=self.aggregate(results_by_dataset=results_per_task),
            baseline_method=baseline_method,
            elo_kwargs=elo_kwargs,
            relative_error_kwargs=relative_error_kwargs,
        )

        results_lst = []
        for metric in _LEADERBOARD_METRICS:
            if not metric.always_on and metric.key not in enabled:
                continue
            if metric.requires_baseline and baseline_method is None:
                continue
            results_lst.extend(metric.produce(ctx))

        # Trailing block: every aggregated column except rank (emitted above). A producer may
        # have popped its own column out of ctx.results_agg (e.g. improvability), so read it
        # here, after the loop.
        cols_to_use = [c for c in ctx.results_agg.columns if c != RANK]
        results_lst.append(ctx.results_agg[cols_to_use])

        results = pd.concat(results_lst, axis=1)

        if sort_by is not None:
            results = results.sort_values(by=sort_by)
        if not include_error:
            results = results.drop(columns=[self.error_col])
        if not include_rescaled_loss:
            results = results.drop(columns=[LOSS_RESCALED])
        if "improvability" not in enabled:
            results = results.drop(columns=[IMPROVABILITY])
        results.index.name = self.method_col

        return results

    @staticmethod
    def _resolve_leaderboard_metrics(metrics: Sequence[str] | None, include_flags: dict[str, bool]) -> set[str]:
        """Resolve the set of enabled metric keys from ``metrics`` or the ``include_*`` flags.

        ``metrics`` (when not ``None``) takes precedence and is validated against the registry;
        otherwise the legacy boolean flags select the keys. ``rank`` is always emitted and is
        not part of this set.
        """
        all_keys = {m.key for m in _LEADERBOARD_METRICS}
        if metrics is not None:
            requested = set(metrics)
            unknown = requested - all_keys
            if unknown:
                raise ValueError(
                    f"Unknown leaderboard metric(s): {sorted(unknown)}. Available: {sorted(all_keys)}.",
                )
            return requested
        return {_LEADERBOARD_FLAG_TO_KEY[flag] for flag, on in include_flags.items() if on}

    # TODO: Consider moving this to a different class or finding a better separation.
    #  The eval code becomes a lot more complicated if we need to account for improperly formatted / invalid data.

    # FIXME: Cleanup

    def get_task_groupby_cols(self, include_seed_col: bool = False):
        task_groupby_cols = self.task_groupby_columns
        if include_seed_col and self.seed_column is not None:
            task_groupby_cols = [*task_groupby_cols, self.seed_column]
        return task_groupby_cols

    def compute_results_per_task(self, data: pd.DataFrame, include_seed_col: bool = False) -> pd.DataFrame:
        groupby_cols = self.groupby_columns
        task_groupby_cols = self.task_groupby_columns
        if include_seed_col and self.seed_column is not None:
            groupby_cols = [*groupby_cols, self.seed_column]
            task_groupby_cols = [*task_groupby_cols, self.seed_column]
        columns_to_agg = self.columns_to_agg
        results_per_task = data[groupby_cols + columns_to_agg].groupby(groupby_cols).mean().reset_index()

        # TODO: Remove `task_groupby_cols` as argument, infer it automatically
        results_per_task_metrics = pd.DataFrame(index=results_per_task.index)
        results_per_task_metrics[RANK] = self.compare_rank_per(results_per_task, task_groupby_cols=task_groupby_cols)
        results_per_task_metrics[IMPROVABILITY] = self.compute_improvability_per(results_per_task, task_groupby_cols)
        results_per_task_metrics[LOSS_RESCALED] = self.compute_loss_rescaled_per(results_per_task, task_groupby_cols)

        return pd.concat(
            [
                results_per_task_metrics,
                results_per_task,
            ],
            axis=1,
        )

    def aggregate(self, results_by_dataset: pd.DataFrame) -> pd.DataFrame:
        if self.seed_column is not None and self.seed_column in results_by_dataset.columns:
            results_by_dataset = results_by_dataset.drop(columns=[self.seed_column])
        results_agg = results_by_dataset.groupby(self.groupby_columns).mean(numeric_only=True)
        # Compute mean
        mean_df = results_agg.groupby([self.method_col]).mean(numeric_only=True)

        # Compute median and prefix column names
        median_df = results_agg.groupby([self.method_col]).median(numeric_only=True)
        median_df.columns = [f"median_{col}" for col in median_df.columns]

        # Combine mean and median
        return pd.concat([mean_df, median_df], axis=1)

    def compute_rank_counts(self, results_per_task: pd.DataFrame) -> pd.DataFrame:
        df = results_per_task.copy()

        group_cols = self.groupby_columns  # e.g., ["task"] or ["task", "seed"]
        task_cols = self.task_groupby_columns
        if self.seed_column is not None and self.seed_column in results_per_task.columns:
            task_seed_cols = [*task_cols, self.seed_column]
        else:
            task_seed_cols = task_cols

        # Per-(group) min/max ranks (1 = best); ties span [min_rank, max_rank]
        min_rank = df.groupby(task_seed_cols)[RANK].rank(method="min", ascending=True)
        max_rank = df.groupby(task_seed_cols)[RANK].rank(method="max", ascending=True)

        # Size of the tie a row belongs to (within group and exact error value)
        tie_size = df.groupby([*task_seed_cols, RANK])[RANK].transform("size").astype(float)

        # Each position k contributes 1 unit per group; split equally across ties covering k
        df["rank=1_count"] = ((min_rank <= 1) & (max_rank >= 1)).astype(float) / tie_size
        df["rank=2_count"] = ((min_rank <= 2) & (max_rank >= 2)).astype(float) / tie_size
        df["rank=3_count"] = ((min_rank <= 3) & (max_rank >= 3)).astype(float) / tie_size

        # Whatever isn't in top-3 goes to >3
        df["rank>3_count"] = 1.0 - (df["rank=1_count"] + df["rank=2_count"] + df["rank=3_count"])

        # Equal-task weighting: average over group_cols (e.g., seeds) then sum per method across tasks
        return (
            df.groupby(group_cols)[["rank=1_count", "rank=2_count", "rank=3_count", "rank>3_count"]]
            .mean()
            .groupby(self.method_col)
            .sum()
        )

    def compute_mrr(self, results_per_task: pd.DataFrame) -> pd.Series:
        """Compute mean reciprocal rank."""
        results_per_task = results_per_task.copy()
        results_per_task["mrr"] = 1 / results_per_task["rank"]
        results_mrr_per_task = results_per_task.groupby(self.groupby_columns)["mrr"].mean()

        results_mrr = results_mrr_per_task.groupby(self.method_col).mean()
        results_mrr.name = "mrr"
        return results_mrr

    def compute_skill_score(
        self,
        results_per_task: pd.DataFrame,
        baseline_method: str,
    ) -> pd.Series:
        relative_error_gmean = self.compute_relative_error(
            results_per_task=results_per_task,
            baseline_method=baseline_method,
            agg="gmean",
        )
        skill_score = 1 - relative_error_gmean
        skill_score.name = "skill_score"
        return skill_score

    def compute_elo(
        self,
        results_per_task: pd.DataFrame,
        calibration_framework: str | None = None,
        calibration_elo: int | None = None,
        INIT_RATING: float = 1000,
        BOOTSTRAP_ROUNDS: int = 100,
        SCALE: int = 400,
        include_quantiles: bool = True,
        round_decimals: int | None = 1,
        use_bootstrap_median: bool = False,
        use_bootstrap_median_for_quantiles: bool = False,
        clip_negative_ci: bool = True,
        post_calibrate: bool = True,
    ) -> pd.DataFrame:
        """Compute Elo ratings for methods evaluated across multiple tasks.

        This aggregates per-task results into head-to-head “battles” and estimates
        per-method Elo scores either by maximum likelihood (single fit) or by a
        bootstrap procedure. Optionally returns uncertainty bars derived from the
        bootstrap distribution.

        Parameters
        ----------
        results_per_task
            Long-form DataFrame with one row per (method, task) containing an error metric.
            Must contain the columns referenced by ``self.method_col`` (method identifier),
            ``self.task_col`` (task identifier), and ``self.error_col`` (lower is better).
        calibration_framework
            Optional name of a reference method to anchor the Elo scale (e.g.,
            set that method’s Elo to ``calibration_elo``).
        calibration_elo
            Elo value assigned to ``calibration_framework`` when provided.
            Ignored if ``calibration_framework`` is ``None``.
        INIT_RATING
            Initial rating used to start optimization / simulation.
        BOOTSTRAP_ROUNDS
            Number of bootstrap resamples of tasks to estimate uncertainty.
            If set to 1, no resampling is performed and quantiles (if requested) collapse to the point estimate.
        SCALE
            Logistic scale factor in the Elo win-probability model (typical value is 400).
            Larger values make probabilities less sensitive to rating differences.
        include_quantiles
            If ``True``, include 2.5%/97.5% quantile bars (or point bars when ``BOOTSTRAP_ROUNDS == 1``).
        round_decimals
            If not ``None``, round the returned values to this many decimal places.
        use_bootstrap_median
            If ``True``, use the bootstrap median rating as the primary Elo estimate instead of the MLE point estimate.
        use_bootstrap_median_for_quantiles
            If ``True``, center the ± bars around the bootstrap median,
            otherwise they are centered around the chosen Elo point estimate.
        clip_negative_ci
            If ``True``, negative widths for ``elo+``/``elo-`` are clipped to 0.
            Negative width can occur if ``use_bootstrap_median=False`` and ``use_bootstrap_median_for_quantiles=False``.
        post_calibrate
            If ``True``, will perform bootstrapping and elo calculation without calibration to determine the 95% CI.
            After determining the 95% CI, the returned `elo` will be adjusted
            so that the `calibration_framework` has an elo of `calibration_elo`.
            This makes the 95% CI +/- independent of the calibration_framework.

        Returns:
        -------
        pd.DataFrame
            DataFrame indexed by method (index name = ``self.method_col``) sorted
            by descending Elo. Always contains:

            - ``elo`` : float
                The Elo rating for each method (rounded if ``round_decimals`` is set).

            If ``include_quantiles`` is ``True``, also contains:

            - ``elo+`` : float
                Upper error bar width (e.g., 97.5% quantile minus center).
            - ``elo-`` : float
                Lower error bar width (e.g., center minus 2.5% quantile).

            When ``BOOTSTRAP_ROUNDS == 1``, ``elo+`` and ``elo-`` will be 0.
        """
        if self.seed_column is not None and self.seed_column in results_per_task.columns:
            split_col = self.seed_column
        else:
            split_col = None

        if post_calibrate:
            post_calibration_framework = calibration_framework
            calibration_framework = None
        else:
            post_calibration_framework = None
        if calibration_elo is None:
            calibration_elo = INIT_RATING

        elo_helper = EloHelper(
            method_col=self.method_col, task_col=self.task_col, error_col=self.error_col, split_col=split_col
        )
        battles = elo_helper.convert_results_to_battles(results_df=results_per_task)

        can_compute_elo = len(battles) > 0
        if not can_compute_elo:
            task_groupby_cols = [self.task_col]
            if split_col is not None:
                task_groupby_cols.append(split_col)

            methods_per_task = results_per_task.groupby(task_groupby_cols)[self.method_col].nunique().sort_values()

            task_method_pairs = (
                results_per_task[[*task_groupby_cols, self.method_col]]
                .drop_duplicates()
                .sort_values([*task_groupby_cols, self.method_col])
            )

            observed_methods = sorted(results_per_task[self.method_col].dropna().unique())

            raise ValueError(
                "Cannot compute Elo because no valid pairwise method comparisons exist. "
                "At least one task/split must contain results for at least two methods.\n"
                f"\nTask grouping columns: {task_groupby_cols}"
                f"\nNum rows: {len(results_per_task)}"
                f"\nNum task/split groups: {len(methods_per_task)}"
                f"\nNum methods: {len(observed_methods)}"
                f"\nObserved methods: {observed_methods}"
                f"\n\nMethods per task/split:\n"
                f"{methods_per_task.to_string()}"
                f"\n\nObserved task/method pairs:\n"
                f"{task_method_pairs.head(100).to_string(index=False)}"
                f"\n\nShowing first {min(len(task_method_pairs), 100)} "
                f"of {len(task_method_pairs)} observed task/method pairs.",
            )

        bootstrap_median = None
        bootstrap_elo_lu = None
        bars_quantiles = None
        if use_bootstrap_median or (include_quantiles and BOOTSTRAP_ROUNDS > 1):
            bootstrap_elo_lu = elo_helper.compute_elo_ratings(
                battles=battles,
                calibration_framework=calibration_framework,
                calibration_elo=calibration_elo,
                INIT_RATING=INIT_RATING,
                BOOTSTRAP_ROUNDS=BOOTSTRAP_ROUNDS,
                SCALE=SCALE,
                show_process=False,
            )
            bootstrap_median = bootstrap_elo_lu.quantile(0.5)

        if use_bootstrap_median:
            elo = bootstrap_median
        else:
            elo = elo_helper.compute_mle_elo(
                battles=battles,
                INIT_RATING=INIT_RATING,
                SCALE=SCALE,
                calibration_framework=calibration_framework,
                calibration_elo=calibration_elo,
            )

        if include_quantiles:
            if BOOTSTRAP_ROUNDS > 1:
                assert bootstrap_elo_lu is not None
                bars_quantiles = pd.DataFrame(
                    dict(
                        lower=bootstrap_elo_lu.quantile(0.025),
                        upper=bootstrap_elo_lu.quantile(0.975),
                    )
                )
            else:
                print(
                    "Warning: Returning 95% CI quantiles for elo when BOOTSTRAP_ROUNDS<=1. "
                    "The CI is invalid and widths will be set to 0.",
                )
                bars_quantiles = pd.DataFrame(
                    dict(
                        lower=elo,
                        upper=elo,
                    )
                )

        bars = pd.DataFrame(
            dict(
                elo=elo,
            )
        )

        if include_quantiles:
            assert bars_quantiles is not None
            relative_to = bootstrap_median if use_bootstrap_median_for_quantiles else elo
            bars["elo+"] = bars_quantiles["upper"] - relative_to
            bars["elo-"] = relative_to - bars_quantiles["lower"]

            if clip_negative_ci:
                bars["elo+"] = bars["elo+"].clip(lower=0)
                bars["elo-"] = bars["elo-"].clip(lower=0)

        if post_calibrate and post_calibration_framework is not None:
            offset = calibration_elo - elo.loc[post_calibration_framework]
            bars["elo"] += offset

        bars = bars.sort_values(by="elo", ascending=False)
        if round_decimals is not None:
            bars["elo"] = np.round(bars["elo"], round_decimals)
            if include_quantiles:
                bars["elo+"] = np.round(bars["elo+"], round_decimals)
                bars["elo-"] = np.round(bars["elo-"], round_decimals)

        bars.index.name = self.method_col

        return bars

    def compute_relative_error(
        self,
        results_per_task: pd.DataFrame,
        baseline_method: str | None,
        agg: str = "mean",
        use_optimal: bool = False,
    ) -> pd.Series:
        assert agg in ["mean", "gmean"]
        results_per_task = results_per_task.copy()
        results_per_task["relative_error"] = self.compute_relative_error_per(
            results_per_task=results_per_task,
            baseline_method=baseline_method,
            use_optimal=use_optimal,
        )
        relative_error_per_task = results_per_task.groupby(self.groupby_columns)["relative_error"].mean()
        if agg == "mean":
            relative_error = relative_error_per_task.groupby(self.method_col).mean()
        elif agg == "gmean":
            relative_error = relative_error_per_task.groupby(self.method_col).apply(gmean)
        else:
            raise ValueError(f"Invalid value for `agg`: {agg}")
        return relative_error

    def compute_relative_error_per(
        self,
        results_per_task: pd.DataFrame,
        baseline_method: str | None,
        use_optimal: bool = False,
    ):
        task_groupby_cols = self._resolve_groupby_columns(results_per_task, task_only=True)
        if use_optimal:
            baseline_result = results_per_task.groupby(task_groupby_cols)[self.error_col].min()
        else:
            assert baseline_method is not None, "baseline_method must not be None!"
            # Collect the baseline error per task (one row per task group)
            baseline_result = results_per_task.loc[
                results_per_task[self.method_col] == baseline_method, [*task_groupby_cols, self.error_col]
            ]
            assert len(baseline_result) > 0, f"Baseline '{baseline_method}' does not exist!"

        baseline_result = baseline_result.rename(columns={self.error_col: "baseline_error"})
        # Map (join) the baseline error back onto every row of its task group
        results_per_task = results_per_task.merge(baseline_result, on=task_groupby_cols, how="left")

        relative_error = results_per_task[self.error_col] / results_per_task["baseline_error"]
        relative_error.name = "relative_error"
        return relative_error

    def compute_winrate(self, results_per_task: pd.DataFrame) -> pd.Series:
        """results_winrate = 1 - ((results_rank - 1) / (len(results)-1))
        results_rank = len(results_winrate) - results_winrate * (len(results_winrate) - 1).
        """
        if self.seed_column is not None and self.seed_column not in results_per_task.columns:
            seed_col = None
        else:
            seed_col = self.seed_column
        return compute_winrate(
            results_per_task=results_per_task,
            task_col=self.task_groupby_columns,
            method_col=self.method_col,
            error_col=self.error_col,
            seed_col=seed_col,
        )

    def compute_winrate_matrix(
        self,
        results_per_task: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute pairwise win-rates between methods.

        Parameters
        ----------
        results_per_task : pd.DataFrame

        Returns:
        -------
        pd.DataFrame
            Square DataFrame indexed and columned by methods.
            Entry (i, j) = win-rate of method i vs method j.
        """
        if self.seed_column is not None and self.seed_column not in results_per_task.columns:
            seed_col = None
        else:
            seed_col = self.seed_column
        return compute_winrate_matrix(
            results_per_task=results_per_task,
            task_col=self.task_groupby_columns,
            method_col=self.method_col,
            error_col=self.error_col,
            seed_col=seed_col,
        )

    def compare_rank_per(
        self,
        df: pd.DataFrame,
        task_groupby_cols: list[str],
    ) -> pd.Series:
        """Add a per-(task, seed) rank column based on error (lower is better).
        - Ties receive average ranks.
        - If `seed_col` is None, each task is treated as a single group.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain task_groupby_cols, self.error_col.
        task_groupby_cols : list[str]
            The groupby columns for calculating rank.

        Returns:
        -------
        pd.Series
            Ranks for each method on each task/split.
        """
        # FIXME: Rounding, parameterize
        #  Maybe rounding should be done as preprocessing?
        # df = df.copy()
        # df[self.error_col] = [round(x[0], 5) for x in zip(df[self.error_col])]

        # Rank within each (task, seed) group; lower error => better (rank 1)
        # groupby(...).rank(...) preserves the original index order
        rank = df.groupby(task_groupby_cols, sort=False)[self.error_col].rank(method="average", ascending=True)
        rank.name = RANK

        return rank

    def compute_improvability_per(self, results_per_task: pd.DataFrame, task_groupby_cols: list[str]) -> pd.Series:
        best_error_per = results_per_task.groupby(task_groupby_cols)[self.error_col].transform("min")
        improvability = (1 - (best_error_per / results_per_task[self.error_col])).fillna(0)
        improvability.name = IMPROVABILITY
        return improvability

    def compute_baseline_advantage(
        self,
        results_per_task: pd.DataFrame,
        baseline_method: str,
    ) -> pd.Series:
        task_groupby_cols = self._resolve_groupby_columns(results_per_task, task_only=True)
        seed_col = self.seed_column if self.seed_column in task_groupby_cols else None
        results_per_task = results_per_task.copy()
        results_per_task["baseline_advantage"] = self.compute_baseline_advantage_per(
            results_per_task,
            task_groupby_cols,
            baseline_method,
        )
        return compute_weighted_mean_by_task(
            df=results_per_task,
            value_col="baseline_advantage",
            task_col=self.task_groupby_columns,
            seed_col=seed_col,
            method_col=self.method_col,
            sort_asc=True,
        )

    def compute_baseline_advantage_per(
        self,
        results_per_task: pd.DataFrame,
        task_groupby_cols: list[str],
        baseline_method: str,
    ) -> pd.Series:
        df = results_per_task.copy()

        # Collect the baseline error per task (one row per task group)
        base = df.loc[df[self.method_col] == baseline_method, [*task_groupby_cols, self.error_col]].rename(
            columns={self.error_col: "baseline_error"}
        )

        # Map (join) the baseline error back onto every row of its task group
        df = df.merge(base, on=task_groupby_cols, how="left")

        # Denominator: max(baseline_error, this_row_error) per row
        denominator = df[[self.error_col, "baseline_error"]].max(axis=1).replace(0, pd.NA)

        # Baseline advantage: (baseline - current) / denom
        baseline_advantage = ((df["baseline_error"] - df[self.error_col]) / denominator).fillna(0)

        baseline_advantage.name = "baseline_advantage"
        baseline_advantage.index = results_per_task.index  # preserve original alignment
        return baseline_advantage

    @staticmethod
    def _loo_min(errors: pd.Series) -> np.ndarray:
        """Leave-one-out minimum: for each entry, the smallest error among all *other* entries.

        This is the group minimum for any row above the minimum, and the 2nd-smallest value for
        the single row holding a unique minimum. When the minimum is tied, the 2nd-smallest equals
        the minimum, so the tied rows correctly keep the minimum. Returns NaN for singleton groups
        (a method with no competitor on the task).
        """
        arr = errors.to_numpy()
        n = arr.shape[0]
        if n < 2:
            return np.full(n, np.nan)
        order = np.argsort(arr, kind="stable")
        smallest = arr[order[0]]
        second = arr[order[1]]
        return np.where(arr > smallest, smallest, second)

    def compute_frontier_advantage_per(self, results_per_task: pd.DataFrame, task_groupby_cols: list[str]) -> pd.Series:
        """Per-(task) frontier advantage: a method's signed margin over the best *other* method.

        For method ``i`` on a task with error ``e_i`` and best-other ("best-in-hindsight ignoring
        itself") error ``e_loo = min_{j != i} e_j`` — the performance frontier excluding ``i`` ::

            frontier_advantage_i = (e_loo - e_i) / max(e_loo, e_i)

        Bounded to ``[-1, 1]`` (higher is stronger). Positive means ``i`` is the unique best and the
        value is its fractional margin over the runner-up; negative means ``i`` trails the best
        method (there it equals ``-improvability``). Unlike ``rank`` / ``improvability``, it does not
        saturate once a method is best — a method that wins by a wide margin scores higher than one
        that barely edges the runner-up — so per-method frontier advantage sorts tasks from
        strongest to weakest for that method.
        """
        e = results_per_task[self.error_col]
        loo_best = results_per_task.groupby(task_groupby_cols, sort=False)[self.error_col].transform(self._loo_min)
        denominator = pd.concat([e, loo_best], axis=1).max(axis=1).replace(0, pd.NA)
        frontier_advantage = ((loo_best - e) / denominator).fillna(0)
        frontier_advantage.name = FRONTIER_ADVANTAGE
        return frontier_advantage

    def compute_frontier_advantage(self, results_per_task: pd.DataFrame) -> pd.Series:
        """Equal-task-weighted mean frontier advantage per method (higher is better)."""
        task_groupby_cols = self._resolve_groupby_columns(results_per_task, task_only=True)
        seed_col = self.seed_column if self.seed_column in task_groupby_cols else None
        results_per_task = results_per_task.copy()
        results_per_task[FRONTIER_ADVANTAGE] = self.compute_frontier_advantage_per(results_per_task, task_groupby_cols)
        return compute_weighted_mean_by_task(
            df=results_per_task,
            value_col=FRONTIER_ADVANTAGE,
            task_col=self.task_groupby_columns,
            seed_col=seed_col,
            method_col=self.method_col,
            sort_asc=False,
        )

    def compute_loss_rescaled_per(self, results_per_task: pd.DataFrame, task_groupby_cols: list[str]) -> pd.Series:
        best_error_per = results_per_task.groupby(task_groupby_cols)[self.error_col].transform("min")
        worst_error_per = results_per_task.groupby(task_groupby_cols)[self.error_col].transform("max")
        loss_rescaled = (results_per_task[self.error_col] - best_error_per) / (worst_error_per - best_error_per).fillna(
            0
        )
        loss_rescaled.name = LOSS_RESCALED
        return loss_rescaled

    def compute_rank(self, results_per_task: pd.DataFrame) -> pd.Series:
        if self.seed_column is not None and self.seed_column not in results_per_task.columns:
            seed_col = None
        else:
            seed_col = self.seed_column

        results_rank = compute_weighted_mean_by_task(
            df=results_per_task,
            value_col=RANK,
            task_col=self.task_groupby_columns,
            seed_col=seed_col,
            method_col=self.method_col,
            sort_asc=True,
        )
        results_rank.name = RANK
        return results_rank

    # TODO: Make faster, can be 100x faster if vectorized properly.
    def _weighted_groupby_mean(self, tasks: list[str], data: pd.DataFrame, agg_column: str) -> pd.Series:
        num_tasks = len(tasks)
        data = data.copy()

        counts = {}
        for task in tasks:
            counts[task] = counts.get(task, 0) + 1
        counts = {k: v / num_tasks for k, v in counts.items()}
        weights = data[self.task_col].map(counts).fillna(0)
        data["_weighted_column"] = data[agg_column] * weights
        column_mean = data.groupby(self.method_col)["_weighted_column"].sum()
        column_mean.index.name = agg_column
        return column_mean

    def _seed_col_if_present(self, df: pd.DataFrame) -> str | None:
        if self.seed_column is not None and self.seed_column in df.columns:
            return self.seed_column
        return None

    def _score_weighted_mean_by_task(
        self,
        df: pd.DataFrame,
        *,
        value_col: str,
        sort_asc: bool,
    ) -> pd.Series:
        """Returns a per-method Series of weighted means using the same equal-task weighting
        logic as other parts of BenchmarkEvaluator.
        """
        seed_col = self._seed_col_if_present(df)
        return compute_weighted_mean_by_task(
            df=df,
            value_col=value_col,
            task_col=self.task_groupby_columns,
            seed_col=seed_col,
            method_col=self.method_col,
            sort_asc=sort_asc,
        )

    def score_if_remove_method(
        self,
        metric: MetricSpec,
        results_per_task: pd.DataFrame,
        *,
        method_1: str,
        method_2: str,
    ) -> float:
        """Compute the scalar score for method_1 after removing method_2 and recomputing metric.
        Returns the resulting score (NOT delta).
        """
        # Keep your prior convention: if we remove method_1 itself, return baseline score on provided df.
        if method_1 == method_2:
            if not self._metric_subset_ok(metric, results_per_task):
                return float("nan")
            metric_values = metric.compute(self, results_per_task)
            return float(metric.score(self, results_per_task, metric_values, method_1))

        subset = results_per_task.loc[results_per_task[self.method_col] != method_2].copy()
        if not self._metric_subset_ok(metric, subset):
            return float("nan")
        metric_values = metric.compute(self, subset)
        return float(metric.score(self, subset, metric_values, method_1))

    def score_series_if_remove_each_method(
        self,
        metric: MetricSpec,
        results_per_task: pd.DataFrame,
        *,
        method_1: str,
    ) -> pd.Series:
        """For a fixed method_1, return a Series indexed by method_2 with values = resulting score
        for method_1 if method_2 were removed.
        """
        methods = results_per_task[self.method_col].dropna().astype(str).unique().tolist()

        scores: dict[str, float] = {}
        for method_2 in methods:
            # Never propose removing required methods (e.g., Elo calibration framework)
            if method_2 in metric.required_methods:
                continue
            scores[method_2] = self.score_if_remove_method(
                metric,
                results_per_task,
                method_1=method_1,
                method_2=method_2,
            )

        s = pd.Series(scores, name=f"{metric.name}_score_for_{method_1}_if_remove_method")
        # Sorting: for min-metrics ascending is "better"; for max-metrics descending is "better"
        return s.sort_values(ascending=(metric.direction == "min"))

    def greedy_remove_methods_optimize_score(
        self,
        metric: MetricSpec,
        results_per_task: pd.DataFrame,
        *,
        method_1: str,
        stop_at_score: float | None = None,
    ) -> pd.Series:
        """Iteratively remove method_2 that yields the best improvement for method_1
        according to metric.direction, recomputing the metric each iteration.

        Returns:
            pd.Series indexed by removed method_2 in removal order
            values = resulting score for method_1 at that iteration (NOT delta).
        """
        current = results_per_task
        removed_in_order: dict[str, float] = {}

        while True:
            # Compute current score for method_1 (and stop checks)
            if not self._metric_subset_ok(metric, current):
                break
            current_metric = metric.compute(self, current)
            cur_score = float(metric.score(self, current, current_metric, method_1))

            # Stop criteria
            if pd.isna(cur_score):
                break
            if stop_at_score is not None:
                if metric.direction == "min" and cur_score <= stop_at_score:
                    break
                if metric.direction == "max" and cur_score >= stop_at_score:
                    break

            remaining_methods = current[self.method_col].dropna().astype(str).unique().tolist()
            # Exclude method_1 and any required methods (e.g., calibration framework)
            candidates = [m for m in remaining_methods if m != method_1 and m not in metric.required_methods]
            if not candidates:
                break

            candidate_scores: dict[str, float] = {}
            for method_2 in candidates:
                subset = current.loc[current[self.method_col] != method_2].copy()
                if not self._metric_subset_ok(metric, subset):
                    if metric.invalid_subset_policy == "skip":
                        continue
                    candidate_scores[method_2] = float("nan")
                    continue
                subset_metric = metric.compute(self, subset)
                candidate_scores[method_2] = float(metric.score(self, subset, subset_metric, method_1))

            scores_s = pd.Series(candidate_scores).dropna()
            if scores_s.empty:
                break

            # Choose best candidate depending on direction
            best_method_2 = scores_s.idxmin() if metric.direction == "min" else scores_s.idxmax()

            best_score = float(scores_s.loc[best_method_2])
            removed_in_order[best_method_2] = best_score

            # Remove best_method_2 and continue
            current = current.loc[current[self.method_col] != best_method_2]

        return pd.Series(removed_in_order, name=f"{metric.name}_score_iter_for_{method_1}")

    def greedy_score_matrix(
        self,
        metric: MetricSpec,
        results_per_task: pd.DataFrame,
        *,
        methods_1: Iterable[str] | None = None,
        stop_at_score: float | None = None,
    ) -> pd.DataFrame:
        """Build a DataFrame:
        rows = method_2 (removed)
        cols = method_1
        cell = resulting score for method_1 at the iteration when method_2 was removed.
        """
        if methods_1 is None:
            methods_1 = results_per_task[self.method_col].dropna().astype(str).unique().tolist()

        col_series: dict[str, pd.Series] = {}
        for method_1 in methods_1:
            col_series[method_1] = self.greedy_remove_methods_optimize_score(
                metric,
                results_per_task,
                method_1=method_1,
                stop_at_score=stop_at_score,
            )

        return pd.DataFrame(col_series)

    # ----------------------------
    # MetricSpec factories
    # ----------------------------

    def metric_spec_error(self) -> MetricSpec:
        """Lower is better. Score = weighted mean error (equal task weighting)."""

        def compute(self: BenchmarkEvaluator, df: pd.DataFrame) -> pd.Series:
            # row-aligned; no recomputation needed
            return df[self.error_col]

        def score(self: BenchmarkEvaluator, df: pd.DataFrame, values: pd.Series, method_1: str) -> float:
            groupby_columns = self._resolve_groupby_columns(df)
            tmp = df[groupby_columns].copy()
            tmp[self.error_col] = values.to_numpy()
            per_method = self._score_weighted_mean_by_task(tmp, value_col=self.error_col, sort_asc=True)
            return float(per_method.get(method_1, float("nan")))

        return MetricSpec(
            name=self.error_col,
            direction="min",
            alignment="row",
            compute=compute,
            score=score,
        )

    def metric_spec_rank(self) -> MetricSpec:
        """Lower is better. Score = weighted mean rank."""

        def compute(self: BenchmarkEvaluator, df: pd.DataFrame) -> pd.Series:
            task_groupby_cols = self._resolve_groupby_columns(df, task_only=True)
            return self.compare_rank_per(df=df, task_groupby_cols=task_groupby_cols)

        def score(self: BenchmarkEvaluator, df: pd.DataFrame, values: pd.Series, method_1: str) -> float:
            groupby_columns = self._resolve_groupby_columns(df)
            tmp = df[groupby_columns].copy()
            tmp[RANK] = values.to_numpy()
            per_method = self._score_weighted_mean_by_task(tmp, value_col=RANK, sort_asc=True)
            return float(per_method.get(method_1, float("nan")))

        return MetricSpec(
            name=RANK,
            direction="min",
            alignment="row",
            compute=compute,
            score=score,
        )

    def metric_spec_improvability(self) -> MetricSpec:
        """Lower is better (0 is ideal). Score = weighted mean improvability."""

        def compute(self: BenchmarkEvaluator, df: pd.DataFrame) -> pd.Series:
            task_groupby_cols = self._resolve_groupby_columns(df, task_only=True)
            return self.compute_improvability_per(results_per_task=df, task_groupby_cols=task_groupby_cols)

        def score(self: BenchmarkEvaluator, df: pd.DataFrame, values: pd.Series, method_1: str) -> float:
            groupby_columns = self._resolve_groupby_columns(df)
            tmp = df[groupby_columns].copy()
            tmp[IMPROVABILITY] = values.to_numpy()
            per_method = self._score_weighted_mean_by_task(tmp, value_col=IMPROVABILITY, sort_asc=True)
            return float(per_method.get(method_1, float("nan")))

        return MetricSpec(
            name=IMPROVABILITY,
            direction="min",
            alignment="row",
            compute=compute,
            score=score,
        )

    def metric_spec_elo(self, **elo_kwargs) -> MetricSpec:
        """Higher is better. Score = Elo value for method_1 computed on the subset."""
        calibration_framework = elo_kwargs.get("calibration_framework")
        required = frozenset([calibration_framework]) if calibration_framework else frozenset()

        def compute(self: BenchmarkEvaluator, df: pd.DataFrame) -> pd.Series:
            bars = self.compute_elo(
                results_per_task=df,
                include_quantiles=False,
                round_decimals=None,
                **elo_kwargs,
            )
            # method-aligned Series
            return bars["elo"]

        def score(self: BenchmarkEvaluator, df: pd.DataFrame, values: pd.Series, method_1: str) -> float:
            return float(values.get(method_1, float("nan")))

        return MetricSpec(
            name="elo",
            direction="max",
            alignment="method",
            compute=compute,
            score=score,
            required_methods=required,
            invalid_subset_policy="raise",
        )

    def _metric_subset_ok(self, metric: MetricSpec, df: pd.DataFrame) -> bool:
        """Return True if df satisfies metric.required_methods; otherwise obey policy."""
        if not metric.required_methods:
            return True
        present = set(df[self.method_col].dropna().astype(str).unique())
        missing = set(metric.required_methods) - present
        if not missing:
            return True
        if metric.invalid_subset_policy == "raise":
            raise ValueError(
                f"Metric {metric.name!r} requires methods {sorted(metric.required_methods)}, "
                f"but subset is missing {sorted(missing)}.",
            )
        if metric.invalid_subset_policy == "nan":
            return False
        # "skip"
        return False


def get_bootstrap_result_lst(
    data: list, func_, rng=None, num_round: int | None = None, func_kwargs=None, seed: int = 0
):
    rows = []
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    if func_kwargs is None:
        func_kwargs = {}
    if num_round is None:
        rows.append(func_(data, **func_kwargs))
    else:
        num_data = len(data)
        for _i in range(num_round):
            data_new = rng.choice(data, size=num_data, replace=True)
            rows.append(func_(data_new, **func_kwargs))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]
