from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Callable, Literal
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from scipy.stats import gmean

from .elo_utils import EloHelper
from .mean_utils import compute_weighted_mean_by_task
from .winrate_utils import compute_winrate, compute_winrate_matrix

RANK = "rank"
IMPROVABILITY = "improvability"
LOSS_RESCALED = "loss_rescaled"

MetricDirection = Literal["min", "max"]
MetricAlignment = Literal["row", "method"]
InvalidSubsetPolicy = Literal["raise", "nan", "skip"]


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """
    Defines how to (re)compute a metric from a subset of results_per_task and how to
    reduce it to a single scalar score for a given method.

    - compute(): returns either
        * row-aligned Series (index == results_per_task.index)  [alignment="row"]
        * method-aligned Series (index == method names)         [alignment="method"]
    - score(): returns a float score for method_1 given the computed metric result
    """
    name: str
    direction: MetricDirection
    alignment: MetricAlignment
    compute: Callable[["TabArena", pd.DataFrame], pd.Series]
    score: Callable[["TabArena", pd.DataFrame, pd.Series, str], float]
    # Methods that must be present in any subset (e.g., Elo calibration framework)
    required_methods: frozenset[str] = frozenset()
    # What to do if required methods are missing from a subset
    invalid_subset_policy: InvalidSubsetPolicy = "raise"


# TODO: Should "data" be an init arg? Probably not.
class TabArena:
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
        self.columns_to_agg = [self.error_col] + self.columns_to_agg_extra
        if groupby_columns is None:
            groupby_columns = []
        self.groupby_columns = [self.method_col, self.task_col] + groupby_columns
        self.task_groupby_columns = [self.task_col] + groupby_columns
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

    def _get_task_groupby_cols(self, results: pd.DataFrame) -> list[str]:
        task_groupby_cols = self.task_groupby_columns
        if self.seed_column is not None and self.seed_column in results.columns:
            task_groupby_cols = task_groupby_cols + [self.seed_column]
        return task_groupby_cols

    def _get_groupby_cols(self, results: pd.DataFrame) -> list[str]:
        groupby_cols = self.groupby_columns
        if self.seed_column is not None and self.seed_column in results.columns:
            groupby_cols = groupby_cols + [self.seed_column]
        return groupby_cols

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
        baseline_method: str | None = None,
        relative_error_kwargs: dict | None = None,
        elo_kwargs: dict | None = None,
        sort_by: str | list[str] | None = "rank",
    ):
        if elo_kwargs is None:
            elo_kwargs = {}
        if relative_error_kwargs is None:
            relative_error_kwargs = {}
        if baseline_method is None:
            baseline_method = elo_kwargs.get("calibration_framework", None)

        self.verify_data(data=data)

        if average_seeds:
            # average each method's task error across the seeds
            # Calculate all metrics on the averaged error for the task.
            results_per_task = self.compute_results_per_task(data=data)
        else:
            # Keep each method's task error for each seed, don't average the error.
            # Calculate all metrics on each seed, then average across seeds to get the metric value for the task.
            results_per_task = self.compute_results_per_task(data=data, include_seed_col=True)

        results_agg = self.aggregate(results_by_dataset=results_per_task)
        results_lst = []

        if include_elo:
            results_lst.append(self.compute_elo(results_per_task=results_per_task, **elo_kwargs))
        results_lst.append(results_agg[RANK])
        if include_winrate:
            results_lst.append(self.compute_winrate(results_per_task=results_per_task).to_frame())
        if include_improvability:
            tasks = list(results_per_task[self.task_col].unique())
            results_per_task_avg = results_per_task.groupby(self.groupby_columns)[IMPROVABILITY].mean().reset_index()
            improvability_bootstrap = get_bootstrap_result_lst(
                data=tasks,
                func_=self._weighted_groupby_mean,
                func_kwargs={"data": results_per_task_avg, "agg_column": IMPROVABILITY},
                num_round=100,
            )
            improvability = results_agg[IMPROVABILITY]
            results_agg = results_agg.drop(columns=[IMPROVABILITY])
            improvability_quantiles = pd.DataFrame({
                f"{IMPROVABILITY}+": improvability_bootstrap.quantile(.975) - improvability,
                f"{IMPROVABILITY}-": improvability - improvability_bootstrap.quantile(.025),
            })

            results_lst += [improvability, improvability_quantiles]
        if include_baseline_advantage and baseline_method is not None:
            results_lst.append(self.compute_baseline_advantage(
                results_per_task,
                baseline_method=baseline_method,
            ))
        if include_mrr:
            results_lst.append(self.compute_mrr(results_per_task=results_per_task).to_frame())
        if baseline_method is not None:
            if include_relative_error:
                results_lst.append(
                    self.compute_relative_error(
                        results_per_task=results_per_task,
                        baseline_method=baseline_method,
                        **relative_error_kwargs
                    ).to_frame()
                )
            if include_skill_score:
                results_lst.append(
                    self.compute_skill_score(results_per_task=results_per_task, baseline_method=baseline_method)
                )

        if include_rank_counts:
            results_lst.append(self.compute_ranks(results_per_task=results_per_task))

        cols_to_use = [c for c in results_agg.columns if c != RANK]
        results_lst.append(results_agg[cols_to_use])

        results = pd.concat(results_lst, axis=1)

        if sort_by is not None:
            results = results.sort_values(by=sort_by)
        if not include_error:
            results = results.drop(columns=[self.error_col])
        if not include_rescaled_loss:
            results = results.drop(columns=[LOSS_RESCALED])
        if not include_improvability:
            results = results.drop(columns=[IMPROVABILITY])
        results.index.name = self.method_col

        return results

    def verify_data(self, data: pd.DataFrame):
        assert isinstance(data, pd.DataFrame)
        data_columns = list(data.columns)
        data_columns_set = set(data_columns)
        assert len(data_columns) == len(data_columns_set)

        missing_columns = []
        present_columns = []
        for c in self.columns_to_agg:
            if c not in data_columns_set:
                missing_columns.append(c)
            else:
                present_columns.append(c)
        for c in self.groupby_columns:
            if c not in data_columns_set:
                missing_columns.append(c)
            else:
                present_columns.append(c)
        if self.seed_column is not None:
            if self.seed_column not in data_columns_set:
                missing_columns.append(self.seed_column)
            else:
                present_columns.append(self.seed_column)

        required_columns = self.groupby_columns + self.columns_to_agg
        if self.seed_column is not None:
            required_columns.append(self.seed_column)
        unused_columns = [d for d in data_columns if d not in required_columns]

        if missing_columns:
            index_names = data.index.names
            missing_in_index = []
            for index_name in index_names:
                if index_name in missing_columns:
                    missing_in_index.append(index_name)
            if missing_in_index:
                msg_extra = (
                    "Columns exist in the index that are required to be columns! "
                    "\n\tEnsure you reset your index to make these columns available: `data = data.reset_index()`\n"
                )
            else:
                msg_extra = ""
            raise ValueError(
                f"{msg_extra}"
                f"Missing required columns:"
                f"\n\tMissing columns ({len(missing_columns)}): {missing_columns}"
                f"\n\tExisting columns ({len(present_columns)}): {present_columns}"
                f"\n\tUnused columns ({len(unused_columns)}): {unused_columns}"
                f"\n\tIndex names ({len(index_names)}): {index_names}"
            )
        if unused_columns:
            print(f"Unused columns: {unused_columns}")

        for c in self.groupby_columns:
            assert data[c].isnull().sum() == 0, f"groupby column {c!r} contains NaN!"
        for c in self.columns_to_agg:
            assert is_numeric_dtype(data[c]), f"aggregation columns must be numeric!"
        for c in self.columns_to_agg:
            if data[c].isnull().sum() != 0:
                invalid_samples = data[data[c].isnull()]

                raise AssertionError(
                    f"Column {c} should not contain null values. "
                    f"Found {data[c].isnull().sum()}/{len(data)} null values! "
                    f"Invalid samples:\n{invalid_samples.head(100).to_markdown()}"
                )

        # TODO: Check no duplicates
        len_data = len(data)
        unique_val_columns = [self.task_col, self.method_col]
        if self.seed_column is not None:
            unique_val_columns.append(self.seed_column)
        len_data_dedupe = len(data.drop_duplicates(unique_val_columns))
        assert len_data == len_data_dedupe

        self.verify_data_is_dense(data=data)
        self.verify_error(data=data)

    def verify_data_is_dense(self, data: pd.DataFrame):
        methods = list(data[self.method_col].unique())
        num_methods = len(methods)
        # FIXME: seed_column
        datasets = list(data[self.task_col].unique())
        num_datasets = len(datasets)

        task_cols = [self.task_col]
        if self.seed_column is not None:
            task_cols.append(self.seed_column)
        unique_tasks = data[task_cols].drop_duplicates().reset_index(drop=True)

        unique_seeds_per_dataset = unique_tasks[self.task_col].value_counts()
        num_tasks = unique_seeds_per_dataset.sum()
        valid_tasks_per_method = data[self.method_col].value_counts()
        valid_methods_per_dataset = data[self.task_col].value_counts()
        valid_methods_per_task = data[task_cols].value_counts()
        invalid_tasks_per_method = (-valid_tasks_per_method + num_tasks).sort_values(ascending=False)
        invalid_methods_per_dataset = (
                -valid_methods_per_dataset + valid_methods_per_dataset.index.map(unique_seeds_per_dataset) * num_methods
        ).sort_values(ascending=False)
        invalid_methods_per_task = (
                -valid_methods_per_task + num_methods
        ).sort_values(ascending=False)

        if (invalid_tasks_per_method != 0).any():
            invalid_tasks_per_method_filtered = invalid_tasks_per_method[invalid_tasks_per_method != 0]
            invalid_methods_per_dataset_filtered = invalid_methods_per_dataset[invalid_methods_per_dataset != 0]
            invalid_methods_per_task_filtered = invalid_methods_per_task[invalid_methods_per_task != 0]
            num_invalid_results = invalid_tasks_per_method.sum()
            # num_invalid_tasks = invalid_methods_per_task_filtered.sum()

            df_experiments_dense = unique_tasks.merge(
                pd.Series(data=methods, name=self.method_col),
                how="cross",
            )
            experiment_cols = task_cols + [self.method_col]
            overlap = pd.merge(df_experiments_dense, data[experiment_cols], on=experiment_cols, how='left', indicator='exist')
            df_missing_experiments = overlap[overlap["exist"] == "left_only"][experiment_cols].sort_values(by=experiment_cols).reset_index(drop=True)

            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
                if len(df_missing_experiments) <= 500:
                    print(f"\nFailed Experiments ({len(df_missing_experiments)}):")
                    print(df_missing_experiments)
                print("\nMethods sorted by failure count:")
                print(invalid_tasks_per_method_filtered)
                print("\nDatasets sorted by failure count:")
                print(invalid_methods_per_dataset_filtered)
            # missing results
            raise AssertionError(
                f"Missing results for some methods. Ensure that all methods have results for all tasks.\n"
                f"If failures exist, fill missing values before passing into this method.\n"
                f"{len(invalid_tasks_per_method_filtered)}/{num_methods} methods with missing tasks. {num_invalid_results} missing results.\n"
                f"{len(invalid_methods_per_dataset_filtered)}/{num_datasets} datasets with missing methods.\n"
                f"{len(invalid_methods_per_task_filtered)}/{num_tasks} tasks with missing methods.\n"
                f"Methods sorted by failure count:\n"
                f"{invalid_tasks_per_method_filtered}"
            )

    def verify_error(self, data: pd.DataFrame):
        min_error = data[self.error_col].min()
        if min_error < 0:
            data_invalid = data[data[self.error_col] < 0]
            num_invalid = len(data_invalid)
            raise ValueError(
                f"Found {num_invalid} rows where {self.error_col} is less than 0! Error can never be less than 0. "
                f"Ensure your error is computed correctly."
                f"\nMinimum value found: {min_error}"
                f"\nSometimes floating point precision can result in a tiny negative value. "
                f"You can fix this by doing: data['{self.error_col}'] = data['{self.error_col}'].clip(lower=0)"
            )

    # TODO: Consider moving this to a different class or finding a better separation.
    #  The eval code becomes a lot more complicated if we need to account for improperly formatted / invalid data.
    def clean_data(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        data = copy.deepcopy(data)
        min_error = data[self.error_col].min()
        if min_error < 0:
            if min_error >= self.negative_error_threshold:
                data[self.error_col] = data[self.error_col].clip(0)
            else:
                self.verify_error(data=data)
        return data

    # FIXME: Cleanup
    def fillna_data(
        self,
        data: pd.DataFrame,
        df_fillna: pd.DataFrame | None = None,
        fillna_method: str = "worst",
    ) -> pd.DataFrame:
        """
        Fills missing (task, seed, method) rows in data with the (task, seed) row in df_fillna.

        Parameters
        ----------
        data : pd.DataFrame
            The data to fill.
        df_fillna : pd.DataFrame | None, default None
            If specified, will fill methods with missing results in `data` with the results in `df_fillna`.
            If specified, `fillna_method` is ignored.
        fillna_method : str, default "worst"
            Either "worst" or the name of a method in self.method_col.
            If "worst", will fill with the result of the method with the worst error on a given task.
            Ignored if `df_fillna` is specified.

        Returns
        -------
        pd.DataFrame
            The filled data.

        """
        if self.seed_column:
            task_columns = [self.task_col, self.seed_column]
        else:
            task_columns = [self.task_col]

        unique_methods = list(data[self.method_col].unique())

        if df_fillna is None:
            if fillna_method == "worst":
                assert df_fillna is None, f"df_fillna must be None if fillna_method='worst'"
                idx_worst = data.groupby(task_columns)[self.error_col].idxmax()
                df_fillna = data.loc[idx_worst]
            elif isinstance(fillna_method, str) and fillna_method in data[self.method_col].unique():
                df_fillna = data.loc[data[self.method_col] == fillna_method]
            else:
                raise AssertionError(
                    f"df_fillna is None and fillna_method {fillna_method!r} is not present in data."
                    f"\n\tValid methods: {list(data[self.method_col].unique())}"
                )
        if self.method_col in df_fillna.columns:
            df_fillna = df_fillna.drop(columns=[self.method_col])

        data = data.set_index([*task_columns, self.method_col], drop=True)

        df_filled = df_fillna[task_columns].merge(
            pd.Series(data=unique_methods, name=self.method_col),
            how="cross",
        )
        df_filled = df_filled.set_index(keys=list(df_filled.columns))

        # missing results
        nan_vals = df_filled.index.difference(data.index)

        # fill valid values
        fill_cols = list(data.columns)
        df_filled[fill_cols] = np.nan
        df_filled[fill_cols] = df_filled[fill_cols].astype(data.dtypes)
        df_filled.loc[data.index] = data

        df_fillna = df_fillna.set_index(task_columns, drop=True)
        a = df_fillna.loc[nan_vals.droplevel(level=self.method_col)]
        a.index = nan_vals
        df_filled.loc[nan_vals] = a
        data = df_filled

        data = data.reset_index(drop=False)

        return data

    def get_task_groupby_cols(self, include_seed_col: bool = False):
        task_groupby_cols = self.task_groupby_columns
        if include_seed_col and self.seed_column is not None:
            task_groupby_cols = task_groupby_cols + [self.seed_column]
        return task_groupby_cols

    def compute_results_per_task(self, data: pd.DataFrame, include_seed_col: bool = False) -> pd.DataFrame:
        groupby_cols = self.groupby_columns
        task_groupby_cols = self.task_groupby_columns
        if include_seed_col and self.seed_column is not None:
            groupby_cols = groupby_cols + [self.seed_column]
            task_groupby_cols = task_groupby_cols + [self.seed_column]
        columns_to_agg = self.columns_to_agg
        results_per_task = data[groupby_cols + columns_to_agg].groupby(groupby_cols).mean().reset_index()

        # TODO: Remove `task_groupby_cols` as argument, infer it automatically
        results_per_task_metrics = pd.DataFrame(index=results_per_task.index)
        results_per_task_metrics[RANK] = self.compare_rank_per(results_per_task, task_groupby_cols=task_groupby_cols)
        results_per_task_metrics[IMPROVABILITY] = self.compute_improvability_per(results_per_task, task_groupby_cols)
        results_per_task_metrics[LOSS_RESCALED] = self.compute_loss_rescaled_per(results_per_task, task_groupby_cols)

        results_per_task = pd.concat([
            results_per_task_metrics,
            results_per_task,
        ], axis=1)
        return results_per_task

    def aggregate(self, results_by_dataset: pd.DataFrame) -> pd.DataFrame:
        if self.seed_column is not None and self.seed_column in results_by_dataset.columns:
            results_by_dataset = results_by_dataset.drop(columns=[self.seed_column])
        results_agg = results_by_dataset.groupby(self.groupby_columns).mean(numeric_only=True)
        # Compute mean
        mean_df = results_agg.groupby([self.method_col]).mean(numeric_only=True)

        # Compute median and prefix column names
        median_df = results_agg.groupby([self.method_col]).median(numeric_only=True)
        median_df.columns = [f'median_{col}' for col in median_df.columns]

        # Combine mean and median
        results_agg = pd.concat([mean_df, median_df], axis=1)
        return results_agg

    def compute_ranks(self, results_per_task: pd.DataFrame) -> pd.DataFrame:
        df = results_per_task.copy()

        group_cols = self.groupby_columns  # e.g., ["task"] or ["task", "seed"]
        task_cols = self.task_groupby_columns
        if self.seed_column is not None and self.seed_column in results_per_task.columns:
            task_seed_cols = task_cols + [self.seed_column]
        else:
            task_seed_cols = task_cols

        # Per-(group) min/max ranks (1 = best); ties span [min_rank, max_rank]
        min_rank = df.groupby(task_seed_cols)[RANK].rank(method="min", ascending=True)
        max_rank = df.groupby(task_seed_cols)[RANK].rank(method="max", ascending=True)

        # Size of the tie a row belongs to (within group and exact error value)
        tie_size = (
            df.groupby(task_seed_cols + [RANK])[RANK]
            .transform("size")
            .astype(float)
        )

        # Each position k contributes 1 unit per group; split equally across ties covering k
        df["rank=1_count"] = ((min_rank <= 1) & (max_rank >= 1)).astype(float) / tie_size
        df["rank=2_count"] = ((min_rank <= 2) & (max_rank >= 2)).astype(float) / tie_size
        df["rank=3_count"] = ((min_rank <= 3) & (max_rank >= 3)).astype(float) / tie_size

        # Whatever isn't in top-3 goes to >3
        df["rank>3_count"] = 1.0 - (df["rank=1_count"] + df["rank=2_count"] + df["rank=3_count"])

        # Equal-task weighting: average over group_cols (e.g., seeds) then sum per method across tasks
        results_ranked = (
            df.groupby(group_cols)[["rank=1_count", "rank=2_count", "rank=3_count", "rank>3_count"]]
            .mean()
            .groupby(self.method_col)
            .sum()
        )

        return results_ranked

    def compute_mrr(self, results_per_task: pd.DataFrame) -> pd.Series:
        """Compute mean reciprocal rank"""
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
            results_per_task=results_per_task, baseline_method=baseline_method, agg="gmean",
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
        """
        Compute Elo ratings for methods evaluated across multiple tasks.

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

        Returns
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

        elo_helper = EloHelper(method_col=self.method_col, task_col=self.task_col, error_col=self.error_col, split_col=split_col)
        battles = elo_helper.convert_results_to_battles(results_df=results_per_task)

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
            bootstrap_median = bootstrap_elo_lu.quantile(.5)

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
                bars_quantiles = pd.DataFrame(dict(
                    lower=bootstrap_elo_lu.quantile(.025),
                    upper=bootstrap_elo_lu.quantile(.975),
                ))
            else:
                print(
                    f"Warning: Returning 95% CI quantiles for elo when BOOTSTRAP_ROUNDS<=1. "
                    f"The CI is invalid and widths will be set to 0."
                )
                bars_quantiles = pd.DataFrame(dict(
                    lower=elo,
                    upper=elo,
                ))

        bars = pd.DataFrame(dict(
            elo=elo,
        ))

        if include_quantiles:
            assert bars_quantiles is not None
            if use_bootstrap_median_for_quantiles:
                relative_to = bootstrap_median
            else:
                relative_to = elo
            bars['elo+'] = bars_quantiles['upper'] - relative_to
            bars['elo-'] = relative_to - bars_quantiles["lower"]

            if clip_negative_ci:
                bars['elo+'] = bars['elo+'].clip(lower=0)
                bars['elo-'] = bars['elo-'].clip(lower=0)

        if post_calibrate and post_calibration_framework is not None:
            offset = calibration_elo - elo.loc[post_calibration_framework]
            bars["elo"] += offset

        bars = bars.sort_values(by="elo", ascending=False)
        if round_decimals is not None:
            bars['elo'] = np.round(bars['elo'], round_decimals)
            if include_quantiles:
                bars['elo+'] = np.round(bars['elo+'], round_decimals)
                bars['elo-'] = np.round(bars['elo-'], round_decimals)

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
        task_groupby_cols = self._get_task_groupby_cols(results=results_per_task)
        if use_optimal:
            baseline_result = results_per_task.groupby(task_groupby_cols)[self.error_col].min()
        else:
            assert baseline_method is not None, f"baseline_method must not be None!"
            # Collect the baseline error per task (one row per task group)
            baseline_result = results_per_task.loc[results_per_task[self.method_col] == baseline_method, task_groupby_cols + [self.error_col]]
            assert len(baseline_result) > 0, f"Baseline '{baseline_method}' does not exist!"

        baseline_result = baseline_result.rename(columns={self.error_col: "baseline_error"})
        # Map (join) the baseline error back onto every row of its task group
        results_per_task = results_per_task.merge(baseline_result, on=task_groupby_cols, how="left")

        relative_error = results_per_task[self.error_col] / results_per_task["baseline_error"]
        relative_error.name = "relative_error"
        return relative_error

    def compute_winrate(self, results_per_task: pd.DataFrame) -> pd.Series:
        """
        results_winrate = 1 - ((results_rank - 1) / (len(results)-1))
        results_rank = len(results_winrate) - results_winrate * (len(results_winrate) - 1)
        """
        if self.seed_column is not None and self.seed_column not in results_per_task.columns:
            seed_col = None
        else:
            seed_col = self.seed_column
        results_winrate = compute_winrate(
            results_per_task=results_per_task,
            task_col=self.task_groupby_columns,
            method_col=self.method_col,
            error_col=self.error_col,
            seed_col=seed_col,
        )
        return results_winrate

    def compute_winrate_matrix(
        self,
        results_per_task: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute pairwise win-rates between methods.

        Parameters
        ----------
        results_per_task : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Square DataFrame indexed and columned by methods.
            Entry (i, j) = win-rate of method i vs method j.
        """
        if self.seed_column is not None and self.seed_column not in results_per_task.columns:
            seed_col = None
        else:
            seed_col = self.seed_column
        winrate_matrix = compute_winrate_matrix(
            results_per_task=results_per_task,
            task_col=self.task_groupby_columns,
            method_col=self.method_col,
            error_col=self.error_col,
            seed_col=seed_col,
        )
        return winrate_matrix

    @staticmethod
    def plot_winrate_matrix(
        winrate_matrix: pd.DataFrame,
        save_path: str | None,
    ):
        import plotly.express as px
        winrate_matrix = winrate_matrix.copy()
        winrate_matrix = (winrate_matrix*100).round().astype('Int64')
        
        fig = px.imshow(
            winrate_matrix,
            color_continuous_scale='PRGn',
            text_auto=".0f"
        )
        fig.update_layout(
            xaxis_title=" Model B: Loser",
            yaxis_title="Model A: Winner",
            xaxis_side="top", height=900, width=1110,
            title=None,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='white',
            coloraxis_colorbar=dict(
                orientation='v',     
                title='Win Rate (%)',
                title_font=dict(size=18),
                tickfont=dict(size=16)
            )
        )
        # axis-specific (optional, if you want a bit larger than global)
        fig.update_xaxes(
            title_font=dict(size=18), 
            tickfont=dict(size=16), 
            showgrid=False
            )
        fig.update_yaxes(
            title_font=dict(size=18), 
            tickfont=dict(size=16), 
            showgrid=False
            )

        fig.update_traces(
            hovertemplate="Model A: %{y}<br>Model B: %{x}<br>Fraction of A Wins: %{z}<extra></extra>",
            textfont=dict(size=16), # numbers inside the heatmap        
        )
        
        if save_path is not None:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_image(save_path)

        return fig

    def compare_rank_per(
        self,
        df: pd.DataFrame,
        task_groupby_cols: list[str],
    ) -> pd.Series:
        """
        Add a per-(task, seed) rank column based on error (lower is better).
        - Ties receive average ranks.
        - If `seed_col` is None, each task is treated as a single group.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain task_groupby_cols, self.error_col.
        task_groupby_cols : list[str]
            The groupby columns for calculating rank.

        Returns
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
        task_groupby_cols = self._get_task_groupby_cols(results=results_per_task)
        seed_col = self.seed_column if self.seed_column in task_groupby_cols else None
        results_per_task = results_per_task.copy()
        results_per_task["baseline_advantage"] = self.compute_baseline_advantage_per(
            results_per_task,
            task_groupby_cols,
            baseline_method,
        )
        results_baseline_advantage = compute_weighted_mean_by_task(
            df=results_per_task,
            value_col="baseline_advantage",
            task_col=self.task_groupby_columns,
            seed_col=seed_col,
            method_col=self.method_col,
            sort_asc=True,
        )
        return results_baseline_advantage

    def compute_baseline_advantage_per(
        self,
        results_per_task: pd.DataFrame,
        task_groupby_cols: list[str],
        baseline_method: str,
    ) -> pd.Series:
        df = results_per_task.copy()

        # Collect the baseline error per task (one row per task group)
        base = (
            df.loc[df[self.method_col] == baseline_method, task_groupby_cols + [self.error_col]]
            .rename(columns={self.error_col: "baseline_error"})
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

    def compute_loss_rescaled_per(self, results_per_task: pd.DataFrame, task_groupby_cols: list[str]) -> pd.Series:
        best_error_per = results_per_task.groupby(task_groupby_cols)[self.error_col].transform("min")
        worst_error_per = results_per_task.groupby(task_groupby_cols)[self.error_col].transform("max")
        loss_rescaled = (results_per_task[self.error_col] - best_error_per) / (
            worst_error_per - best_error_per
        ).fillna(0)
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

    def dataset_outlier(self, results_per_task: pd.DataFrame):
        # Compute how much of an outlier the results of a given dataset are (squared rank differential?)
        raise NotImplementedError

    # TODO: Should plotting live in a separate class?
    def plot_critical_diagrams(self, results_per_task, save_path: str | None = None, show: bool = False, reverse: bool = False):
        import matplotlib.pyplot as plt
        with plt.rc_context({'text.usetex': False}):
            from autorank import autorank
            from autorank._util import cd_diagram
            plt.rcParams.update({'font.size': 12})

            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(1, 1, 1)

            data = results_per_task.pivot_table(index=self.task_col, columns=self.method_col, values="rank")
            result = autorank(data, alpha=0.05, verbose=False, order="ascending", force_mode="nonparametric")

            try:
                _ = cd_diagram(result, reverse=reverse, ax=ax, width=6)
            except KeyError:
                print(f"Not enough methods to generate cd_diagram, skipping...")
                return

            # plt.tight_layout()  # cuts off text
            if save_path is not None:
                parent_dir = str(Path(save_path).parent)
                os.makedirs(parent_dir, exist_ok=True)
                plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            if show:
                plt.show()

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
        """
        Returns a per-method Series of weighted means using the same equal-task weighting
        logic as other parts of TabArena.
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
        """
        Compute the scalar score for method_1 after removing method_2 and recomputing metric.
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
        """
        For a fixed method_1, return a Series indexed by method_2 with values = resulting score
        for method_1 if method_2 were removed.
        """
        methods = (
            results_per_task[self.method_col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

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
        """
        Iteratively remove method_2 that yields the best improvement for method_1
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
            candidates = [
                m for m in remaining_methods
                if m != method_1 and m not in metric.required_methods
            ]
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
            if metric.direction == "min":
                best_method_2 = scores_s.idxmin()
            else:
                best_method_2 = scores_s.idxmax()

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
        """
        Build a DataFrame:
          rows = method_2 (removed)
          cols = method_1
          cell = resulting score for method_1 at the iteration when method_2 was removed
        """
        if methods_1 is None:
            methods_1 = (
                results_per_task[self.method_col].dropna().astype(str).unique().tolist()
            )

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
        """
        Lower is better. Score = weighted mean error (equal task weighting).
        """
        def compute(self: "TabArena", df: pd.DataFrame) -> pd.Series:
            # row-aligned; no recomputation needed
            return df[self.error_col]

        def score(self: "TabArena", df: pd.DataFrame, values: pd.Series, method_1: str) -> float:
            groupby_columns = self._get_groupby_cols(df)
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
        """
        Lower is better. Score = weighted mean rank.
        """
        def compute(self: "TabArena", df: pd.DataFrame) -> pd.Series:
            task_groupby_cols = self._get_task_groupby_cols(results=df)
            return self.compare_rank_per(df=df, task_groupby_cols=task_groupby_cols)

        def score(self: "TabArena", df: pd.DataFrame, values: pd.Series, method_1: str) -> float:
            groupby_columns = self._get_groupby_cols(df)
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
        """
        Lower is better (0 is ideal). Score = weighted mean improvability.
        """
        def compute(self: "TabArena", df: pd.DataFrame) -> pd.Series:
            task_groupby_cols = self._get_task_groupby_cols(results=df)
            return self.compute_improvability_per(results_per_task=df, task_groupby_cols=task_groupby_cols)

        def score(self: "TabArena", df: pd.DataFrame, values: pd.Series, method_1: str) -> float:
            groupby_columns = self._get_groupby_cols(df)
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
        """
        Higher is better. Score = Elo value for method_1 computed on the subset.
        """
        calibration_framework = elo_kwargs.get("calibration_framework", None)
        required = frozenset([calibration_framework]) if calibration_framework else frozenset()

        def compute(self: "TabArena", df: pd.DataFrame) -> pd.Series:
            bars = self.compute_elo(
                results_per_task=df,
                include_quantiles=False,
                round_decimals=None,
                **elo_kwargs,
            )
            # method-aligned Series
            return bars["elo"]

        def score(self: "TabArena", df: pd.DataFrame, values: pd.Series, method_1: str) -> float:
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
                f"but subset is missing {sorted(missing)}."
            )
        if metric.invalid_subset_policy == "nan":
            return False
        # "skip"
        return False

    def plot_dataset_metric_distribution(
        self,
        results_per_task: pd.DataFrame,
        dataset: str,
        metric_col: str | None = None,
        *,
        sort_by: str = "median",  # {"median", "mean"}
        ascending: bool = True,  # lower is better for error-like metrics
        kind: str = "violin",  # {"violin", "box", "bar"}
        show_points: str | bool = "outliers",  # plotly: True, False, "all", "outliers"
        log_y: bool = False,
        max_methods: int | None = 50,  # cap for readability
        save_path: str | None = None,
        title: str | None = None,
    ):
        """
        Plot a single dataset's metric distribution across methods.

        Parameters
        ----------
        results_per_task : pd.DataFrame
            Output of `self.compute_results_per_task(data=data)` (possibly with seeds).
            Must contain at least: [self.task_col, self.method_col, metric_col].
        dataset : str
            Dataset/task name to filter on (value from `self.task_col`).
        metric_col : str | None, default None
            Column to plot. If None, defaults to `self.error_col`.
            Common choices: self.error_col, "rank", "improvability", "loss_rescaled".
        sort_by : {"median","mean"}, default "median"
            How to sort methods on the x-axis.
        ascending : bool, default True
            Whether to sort ascending (typical for error-like metrics).
        kind : {"violin","box","bar"}, default "violin"
            Plot type. If the dataset has only one value per method, it will fall back to "bar".
        show_points : str | bool, default "outliers"
            Whether to show individual points for violin/box.
        log_y : bool, default False
            Whether to log-scale the y-axis.
        max_methods : int | None, default 50
            If set, keep only the top-N methods by the chosen sorter (post-sort).
        save_path : str | None, default None
            If provided, writes the figure to this path (format inferred by extension).
        title : str | None, default None
            Optional plot title.

        Returns
        -------
        fig : plotly.graph_objects.Figure
        df_plot : pd.DataFrame
            Filtered dataframe used for plotting (one row per observation).
        """
        import os
        import plotly.express as px

        if metric_col is None:
            metric_col = self.error_col

        required = {self.task_col, self.method_col, metric_col}
        missing = [c for c in required if c not in results_per_task.columns]
        if missing:
            raise ValueError(
                f"results_per_task is missing required columns: {missing}\n"
                f"Available columns: {list(results_per_task.columns)}"
            )

        df = results_per_task.loc[results_per_task[self.task_col] == dataset].copy()
        if df.empty:
            raise ValueError(
                f"No rows found for {self.task_col} == {dataset!r}.\n"
                f"Available datasets: {sorted(results_per_task[self.task_col].unique())[:50]}"
            )

        # Drop NaNs in the plotted metric (shouldn’t normally exist, but be safe)
        df = df.dropna(subset=[metric_col])

        # Determine whether we actually have a distribution per method (e.g. seeds)
        # If each method has a single value, violin/box isn't very informative.
        per_method_counts = df.groupby(self.method_col)[metric_col].size()
        has_distribution = bool((per_method_counts > 1).any())

        # Sort methods by mean/median for consistent x-axis ordering
        if sort_by not in {"median", "mean"}:
            raise ValueError(f"Invalid sort_by={sort_by!r}. Expected 'median' or 'mean'.")

        if sort_by == "median":
            sorter = df.groupby(self.method_col)[metric_col].median()
        else:
            sorter = df.groupby(self.method_col)[metric_col].mean()

        sorter = sorter.sort_values(ascending=ascending)

        if max_methods is not None and len(sorter) > max_methods:
            sorter = sorter.iloc[:max_methods]
            df = df[df[self.method_col].isin(sorter.index)].copy()

        method_order = list(sorter.index)

        # Pick plot type (force bar if no distribution)
        plot_kind = kind
        if plot_kind in {"violin", "box"} and not has_distribution:
            plot_kind = "bar"

        if title is None:
            title = f"{dataset}: {metric_col} across methods"

        if plot_kind == "violin":
            fig = px.violin(
                df,
                x=self.method_col,
                y=metric_col,
                category_orders={self.method_col: method_order},
                box=True,
                points=show_points,
                title=title,
            )
        elif plot_kind == "box":
            fig = px.box(
                df,
                x=self.method_col,
                y=metric_col,
                category_orders={self.method_col: method_order},
                points=show_points,
                title=title,
            )
        elif plot_kind == "bar":
            # One value per method (or user forced bar): use median/mean as height
            agg = sorter.rename(metric_col).reset_index()
            fig = px.bar(
                agg,
                x=self.method_col,
                y=metric_col,
                category_orders={self.method_col: method_order},
                title=title,
            )
        else:
            raise ValueError(f"Invalid kind={kind!r}. Expected 'violin', 'box', or 'bar'.")

        fig.update_layout(
            xaxis_title="Method",
            yaxis_title=metric_col,
            xaxis_tickangle=45,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="white",
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True)

        if log_y:
            fig.update_yaxes(type="log")

        if save_path is not None:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_image(save_path)

        return fig, df

    def dataset_representativeness(
        self,
        results_per_task: pd.DataFrame,
        value_col: str = RANK,
        *,
        similarity: str = "spearman",  # {"spearman", "pearson"}
        population_mode: str = "loo_mean",  # {"loo_mean", "global_mean"}
        score_mode: str = "to_population",  # {"to_population", "mean_pairwise"}
        min_methods: int | None = None,
    ) -> dict:
        """
        Quantify how similar each dataset is to the others based on model performance,
        then identify the most- and least-representative datasets.

        Intuition
        ---------
        Each dataset defines a vector over methods (e.g., ranks or errors).
        Two datasets are "similar" if these vectors are correlated across methods.
        Representativeness is measured by how well a dataset aligns with the population.

        Parameters
        ----------
        results_per_task : pd.DataFrame
            Output of `self.compute_results_per_task(data=data)` (may include seed col).
        value_col : str, default "rank"
            Column used to compare method performance. Common:
            - RANK (lower is better)
            - self.error_col (lower is better)
            - IMPROVABILITY (higher is better) [still fine; correlation handles direction]
            - LOSS_RESCALED (lower is better)
        similarity : {"spearman","pearson"}, default "spearman"
            Similarity metric for comparing datasets via correlation across methods.
            Spearman is recommended (rank-based; robust to scale).
        population_mode : {"loo_mean","global_mean"}, default "loo_mean"
            How to define the population reference vector:
            - "loo_mean": reference for dataset d is mean vector over all datasets except d
            - "global_mean": reference is mean vector over all datasets (includes d)
        score_mode : {"to_population","mean_pairwise"}, default "to_population"
            How to assign a single representativeness score per dataset:
            - "to_population": correlation(dataset_vector, population_reference_vector)
            - "mean_pairwise": average correlation to all other datasets
        min_methods : int | None
            If set, require at least this many methods to compute correlations.

        Returns
        -------
        out : dict with keys
            - "representativeness": pd.DataFrame indexed by dataset with columns:
                ["score", "num_methods", "num_obs"]
            - "most_representative": str
            - "least_representative": str
            - "dataset_similarity_matrix": pd.DataFrame (dataset x dataset)
            - "method_matrix": pd.DataFrame (dataset x method), values = value_col (after seed-avg)
        """
        if similarity not in {"spearman", "pearson"}:
            raise ValueError(f"similarity must be 'spearman' or 'pearson', got {similarity!r}")
        if population_mode not in {"loo_mean", "global_mean"}:
            raise ValueError(f"population_mode must be 'loo_mean' or 'global_mean', got {population_mode!r}")
        if score_mode not in {"to_population", "mean_pairwise"}:
            raise ValueError(f"score_mode must be 'to_population' or 'mean_pairwise', got {score_mode!r}")

        required = {self.task_col, self.method_col, value_col}
        missing = [c for c in required if c not in results_per_task.columns]
        if missing:
            raise ValueError(
                f"results_per_task is missing required columns: {missing}\n"
                f"Available columns: {list(results_per_task.columns)}"
            )

        df = results_per_task.copy()

        # If seed column exists, average within (task, method) so each dataset has one value per method.
        group_cols = [self.task_col, self.method_col]
        if self.seed_column is not None and self.seed_column in df.columns:
            df = df.groupby(group_cols, as_index=False)[value_col].mean()

        # Pivot to dataset x method matrix
        M = df.pivot(index=self.task_col, columns=self.method_col, values=value_col)

        # Optional sanity: require sufficient overlap
        num_methods_per_dataset = M.notna().sum(axis=1)
        if min_methods is not None:
            keep = num_methods_per_dataset >= min_methods
            M = M.loc[keep]
            num_methods_per_dataset = num_methods_per_dataset.loc[keep]
            if M.shape[0] == 0:
                raise ValueError(f"No datasets have >= {min_methods} methods after filtering.")

        # If you're using TabArena's verify_data_is_dense, M should be fully dense.
        # But we handle missingness by using pairwise corr + aligning vectors where needed.

        # 1) Dataset↔dataset similarity matrix (correlation across methods)
        # corr over rows => easiest via transpose (methods as rows, datasets as cols)
        dataset_sim = M.T.corr(method=similarity)

        # 2) Representativeness score per dataset
        if score_mode == "mean_pairwise":
            # Mean similarity to all other datasets (exclude self)
            score = (dataset_sim.sum(axis=1) - 1.0) / (dataset_sim.shape[0] - 1) if dataset_sim.shape[0] > 1 else \
            dataset_sim.iloc[:, 0]
        else:
            # Similarity to a "population reference vector"
            # Reference vector is mean performance across datasets (optionally leave-one-out).
            if population_mode == "global_mean":
                ref = M.mean(axis=0)
                score = M.apply(lambda row: row.corr(ref, method=similarity), axis=1)
            else:
                # Leave-one-out mean reference: ref_d = mean of all datasets except d
                # Compute efficiently: total_sum - row, then divide by (n-1)
                n = M.shape[0]
                if n <= 1:
                    score = pd.Series(index=M.index, data=np.nan, dtype=float)
                else:
                    total = M.sum(axis=0)

                    def loo_corr(row: pd.Series) -> float:
                        ref_d = (total - row) / (n - 1)
                        return row.corr(ref_d, method=similarity)

                    score = M.apply(loo_corr, axis=1)

        rep = pd.DataFrame({
            "score": score,
            "num_methods": num_methods_per_dataset,
            "num_obs": df.groupby(self.task_col).size().reindex(M.index).astype(int),
        }).sort_values("score", ascending=False)

        most_rep = rep.index[0]
        least_rep = rep.index[-1]

        return {
            "representativeness": rep,
            "most_representative": most_rep,
            "least_representative": least_rep,
            "dataset_similarity_matrix": dataset_sim.loc[M.index, M.index],
            "method_matrix": M,
        }

    def dataset_fold_similarity(
        self,
        results_per_task: pd.DataFrame,
        dataset: str,
        value_col: str = RANK,
        *,
        similarity: str = "spearman",  # {"spearman", "pearson"}
        agg_across_methods: str = "mean",  # {"mean", "median"}
        min_methods: int | None = None,
        return_pairwise: bool = True,
    ) -> dict:
        """
        Quantify how similar a dataset's folds/seeds are, based on how methods perform.

        Assumes `results_per_task` was produced with `include_seed_col=True`, so that
        each row corresponds to (task, seed, method) with a metric like rank/error.

        Approach
        --------
        For the chosen dataset, build a matrix:
            rows = fold/seed
            cols = method
            values = value_col (e.g., rank or metric_error)

        Then compute fold↔fold similarity as correlation across methods.

        Parameters
        ----------
        results_per_task : pd.DataFrame
            Output of `self.compute_results_per_task(..., include_seed_col=True)`.
            Must include `self.seed_column` and `value_col`.
        dataset : str
            Task/dataset name (value from `self.task_col`) to analyze.
        value_col : str, default RANK
            Performance column to compare across folds. Good defaults:
            - RANK (ranking alignment)
            - self.error_col (raw metric error)
            - IMPROVABILITY, LOSS_RESCALED, etc.
        similarity : {"spearman","pearson"}, default "spearman"
            Correlation type for fold similarity.
        agg_across_methods : {"mean","median"}, default "mean"
            How to aggregate a per-fold similarity score from its similarities to other folds.
        min_methods : int | None
            If set, require each fold have at least this many methods present
            (after pivot) to be included.
        return_pairwise : bool, default True
            If True, also return a tidy dataframe of fold-pair similarities.

        Returns
        -------
        out : dict
            - "fold_similarity_matrix": pd.DataFrame (fold x fold)
            - "fold_scores": pd.DataFrame indexed by fold with columns:
                ["score", "num_methods"]
            - "most_similar_pair": tuple | None  (fold_i, fold_j, sim)
            - "least_similar_pair": tuple | None (fold_i, fold_j, sim)
            - "fold_method_matrix": pd.DataFrame (fold x method)
            - "pairwise": pd.DataFrame (optional) columns [seed_i, seed_j, similarity]
        """
        if self.seed_column is None:
            raise ValueError("TabArena.seed_column is None, but fold similarity requires a seed/fold column.")
        if self.seed_column not in results_per_task.columns:
            raise ValueError(
                f"results_per_task must include seed column {self.seed_column!r}. "
                "Ensure you called compute_results_per_task(..., include_seed_col=True)."
            )
        if similarity not in {"spearman", "pearson"}:
            raise ValueError(f"similarity must be 'spearman' or 'pearson', got {similarity!r}")
        if agg_across_methods not in {"mean", "median"}:
            raise ValueError(f"agg_across_methods must be 'mean' or 'median', got {agg_across_methods!r}")

        required = {self.task_col, self.method_col, self.seed_column, value_col}
        missing = [c for c in required if c not in results_per_task.columns]
        if missing:
            raise ValueError(
                f"results_per_task is missing required columns: {missing}\n"
                f"Available columns: {list(results_per_task.columns)}"
            )

        df = results_per_task.loc[results_per_task[self.task_col] == dataset].copy()
        if df.empty:
            raise ValueError(
                f"No rows found for {self.task_col} == {dataset!r}.\n"
                f"Available datasets: {sorted(results_per_task[self.task_col].unique())[:50]}"
            )

        # If duplicates exist for (task, seed, method), average them.
        df = df.groupby([self.task_col, self.seed_column, self.method_col], as_index=False)[value_col].mean()

        # Pivot -> folds x methods
        M = df.pivot(index=self.seed_column, columns=self.method_col, values=value_col)

        # Optionally filter folds with insufficient method coverage
        num_methods_per_fold = M.notna().sum(axis=1)
        if min_methods is not None:
            keep = num_methods_per_fold >= min_methods
            M = M.loc[keep]
            num_methods_per_fold = num_methods_per_fold.loc[keep]
            if M.shape[0] == 0:
                raise ValueError(f"No folds have >= {min_methods} methods after filtering.")

        # Correlation across methods between folds
        # corr over rows => transpose to make folds columns
        fold_sim = M.T.corr(method=similarity)

        # Per-fold "representativeness among folds": average similarity to other folds
        if fold_sim.shape[0] <= 1:
            scores = pd.Series(index=fold_sim.index, data=np.nan, dtype=float)
        else:
            off_diag = fold_sim.copy()
            np.fill_diagonal(off_diag.values, np.nan)
            if agg_across_methods == "mean":
                scores = off_diag.mean(axis=1, skipna=True)
            else:
                scores = off_diag.median(axis=1, skipna=True)

        fold_scores = pd.DataFrame({
            "score": scores,
            "num_methods": num_methods_per_fold.reindex(fold_sim.index).astype(int),
        }).sort_values("score", ascending=False)

        # Identify most/least similar fold pairs
        most_pair = None
        least_pair = None
        if fold_sim.shape[0] >= 2:
            # consider only upper triangle (excluding diagonal)
            sim_vals = fold_sim.where(np.triu(np.ones(fold_sim.shape), k=1).astype(bool))
            stacked = sim_vals.stack(dropna=True)  # MultiIndex (seed_i, seed_j) -> similarity
            if len(stacked) > 0:
                (i_max, j_max), v_max = stacked.idxmax(), float(stacked.max())
                (i_min, j_min), v_min = stacked.idxmin(), float(stacked.min())
                most_pair = (i_max, j_max, v_max)
                least_pair = (i_min, j_min, v_min)

        out = {
            "fold_similarity_matrix": fold_sim,
            "fold_scores": fold_scores,
            "most_similar_pair": most_pair,
            "least_similar_pair": least_pair,
            "fold_method_matrix": M,
        }

        if return_pairwise and fold_sim.shape[0] >= 2:
            sim_vals = fold_sim.where(np.triu(np.ones(fold_sim.shape), k=1).astype(bool))

            # Ensure the row/col index level names are unique before stacking,
            # otherwise reset_index can try to insert a column that already exists.
            idx_name = fold_sim.index.name or self.seed_column or "fold"
            col_name = fold_sim.columns.name or self.seed_column or "fold"

            # Use internal unique names for stack/reset_index
            sim_vals_safe = sim_vals.copy()
            sim_vals_safe.index = sim_vals_safe.index.rename(f"__{idx_name}_i")
            sim_vals_safe.columns = sim_vals_safe.columns.rename(f"__{col_name}_j")

            # pandas >= 2.1: support future_stack=True (silences FutureWarning)
            try:
                s = sim_vals_safe.stack(future_stack=True)
            except TypeError:
                s = sim_vals_safe.stack(dropna=True)
            s = s.dropna()

            pairwise = s.rename("similarity").reset_index()

            # Rename back to desired output names (avoid collisions on rename too)
            col_i = f"{self.seed_column}_i"
            col_j = f"{self.seed_column}_j"
            pairwise = pairwise.rename(columns={
                f"__{idx_name}_i": col_i,
                f"__{col_name}_j": col_j,
            })

            pairwise = pairwise.sort_values("similarity", ascending=False).reset_index(drop=True)
            out["pairwise"] = pairwise
        return out

    def rank_datasets_by_fold_similarity(
        self,
        results_per_task: pd.DataFrame,
        value_col: str = RANK,
        *,
        similarity: str = "spearman",  # {"spearman", "pearson"}
        agg_fold_score: str = "mean_pairwise",  # {"mean_pairwise", "median_pairwise"}
        min_folds: int = 2,
        min_methods: int | None = None,
        include_pairwise_extremes: bool = True,
        # --- new: stability estimation controls ---
        target_reliability: float = 0.90,
        stability_cap_folds: int = 100,
        stability_conservative: bool = True,
        stability_rho_floor: float = 0.01,
    ) -> dict:
        """
        Rank datasets by how consistently their folds/seeds agree with each other,
        and estimate how many folds are needed to get a stable global ordering.

        Assumes `results_per_task` was computed with `include_seed_col=True` so that
        each row corresponds to (task, seed, method) with a metric like rank/error.

        For each dataset:
          1) build fold x method matrix
          2) compute fold-fold similarity matrix (correlation across methods)
          3) summarize it into a single "fold agreement" score
          4) estimate stability/reliability of the k-fold aggregate and folds needed
             to hit `target_reliability`, using Spearman–Brown style extrapolation:
                Rel(k) = (k*rho) / (1 + (k-1)*rho)
             where rho ~= fold_agreement.

        Parameters
        ----------
        results_per_task : pd.DataFrame
            Output of `self.compute_results_per_task(..., include_seed_col=True)`.
            Must include `self.seed_column`.
        value_col : str, default RANK
            Column used to compare method performance across folds.
        similarity : {"spearman","pearson"}, default "spearman"
            Similarity metric for fold-fold correlation across methods.
        agg_fold_score : {"mean_pairwise","median_pairwise"}, default "mean_pairwise"
            How to aggregate fold-fold similarities into a dataset score.
        min_folds : int, default 2
            Require at least this many folds/seeds for a dataset to be scored.
        min_methods : int | None
            If set, require each fold have at least this many methods present.
            Passed through to `dataset_fold_similarity`.
        include_pairwise_extremes : bool, default True
            If True, include the most/least similar fold pair per dataset.
        target_reliability : float, default 0.90
            Stability threshold τ in (0,1). Higher = stricter stability requirement.
        stability_cap_folds : int, default 100
            Maximum folds to return for `folds_needed_for_stability@τ`.
        stability_conservative : bool, default True
            If True, treat rho<=0 as "not stably orderable" and return cap (with rho floored).
        stability_rho_floor : float, default 0.01
            Minimum rho used when conservative and rho is extremely small/negative.

        Returns
        -------
        out : dict
            - "dataset_ranking": pd.DataFrame indexed by dataset with columns:
                ["fold_agreement", "num_folds", "num_methods_min", "num_methods_mean",
                 "rho_used", "stability_at_num_folds@{τ}", "folds_needed_for_stability@{τ}",
                 "most_similar_pair", "least_similar_pair" (optional), "error" (optional)]
              sorted descending by fold_agreement.
            - "per_dataset": dict[str, dict]
              Lightweight per-dataset summary.
            - "stability_params": dict
              Echo of stability configuration.
        """
        import numpy as np
        import pandas as pd

        if self.seed_column is None:
            raise ValueError("TabArena.seed_column is None, but fold similarity ranking requires a seed/fold column.")
        if self.seed_column not in results_per_task.columns:
            raise ValueError(
                f"results_per_task must include seed column {self.seed_column!r}. "
                "Ensure you called compute_results_per_task(..., include_seed_col=True)."
            )
        if agg_fold_score not in {"mean_pairwise", "median_pairwise"}:
            raise ValueError(
                f"agg_fold_score must be 'mean_pairwise' or 'median_pairwise', got {agg_fold_score!r}"
            )
        if similarity not in {"spearman", "pearson"}:
            raise ValueError(f"similarity must be 'spearman' or 'pearson', got {similarity!r}")
        if not (0 < float(target_reliability) < 1):
            raise ValueError(f"target_reliability must be in (0,1), got {target_reliability!r}")

        tau = float(target_reliability)
        tau_tag = f"{tau:.2f}".rstrip("0").rstrip(".")  # e.g. "0.9" or "0.95"
        col_stability_at_k = f"stability_at_num_folds"
        col_folds_needed = f"folds_needed_for_stability@{tau_tag}"

        def _rel(k: int, rho: float) -> float:
            if k <= 0 or not np.isfinite(rho) or rho <= 0:
                return 0.0
            return (k * rho) / (1.0 + (k - 1) * rho)

        def _folds_needed(rho: float) -> int:
            # k >= tau(1-rho)/(rho(1-tau))
            if not np.isfinite(rho) or rho <= 0:
                return int(stability_cap_folds)
            k_float = (tau * (1.0 - rho)) / (rho * (1.0 - tau))
            k_req = int(np.ceil(k_float))
            return max(1, min(int(stability_cap_folds), k_req))

        datasets = list(pd.Index(results_per_task[self.task_col].unique()).sort_values())

        rows: list[dict] = []
        per_dataset: dict[str, dict] = {}

        for ds in datasets:
            try:
                out_ds = self.dataset_fold_similarity(
                    results_per_task=results_per_task,
                    dataset=ds,
                    value_col=value_col,
                    similarity=similarity,
                    agg_across_methods="mean",  # not used for dataset score
                    min_methods=min_methods,
                    return_pairwise=False,
                )
            except Exception as e:
                rows.append({
                    self.task_col: ds,
                    "fold_agreement": np.nan,
                    "num_folds": 0,
                    "num_methods_min": np.nan,
                    "num_methods_mean": np.nan,
                    "rho_used": np.nan,
                    col_stability_at_k: np.nan,
                    col_folds_needed: np.nan,
                    "error": repr(e),
                })
                continue

            fold_sim = out_ds["fold_similarity_matrix"]
            fold_method = out_ds["fold_method_matrix"]

            num_folds = int(fold_sim.shape[0])
            methods_per_fold = fold_method.notna().sum(axis=1)
            num_methods_min = int(methods_per_fold.min()) if len(methods_per_fold) else np.nan
            num_methods_mean = float(methods_per_fold.mean()) if len(methods_per_fold) else np.nan

            if num_folds < min_folds:
                rows.append({
                    self.task_col: ds,
                    "fold_agreement": np.nan,
                    "num_folds": num_folds,
                    "num_methods_min": num_methods_min,
                    "num_methods_mean": num_methods_mean,
                    "rho_used": np.nan,
                    col_stability_at_k: np.nan,
                    col_folds_needed: np.nan,
                    "error": f"Too few folds (min_folds={min_folds})",
                })
                continue

            # Aggregate off-diagonal similarities into dataset agreement score (rho estimate)
            sim_vals = fold_sim.to_numpy(copy=True)
            np.fill_diagonal(sim_vals, np.nan)

            if agg_fold_score == "mean_pairwise":
                rho = float(np.nanmean(sim_vals))
            else:
                rho = float(np.nanmedian(sim_vals))

            rho_used = rho
            notes = None
            if stability_conservative:
                # clamp to [-1,1]
                if rho_used > 1:
                    rho_used = 1.0
                if rho_used < -1:
                    rho_used = -1.0
                if rho_used <= 0:
                    # treat as not reliably aggregatable; use floor for rel(k) curve but folds_needed -> cap
                    notes = "rho<=0; stability may not be achievable by averaging folds alone"
                    rho_used = max(float(stability_rho_floor), 0.0)

            stability_at_k = float(_rel(num_folds, rho_used))
            folds_needed = int(_folds_needed(rho_used))

            row = {
                self.task_col: ds,
                "fold_agreement": rho,  # raw estimate from observed fold similarities
                "num_folds": num_folds,
                "num_methods_min": num_methods_min,
                "num_methods_mean": num_methods_mean,
                "rho_used": rho_used,  # what we used for stability extrapolation
                col_stability_at_k: stability_at_k,
                col_folds_needed: folds_needed,
            }
            if notes is not None:
                row["stability_note"] = notes

            if include_pairwise_extremes and num_folds >= 2:
                tri = np.triu(np.ones_like(sim_vals, dtype=bool), k=1)
                vals = sim_vals[tri]
                if np.isfinite(vals).any():
                    fold_labels = list(fold_sim.index)
                    ij = np.argwhere(tri)
                    k_max = int(np.nanargmax(vals))
                    i_max, j_max = map(int, ij[k_max])
                    row["most_similar_pair"] = (fold_labels[i_max], fold_labels[j_max], float(vals[k_max]))

                    k_min = int(np.nanargmin(vals))
                    i_min, j_min = map(int, ij[k_min])
                    row["least_similar_pair"] = (fold_labels[i_min], fold_labels[j_min], float(vals[k_min]))
                else:
                    row["most_similar_pair"] = None
                    row["least_similar_pair"] = None

            rows.append(row)

            per_dataset[ds] = {
                "fold_agreement": rho,
                "num_folds": num_folds,
                "num_methods_min": num_methods_min,
                "num_methods_mean": num_methods_mean,
                "rho_used": rho_used,
                col_stability_at_k: stability_at_k,
                col_folds_needed: folds_needed,
            }
            if notes is not None:
                per_dataset[ds]["stability_note"] = notes
            if include_pairwise_extremes:
                per_dataset[ds]["most_similar_pair"] = row.get("most_similar_pair", None)
                per_dataset[ds]["least_similar_pair"] = row.get("least_similar_pair", None)

        ranking = pd.DataFrame(rows).set_index(self.task_col)
        ranking = ranking.sort_values(by="fold_agreement", ascending=False)

        return {
            "dataset_ranking": ranking,
            "per_dataset": per_dataset,
            "stability_params": {
                "target_reliability": tau,
                "cap_folds": stability_cap_folds,
                "conservative": stability_conservative,
                "rho_floor": stability_rho_floor,
                "similarity": similarity,
                "agg_fold_score": agg_fold_score,
            },
        }

    @staticmethod
    def estimate_folds_for_stable_ordering(
        fold_agreement: float,
        num_folds: int,
        *,
        target_reliability: float = 0.90,  # "stable ordering" threshold
        cap: int = 100,  # safety cap
        conservative: bool = True,
        rho_floor: float = 0.01,
    ) -> dict:
        """
        Estimate how many folds are needed to get a stable global ordering on a dataset,
        based on observed inter-fold agreement.

        Uses Spearman–Brown style extrapolation:
            Rel(k) = (k*rho) / (1 + (k-1)*rho)
        where rho ~ mean pairwise fold correlation ("fold_agreement").

        Parameters
        ----------
        fold_agreement : float
            Mean pairwise fold similarity (e.g., Spearman correlation across methods).
        num_folds : int
            Number of folds used to estimate fold_agreement.
            Used mainly for reporting; the extrapolation itself uses fold_agreement.
        target_reliability : float, default 0.90
            Target reliability for considering the ordering "stable".
        cap : int, default 100
            Maximum folds to return (avoid absurd numbers).
        conservative : bool, default True
            If True, clamp rho into a plausible range and warn on edge cases.
        rho_floor : float, default 0.01
            Minimum rho to use when conservative and rho is extremely small/negative.

        Returns
        -------
        dict with:
            - "rho": used rho
            - "target_reliability": tau
            - "k_required": estimated folds needed (int, capped)
            - "rel_at_num_folds": estimated reliability at the observed num_folds
            - "rel_curve": optional small table (k, Rel(k)) for k in [1..min(cap, 20)]
            - "notes": list[str]
        """
        notes = []
        rho = float(fold_agreement)

        if not np.isfinite(rho):
            raise ValueError("fold_agreement must be finite.")

        # Correlation should be in [-1, 1]; we only have meaningful fold-averaging reliability for rho > 0
        if conservative:
            if rho > 1:
                notes.append("rho > 1 encountered; clamping to 1.")
                rho = 1.0
            if rho < -1:
                notes.append("rho < -1 encountered; clamping to -1.")
                rho = -1.0

            if rho <= 0:
                notes.append(
                    "fold_agreement <= 0 suggests folds disagree or are unrelated; "
                    "a stable global ordering may not be achievable by averaging folds alone. "
                    "Returning cap."
                )
                rho = max(rho_floor, 0.0)

        tau = float(target_reliability)
        if not (0 < tau < 1):
            raise ValueError("target_reliability must be in (0, 1).")

        def rel(k: int) -> float:
            if k <= 0:
                return np.nan
            if rho <= 0:
                return 0.0
            return (k * rho) / (1.0 + (k - 1) * rho)

        rel_at_num = rel(int(num_folds))

        # Solve k >= tau(1-rho)/(rho(1-tau))
        if rho <= 0:
            k_req = cap
        else:
            k_float = (tau * (1.0 - rho)) / (rho * (1.0 - tau))
            k_req = int(np.ceil(k_float))
            k_req = max(1, k_req)

        if k_req > cap:
            notes.append(f"Estimated folds ({k_req}) exceed cap ({cap}); returning cap.")
            k_req = cap

        rel_curve = [(k, rel(k)) for k in range(1, min(cap, 20) + 1)]
        rel_curve_df = pd.DataFrame(rel_curve, columns=["k", "reliability"])

        return {
            "rho": rho,
            "target_reliability": tau,
            "k_required": k_req,
            "rel_at_num_folds": rel_at_num,
            "rel_curve": rel_curve_df,
            "notes": notes,
        }


def get_bootstrap_result_lst(data: list, func_, rng=None, num_round: int = None, func_kwargs=None, seed: int = 0):
    rows = []
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    if func_kwargs is None:
        func_kwargs = {}
    if num_round is None:
        rows.append(func_(data, **func_kwargs))
    else:
        num_data = len(data)
        for i in range(num_round):
            data_new = rng.choice(data, size=num_data, replace=True)
            rows.append(func_(data_new, **func_kwargs))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]
