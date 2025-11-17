from __future__ import annotations

import copy
import os
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
        average_seeds: bool = True,
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
