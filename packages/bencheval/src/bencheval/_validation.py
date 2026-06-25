from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


class ResultsValidationMixin:
    """Data validation, cleaning and fill-in for results frames (mixed into BenchmarkEvaluator)."""

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
                f"\n\tIndex names ({len(index_names)}): {index_names}",
            )

        for c in self.groupby_columns:
            assert data[c].isnull().sum() == 0, f"groupby column {c!r} contains NaN!"
        for c in self.columns_to_agg:
            assert is_numeric_dtype(data[c]), "aggregation columns must be numeric!"
        for c in self.columns_to_agg:
            if data[c].isnull().sum() != 0:
                invalid_samples = data[data[c].isnull()]

                raise AssertionError(
                    f"Column {c} should not contain null values. "
                    f"Found {data[c].isnull().sum()}/{len(data)} null values! "
                    f"Invalid samples:\n{invalid_samples.head(100).to_markdown()}",
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

        task_cols = self.get_task_groupby_cols(include_seed_col=True)
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
        invalid_methods_per_task = (-valid_methods_per_task + num_methods).sort_values(ascending=False)

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
            experiment_cols = [*task_cols, self.method_col]
            overlap = pd.merge(
                df_experiments_dense, data[experiment_cols], on=experiment_cols, how="left", indicator="exist"
            )
            df_missing_experiments = (
                overlap[overlap["exist"] == "left_only"][experiment_cols]
                .sort_values(by=experiment_cols)
                .reset_index(drop=True)
            )

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
                f"{invalid_tasks_per_method_filtered}",
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
                f"You can fix this by doing: data['{self.error_col}'] = data['{self.error_col}'].clip(lower=0)",
            )

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

    def fillna_data(
        self,
        data: pd.DataFrame,
        df_fillna: pd.DataFrame | None = None,
        fillna_method: str = "worst",
        imputed_col: str | None = None,
    ) -> pd.DataFrame:
        """Fills missing (task, seed, method) rows in data with the (task, seed) row in df_fillna.

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
        imputed_col : str | None, default None
            If specified, add (or update) this boolean column: ``True`` for rows filled from
            `df_fillna`, ``False`` for rows already present in `data`. When ``None`` (default), no
            such column is added.

        Returns:
        -------
        pd.DataFrame
            The filled data.

        """
        task_columns = [self.task_col, self.seed_column] if self.seed_column else [self.task_col]

        unique_methods = list(data[self.method_col].unique())

        if df_fillna is None:
            if fillna_method == "worst":
                assert df_fillna is None, "df_fillna must be None if fillna_method='worst'"
                idx_worst = data.groupby(task_columns)[self.error_col].idxmax()
                df_fillna = data.loc[idx_worst]
            elif isinstance(fillna_method, str) and fillna_method in data[self.method_col].unique():
                df_fillna = data.loc[data[self.method_col] == fillna_method]
            else:
                raise AssertionError(
                    f"df_fillna is None and fillna_method {fillna_method!r} is not present in data."
                    f"\n\tValid methods: {list(data[self.method_col].unique())}",
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

        if imputed_col is not None:
            if imputed_col not in df_filled.columns:
                df_filled[imputed_col] = False
            df_filled.loc[nan_vals, imputed_col] = True
            df_filled[imputed_col] = df_filled[imputed_col].fillna(0).astype(bool)

        return df_filled.reset_index(drop=False)
