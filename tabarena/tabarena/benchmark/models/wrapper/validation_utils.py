from __future__ import annotations

import numpy as np
import pandas as pd


class TabArenaValidationProtocolExecMixin:
    """Logic to adjust to various validation data use cases for benchmarking."""

    def __init__(
        self,
        *,
        use_task_specific_validation: bool = False,
        target_name: str | None = None,
        group_on: str | list[str] | None = None,
        stratify_on: str | None = None,
        time_on: str | None = None,
        group_time_on: str | None = None,
    ):
        """Mixin to handle validation protocol logic for benchmarking.

        Parameters
        ----------
        use_task_specific_validation: bool
            If True, this class will automatically update the validation splitting logic
            based on the characteristics of the data for an experiment.
            If False, this class does nothing.
        target_name:
            The name of the target column. This might be needed for splitting.
        stratify_on:
            The name of the column used for stratification during splitting.
        group_on:
            Name of the column that identifies groups for group-wise splitting during
            validation. If not None, this column will be used to ensure that all rows
            with the same value in this column are kept in the same split.
        time_on:
            The name of the column that identifies time for time-based splitting during
        group_time_on:
            The name of the column that identifies column that identifies time within
            groups.
        """
        self.use_task_specific_validation = use_task_specific_validation
        self.target_name = target_name
        self.stratify_on = stratify_on
        self.group_on = group_on
        self.time_on = time_on
        self.group_time_on = group_time_on
        self.groups_indicator_col_name = "__tabarena_group_split_indicator__"

    def resolve_validation_splits(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        num_folds: int | None,
        num_repeats: int | None,
    ) -> tuple[pd.Series | None, int | None, int | None]:
        """Build a custom group split indicator if needed.

        Returns:
        -------
        groups_indicator: np.ndarray or None
            None, if no splits indicator is needed.
            Otherwise, a Series of shape (n_samples,) where each value is an integer
            representing the split assignment for that row.
        num_folds: int or None
            The number of folds to use for validation.
            This may be updated based on the number of group instances in the data.
        num_repeats: int or None
            The number of repeats to use for validation.
            This may be updated based on the number of group instances in the data.
        """
        groups_indicator = None

        if not self.use_task_specific_validation:
            return None, num_folds, num_repeats

        print("==== Using task-specific validation logic!")
        # Stop early if the model does not want to do any validation.
        if (num_folds is None) or (num_folds <= 1):
            print(
                "Info: num_folds is None or <= 1, skipping validation splitting logic."
                "\n\t The model is configured to do not validation at all!"
            )
            return groups_indicator, num_folds, num_repeats

        num_group_instances = self.get_num_group_instances(X=X)
        print(f"Number of group instances in data: {num_group_instances}")
        num_folds, num_repeats = self._resolve_number_of_splits(
            num_folds=num_folds,
            num_repeats=num_repeats,
            num_group_instances=num_group_instances,
        )

        if self.group_on is not None:
            groups_indicator = self._resolve_group_splits(X=X, y=y, num_folds=num_folds)

        return groups_indicator, num_folds, num_repeats

    def _resolve_number_of_splits(
        self, *, num_folds: int, num_repeats: int, num_group_instances: int
    ) -> tuple[int, int]:
        """Determine the number of splits we want to use.

        Parameters
        ----------
        num_folds: int
            The number of folds entered for validation.
        num_repeats: int
            The number of repeats entered for validation.
        num_group_instances: int
            The number of group instances in the data.
        """
        # FIXME: do not hardcode this in the future.
        new_num_folds, new_num_repeats = None, None
        if num_group_instances <= 500:
            new_num_folds = 5
            new_num_repeats = 10
        else:
            # We want these by default for all other data in our benchmark.
            assert num_folds == 8
            assert num_repeats == 1

        if new_num_folds is not None:
            print(
                f"Updating num_bag_folds from {new_num_folds} to {new_num_folds}"
                f" since number of group instances is less than num_bag_folds."
            )
            num_folds = new_num_folds

        if new_num_repeats is not None:
            print(
                f"Updating num_bag_sets from {num_repeats} to {new_num_repeats}"
                f" since number of group instances is less than num_bag_folds."
            )
            num_repeats = new_num_repeats

        return num_folds, num_repeats

    def _resolve_group_splits(
        self, *, X: pd.DataFrame, y: pd.Series, num_folds: int
    ) -> pd.Series:
        """Create a group-based split given the specified group_on column(s).
        Then transform this into split indices for AutoGluon's group splitter logic.

        Some comments about this logic:
            - If we are given a list of group_on columns, we create a combined group
                identifier by concatenating the values in those columns.
            - We use stratification if stratify_on is specified.
            - We dynamically adjust the number of splits to be the minimum of
                and the number of unique groups in the data.
        """
        from sklearn.model_selection import (
            GroupKFold,
            StratifiedGroupKFold,
        )

        # Get group label
        group_col = self.group_on
        if isinstance(group_col, list):
            # If multiple columns are specified, create a combined group identifier
            groups_data = X[group_col].astype(str).agg("_".join, axis=1)
        else:
            groups_data = X[group_col]

        n_groups_in_data = groups_data.nunique()
        assert n_groups_in_data > 1, (
            f"Need at least 2 unique groups for group-wise splitting, but got {n_groups_in_data} unique groups from column(s) {group_col}!"
        )
        num_folds = min(num_folds, n_groups_in_data)
        print(f"Found #groups in data: {n_groups_in_data}")

        if self.stratify_on is None:
            stratify_on_data = None
        else:
            assert self.target_name is not None
            if self.stratify_on == self.target_name:
                stratify_on_data = y
            else:
                assert self.stratify_on in X.columns, (
                    f"Stratification column '{self.stratify_on}' not found in features!"
                )
                stratify_on_data = X[self.stratify_on]

        splitter = GroupKFold if stratify_on_data is None else StratifiedGroupKFold
        sklearn_splits = splitter(
            n_splits=num_folds, random_state=42, shuffle=True
        ).split(X=X, y=y, groups=groups_data)

        groups_indicator = np.full(shape=len(X), fill_value=-1, dtype=int)
        for fold_idx, (_, test_index) in enumerate(sklearn_splits):
            groups_indicator[test_index] = fold_idx

        return pd.Series(groups_indicator)

    def get_num_group_instances(self, X: pd.DataFrame):
        """Compute the number of rows that represent how much (multi-instance) samples
        the data has. This is used to determine which splits to use.
        """
        from tabarena.benchmark.task.user_task import TabArenaTaskMetadataMixin

        return TabArenaTaskMetadataMixin.get_num_instance_groups(
            X=X, group_on=self.group_on
        )
