from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from tabarena.benchmark.task.user_task import (
    GroupLabelTypes,
)

if TYPE_CHECKING:
    from tabarena.benchmark.task.user_task import (
        SplitTimeHorizonTypes,
        SplitTimeHorizonUnitTypes,
    )


class TabArenaValidationProtocolExecMixin:
    """Logic to adjust to various validation data use cases for benchmarking."""

    tiny_data_num_folds: int = 5
    tiny_data_num_repeats: int = 3
    max_samples_for_tiny_data: int = 500

    def __init__(
        self,
        *,
        use_task_specific_validation: bool = False,
        target_name: str | None = None,
        group_on: str | list[str] | None = None,
        stratify_on: str | None = None,
        time_on: str | None = None,
        group_time_on: str | None = None,
        group_labels: GroupLabelTypes | None = None,
        split_time_horizon: SplitTimeHorizonTypes | None = None,
        split_time_horizon_unit: SplitTimeHorizonUnitTypes | None = None,
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
        group_labels:
            Whether the group_on column(s) contain labels for each sample, or for each group.
        split_time_horizon:
            Time horizon for deployment/test data
        split_time_horizon_unit:
            Unit for time horizon for deployment/test data (e.g. days, months, years)
        """
        self.use_task_specific_validation = use_task_specific_validation
        self.target_name = target_name
        self.stratify_on = stratify_on
        self.group_on = group_on
        self.time_on = time_on
        self.group_time_on = group_time_on
        self.group_labels = group_labels
        self.split_time_horizon = split_time_horizon
        self.split_time_horizon_unit = split_time_horizon_unit

    def resolve_validation_splits(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        num_folds: int | None,
        num_repeats: int | None,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]] | None, int | None, int | None]:
        """Determine which splits setting to use, and if needed, which custom splits.

        Returns:
        -------
        custom_splits: list[tuple[np.ndarray, np.ndarray]] or None
            None, if no custom splits are needed.
            Otherwise, a list of tuples of train/test indices (as np.ndarrays) to use
            for validation splitting.
            IMPORTANT: the split will return the index of the input data X!
        num_folds: int or None
            The number of folds to use for validation.
            This may be updated based on the number of group instances in the data.
        num_repeats: int or None
            The number of repeats to use for validation.
            This may be updated based on the number of group instances in the data.
        """
        custom_splits = None

        if not self.use_task_specific_validation:
            return None, num_folds, num_repeats

        # Stop early if the model does not want to do any validation.
        if (num_folds is None) or (num_folds <= 1):
            logger.info(
                "\nnum_folds is None or <= 1, skipping validation splitting logic."
                "\n\t The model is configured to do not validation at all!"
            )
            return custom_splits, num_folds, num_repeats

        num_group_instances = self.get_num_group_instances(X=X)
        logger.info(
            "\n=== Using task-specific validation logic!"
            f"\n\tNumber of groups/instances in data: {num_group_instances}"
            f"\n\tStratify_on: {self.stratify_on}"
            f"\n\tGroup_on: {self.group_on}"
            f"\n\tTime_on: {self.time_on}"
            f"\n\tGroup_time_on: {self.group_time_on}"
            f"\n\tGroup_labels: {self.group_labels}"
            f"\n\tSplit_time_horizon: {self.split_time_horizon}"
            f"\n\tSplit_time_horizon_unit: {self.split_time_horizon_unit}"
        )

        num_folds, num_repeats = self._resolve_number_of_splits(
            num_folds=num_folds,
            num_repeats=num_repeats,
            num_group_instances=num_group_instances,
        )

        stratify_on_data = None
        if self.stratify_on is not None:
            stratify_on_data = (
                X[self.stratify_on] if self.stratify_on in X.columns else y
            )

        groups_data = None
        group_labels = None
        if (self.time_on is not None) and (self.group_on is not None):
            raise NotImplementedError

        if self.time_on is not None:
            groups_data, num_folds_new = self.time_on_to_groups_data(
                X=X, time_on=self.time_on, num_folds=num_folds
            )
            num_repeats = 1
            logger.info(
                f"\n\tFolds time-based grouping: before={num_folds}; "
                f"after={num_folds_new}"
                f"\n\tnum_repeats set to 1!"
            )
            num_folds = num_folds_new
            # Set group labels as needed for time split
            group_labels = GroupLabelTypes.PER_SAMPLE

        if self.group_on is not None:
            groups_data = self.group_on_to_groups_data(X=X, group_on=self.group_on)
            group_labels = self.group_labels

        if groups_data is not None:
            custom_splits = self._resolve_group_splits(
                X=X,
                num_folds=num_folds,
                num_repeats=num_repeats,
                stratify_on_data=stratify_on_data,
                groups_data=groups_data,
                group_labels=group_labels,
            )

        # Sanity checks for custom splits
        if custom_splits is not None:
            for train_idx, test_idx in custom_splits:
                if stratify_on_data is not None:
                    stratify_values = stratify_on_data.unique()
                    train_stratify_values = set(
                        stratify_on_data.iloc[train_idx].unique()
                    )
                    test_stratify_values = set(stratify_on_data.iloc[test_idx].unique())
                    assert (
                        train_stratify_values
                        == test_stratify_values
                        == set(stratify_values)
                    ), (
                        f"Stratification values in train and test splits do not match!"
                        f"\n\tOverall stratification values: {stratify_values}"
                        f"\n\tTrain stratification values: {train_stratify_values}"
                        f"\n\tTest stratification values: {test_stratify_values}"
                    )

        return custom_splits, num_folds, num_repeats

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
        new_num_folds, new_num_repeats = None, None
        if num_group_instances <= self.max_samples_for_tiny_data:
            new_num_folds = self.tiny_data_num_folds
            new_num_repeats = self.tiny_data_num_repeats
        else:
            # We want these by default for all other data in our benchmark.
            assert num_folds == 8
            assert (num_repeats == 1) or (num_repeats is None)

        if new_num_folds is not None:
            logger.info(
                f"\nUpdating num_bag_folds from {new_num_folds} to {new_num_folds}"
                f" since number of group instances is less than num_bag_folds."
            )
            num_folds = new_num_folds

        if new_num_repeats is not None:
            logger.info(
                f"\nUpdating num_bag_sets from {num_repeats} to {new_num_repeats}"
                f" since number of group instances is less than num_bag_folds."
            )
            num_repeats = new_num_repeats

        return num_folds, num_repeats

    def _resolve_group_splits(
        self,
        *,
        X: pd.DataFrame,
        num_folds: int,
        num_repeats: int,
        stratify_on_data: pd.Series | None,
        groups_data: pd.Series,
        group_labels: GroupLabelTypes | None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Create a group-based split given the specified group_on column(s).
        Then transform this into split indices for AutoGluon's group splitter logic.

        Some comments about this logic:
            - If we are given a list of group_on columns, we create a combined group
                identifier by concatenating the values in those columns.
            - We use stratification if stratify_on is specified.
            - We dynamically adjust the number of splits to be the minimum of
                and the number of unique groups in the data.
        """
        from data_foundry.curation_recommendations import get_recommended_grouped_splits

        assert not groups_data.isna().any(), (
            "Group column(s) contain NaN values, which is not allowed for group-wise splitting!"
        )

        if group_labels not in [
            GroupLabelTypes.PER_GROUP,
            GroupLabelTypes.PER_SAMPLE,
        ]:
            raise ValueError(f"Invalid group_labels value: {group_labels}")

        if group_labels == GroupLabelTypes.PER_GROUP:
            n_groups_in_data = groups_data.nunique()
            assert n_groups_in_data > 1, (
                f"Need at least 2 unique groups for group-wise splitting, but got {n_groups_in_data} unique groups from column(s)!"
            )
            num_folds = min(num_folds, n_groups_in_data)

        # Build dummy split data object
        splits_data = pd.Series(np.zeros(len(X)), index=X.index).to_frame()
        splits_data["group"] = groups_data
        stratify_on = None
        if self.stratify_on is not None:
            splits_data["stratify"] = stratify_on_data
            stratify_on = "stratify"
        splits_data = splits_data.reset_index(drop=True)

        custom_splits = get_recommended_grouped_splits(
            dataset=splits_data,
            group_on="group",
            stratify_on=stratify_on,
            n_splits=num_folds,
            n_repeats=num_repeats,
            test_size=None,
            group_labels=group_labels.value,
        )
        del splits_data

        custom_splits_for_index = []
        for repeat_splits in custom_splits.values():
            for train_idx, test_idx in repeat_splits.values():
                custom_splits_for_index.append((train_idx, test_idx))
        return custom_splits_for_index

    @staticmethod
    def group_on_to_groups_data(*, X: pd.DataFrame, group_on: str | list[str]):
        """Groups on to a unique group column."""
        # Get group label
        if isinstance(group_on, list):
            # If multiple columns are specified, create a combined group identifier
            groups_data = X[group_on].astype(str).agg("_".join, axis=1)
        else:
            groups_data = X[group_on]
        return groups_data.copy()

    @staticmethod
    def time_on_to_groups_data(
        *, X: pd.DataFrame, time_on: str, num_folds: int
    ) -> tuple[pd.Series, int]:
        """Go from time column to a group column for splits."""
        time_data = X[time_on]

        if pd.api.types.is_datetime64_any_dtype(time_data):
            time_data = time_data.view("int64")
        assert pd.api.types.is_numeric_dtype(time_data), (
            "Time_on column is not datetime or numeric!"
        )

        return split_time_index_into_intervals(
            time_data=time_data,
            goal_n_intervals=num_folds,
        )

    def get_num_group_instances(self, X: pd.DataFrame):
        """Compute the number of rows that represent how much (multi-instance) samples
        the data has. This is used to determine which splits to use.
        """
        from tabarena.benchmark.task.user_task import TabArenaTaskMetadataMixin

        return TabArenaTaskMetadataMixin.get_num_instance_groups(
            X=X,
            group_on=self.group_on,
            group_labels=self.group_labels,
        )


def split_time_index_into_intervals(
    *,
    time_data: pd.Series,
    goal_n_intervals: int,
    balance_on: str = "rows",
) -> tuple[pd.Series, int]:
    """Split a monotonically ordered time index into contiguous dynamic intervals.

    Rules:
    - Larger time values are always later in time
    - Equal time values are always assigned to the same interval
    - Intervals are created from the observed data, not equal-width spacing
    - Tries to create `goal_n_intervals`, but falls back to a smaller number if needed
    - Never returns fewer than 2 intervals

    Parameters
    ----------
    time_data : pd.Series
        Time input
    goal_n_intervals : int
        Desired number of intervals.
    balance_on : {"rows", "unique"}, default "rows"
        - "rows": balance intervals by number of rows
        - "unique": balance intervals by number of unique time values

    Returns:
    -------
    time_intervals: pd.Series
        Interval label for each row in the input time_data.
    actual_n_intervals : int
        Number of intervals actually used.
    """
    if goal_n_intervals < 2:
        raise ValueError("n_intervals must be at least 2.")
    if balance_on not in {"rows", "unique"}:
        raise ValueError("balance_on must be either 'rows' or 'unique'.")

    assert not time_data.isna().any(), "Time column contains nan values!"

    s = time_data.copy()

    # Aggregate identical time values so duplicates stay together
    counts = s.value_counts(dropna=False).sort_index().rename("row_count").to_frame()
    counts["unique_weight"] = 1

    n_unique = len(counts)
    if n_unique < 2:
        raise ValueError(
            "Need at least 2 unique time values to create at least 2 intervals."
        )
    actual_n_intervals = min(goal_n_intervals, n_unique)
    if actual_n_intervals < 2:
        raise ValueError("Could not create at least 2 intervals.")

    weight_col = "row_count" if balance_on == "rows" else "unique_weight"
    weights = counts[weight_col].to_numpy()

    # Greedy partition of sorted unique values into contiguous groups
    # aiming for equal cumulative weight per interval.
    total_weight = weights.sum()
    cut_positions = []
    start = 0
    cumulative = np.cumsum(weights)

    for group_num in range(1, actual_n_intervals):
        target = group_num * total_weight / actual_n_intervals

        # Candidate cut indices are between unique values:
        # cut after index j means next group starts at j+1
        min_j = start
        max_j = n_unique - (actual_n_intervals - group_num) - 1

        # Choose cut whose cumulative weight is closest to target
        candidates = np.arange(min_j, max_j + 1)
        j = candidates[np.argmin(np.abs(cumulative[candidates] - target))]
        cut_positions.append(j)
        start = j + 1

    # Assign interval labels to each unique time value
    interval_labels_for_unique = np.empty(n_unique, dtype=int)
    prev = 0
    for interval_id, cut in enumerate(cut_positions):
        interval_labels_for_unique[prev : cut + 1] = interval_id
        prev = cut + 1
    interval_labels_for_unique[prev:] = len(cut_positions)

    mapping = pd.Series(interval_labels_for_unique, index=counts.index)

    time_intervals = time_data.map(mapping)

    return time_intervals, actual_n_intervals
