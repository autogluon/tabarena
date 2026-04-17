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
    tiny_data_num_repeats: int = 5
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
        drop_group_columns: bool = True,
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
        drop_group_columns:
            If True (default), drop group_on from the training (and tuning) data after
            using them to compute the splits. These columns are used solely for defining
            the validation protocol and should not be fed to the model as features.
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
        self.drop_group_columns = drop_group_columns

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
            stratify_on_data = X[self.stratify_on] if self.stratify_on in X.columns else y
            # Enforce categorical dtype for stratification column, as some splitting logic relies on it.
            stratify_on_data = stratify_on_data.astype("category")

        groups_data = None
        group_labels = None
        if (self.time_on is not None) and (self.group_on is not None):
            raise NotImplementedError

        if self.time_on is not None:
            groups_data, num_folds_new = self.time_on_to_groups_data(X=X, time_on=self.time_on, num_folds=num_folds)
            num_repeats = 1
            logger.info(
                f"\n\tFolds time-based grouping: before={num_folds}; after={num_folds_new}\n\tnum_repeats set to 1!"
            )
            num_folds = num_folds_new
            # Set group labels as needed for time split
            group_labels = GroupLabelTypes.PER_SAMPLE

        if self.group_on is not None:
            groups_data = self.group_on_to_groups_data(X=X, group_on=self.group_on)
            group_labels = self.group_labels

        if groups_data is not None:
            if num_repeats is None:
                num_repeats = 1

            n_groups = groups_data.nunique()
            if n_groups < num_folds:
                logger.info(
                    f"Number of unique groups in the data ({n_groups}) is less than the "
                    f"number of folds ({num_folds})! Adjusting the number of folds to be equal to the number of "
                    f"unique groups, and setting num_repeats to 1."
                )
                num_folds = n_groups
                num_repeats = 1

            if stratify_on_data is not None:
                n_samples_minority_class = int(stratify_on_data.value_counts().min())
                if n_samples_minority_class < num_folds:
                    logger.warning(
                        f"Number of samples in minority class for stratification ({n_samples_minority_class}) is less "
                        f"than the number of folds ({num_folds})! We set num_folds to be equal to the number of samples"
                        " in the minority class, and num_repeats to 1."
                    )
                    num_folds = n_samples_minority_class
                    num_repeats = 1

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

                assert len(train_idx) > 0, "Train split is empty!"
                assert len(test_idx) > 0, "Test split is empty!"

                if stratify_on_data is not None:
                    stratify_values = set(stratify_on_data.unique())
                    train_stratify_values = set(stratify_on_data.iloc[train_idx].unique())
                    test_stratify_values = set(stratify_on_data.iloc[test_idx].unique())

                    assert train_stratify_values == stratify_values, (
                        "[Missing Train Stratification Values] "
                        "Stratification values in train split do not match overall stratification values!"
                        f"\n\tOverall stratification values: {stratify_values}"
                        f"\n\tTrain stratification values: {train_stratify_values}"
                    )
                    assert test_stratify_values.issubset(train_stratify_values), (
                        "[Unseen Test Stratification Values] "
                        "Stratification values in test split are not a subset of train stratification values!"
                        f"\n\tTrain stratification values: {train_stratify_values}"
                        f"\n\tTest stratification values: {test_stratify_values}"
                    )

                    if train_stratify_values != stratify_values:
                        # Check if test has all labels for binary, as metrics require it.
                        if len(stratify_values) == 2:
                            raise ValueError(
                                "[Binary Metric Missing Stratification Values in Test] "
                                "Stratification values in train and test splits do not match!"
                                f"\n\tOverall stratification values: {stratify_values}"
                                f"\n\tTrain stratification values: {train_stratify_values}"
                                f"\n\tTest stratification values: {test_stratify_values}"
                            )

                        # For multi-stratify values, we do not allow missing a stratify value in the test split
                        logger.warning(
                            "[Stratification Value Missing in Test Data] "
                            "Stratification values in train and test splits are not identical."
                            "This means the validation data is likely missing some classes."
                            f"\n\tOverall stratification values: {stratify_values}"
                            f"\n\tTrain stratification values: {train_stratify_values}"
                            f"\n\tTest stratification values: {test_stratify_values}"
                        )

        return custom_splits, num_folds, num_repeats

    def _resolve_number_of_splits(
        self, *, num_folds: int, num_repeats: int | None, num_group_instances: int
    ) -> tuple[int, int | None]:
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
        new_num_folds_reason, new_num_repeats_reason = "", ""
        if num_group_instances <= self.max_samples_for_tiny_data:
            new_num_folds = self.tiny_data_num_folds
            new_num_repeats = self.tiny_data_num_repeats
            new_num_folds_reason += "Tiny data"
            new_num_repeats_reason += "Tiny data"
        else:
            # We want these by default for all other data in our benchmark.
            assert num_folds == 8
            assert (num_repeats == 1) or (num_repeats is None)



        if new_num_folds is not None:
            logger.info(
                f"\nUpdating num_bag_folds from {num_folds} to {new_num_folds} "
                f"because: {new_num_folds_reason}"
            )
            num_folds = new_num_folds

        if new_num_repeats is not None:
            logger.info(
                f"\nUpdating new_num_repeats from {num_repeats} to {new_num_repeats}"
                f"because: {new_num_repeats_reason}"
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
            group_labels=group_labels,
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
    def time_on_to_groups_data(*, X: pd.DataFrame, time_on: str, num_folds: int) -> tuple[pd.Series, int]:
        """Go from time column to a group column for splits."""
        time_data = X[time_on]

        if pd.api.types.is_datetime64_any_dtype(time_data):
            time_data = time_data.astype("int64")
        assert pd.api.types.is_numeric_dtype(time_data), "Time_on column is not datetime or numeric!"

        return split_time_index_into_intervals(
            time_data=time_data,
            goal_n_intervals=num_folds,
        )

    def _get_group_columns_to_drop(self) -> list[str]:
        """Return the group/time columns that should be dropped from the model input.

        Only populated when ``drop_group_columns=True``.  The caller is responsible
        for dropping these columns *after* the splits have been computed (the columns
        are still needed by ``resolve_validation_splits``).
        """
        if not self.drop_group_columns:
            return []
        cols: list[str] = []
        if self.group_on is not None:
            cols += self.group_on if isinstance(self.group_on, list) else [self.group_on]
        return cols

    def get_num_group_instances(self, X: pd.DataFrame, *, group_labels: None = None) -> int:
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
        raise ValueError("Need at least 2 unique time values to create at least 2 intervals.")
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
