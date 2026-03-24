from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


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

        if self.group_on is not None:
            custom_splits = self._resolve_group_splits(
                X=X,
                num_folds=num_folds,
                num_repeats=num_repeats,
                stratify_on_data=stratify_on_data,
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
        # FIXME: do not hardcode this in the future.
        new_num_folds, new_num_repeats = None, None
        if num_group_instances <= 500:
            new_num_folds = 5
            new_num_repeats = 3
        else:
            # We want these by default for all other data in our benchmark.
            assert num_folds == 8
            assert num_repeats == 1

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

        # Get group label
        group_col = self.group_on
        if isinstance(group_col, list):
            # If multiple columns are specified, create a combined group identifier
            groups_data = X[group_col].astype(str).agg("_".join, axis=1)
        else:
            groups_data = X[group_col]

        assert not groups_data.isna().any(), "Group column(s) contain NaN values, which is not allowed for group-wise splitting!"

        n_groups_in_data = groups_data.nunique()
        assert n_groups_in_data > 1, (
            f"Need at least 2 unique groups for group-wise splitting, but got {n_groups_in_data} unique groups from column(s) {group_col}!"
        )
        num_folds = min(num_folds, n_groups_in_data)

        splits_data = pd.Series(np.zeros(len(X)), index=X.index).to_frame()
        splits_data["group"] = groups_data

        stratify_on = None
        if self.stratify_on is not None:
            splits_data["stratify"] = stratify_on_data
            stratify_on = "stratify"

        # Ensure correct indexing for splits
        splits_data = splits_data.reset_index(drop=True)

        custom_splits = get_recommended_grouped_splits(
            dataset=splits_data,
            group_on="group",
            stratify_on=stratify_on,
            n_splits=num_folds,
            n_repeats=num_repeats,
            test_size=None,
        )
        del splits_data

        custom_splits_for_index = []
        for repeat_splits in custom_splits.values():
            for train_idx, test_idx in repeat_splits.values():
                custom_splits_for_index.append((train_idx, test_idx))
        return custom_splits_for_index

    def get_num_group_instances(self, X: pd.DataFrame):
        """Compute the number of rows that represent how much (multi-instance) samples
        the data has. This is used to determine which splits to use.
        """
        from tabarena.benchmark.task.user_task import TabArenaTaskMetadataMixin

        return TabArenaTaskMetadataMixin.get_num_instance_groups(
            X=X, group_on=self.group_on
        )
