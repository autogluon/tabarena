"""Grouped and IID train/validation splits for the task-specific inner validation protocol.

The split builders of Data Foundry's ``curation_recommendations`` module (data-foundry
``0.0.6.dev20261112094419``, Apache-2.0), which the inner validation of grouped and temporal tasks resolves its
folds with. Data Foundry dropped the module in ``0.0.6.dev20261112143209``. The split logic here is kept as it
was, so the folds stay those that every run since 2026-09-29 used; only the progress prints and the split-display
helper are left out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tabarena.benchmark.task.metadata import GroupLabelTypes

#: The seed of every split built here unless a caller passes ``random_state``: repeated k-fold, grouped k-fold
#: and the single train/test split. AutoGluon's learner seeds its structure-aware splits from its own
#: ``random_state`` (default 0), so a fit that lets AutoGluon build the splits reproduces these folds only with
#: this seed.
SPLIT_RANDOM_STATE = 4267

Splits = dict[int, dict[int, tuple[list[int], list[int]]]]


def get_recommended_iid_splits(
    *,
    dataset: pd.DataFrame,
    n_repeats: int,
    n_splits: int,
    test_size: int | None,
    stratify_on: str | None,
    random_state: int = SPLIT_RANDOM_STATE,
) -> Splits:
    """IID splits of ``dataset``: repeated (stratified) k-fold, or one train/test split when both counts are 1.

    Args:
        dataset: The rows to split, with a ``RangeIndex`` from 0.
        n_repeats: Number of repeats.
        n_splits: Number of folds per repeat.
        test_size: Size of the test side of the single train/test split; unused for k-fold.
        stratify_on: Column to stratify on, or None.
        random_state: Seed of the k-fold splitter and of the single split.

    Returns:
        ``{repeat: {fold: (train_indices, test_indices)}}``.
    """
    from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, train_test_split

    if not dataset.index.equals(pd.RangeIndex(start=0, stop=len(dataset))):
        raise ValueError("Dataset index must be a RangeIndex starting from 0 (do reset_index!).")

    y = dataset[stratify_on] if stratify_on is not None else None
    splits: Splits = {}

    if n_repeats == 1 and n_splits == 1:
        train_indices, test_indices = train_test_split(
            dataset.index,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
        splits[0] = {0: (train_indices.tolist(), test_indices.tolist())}
        return splits

    if stratify_on is not None:
        rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    else:
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    for split_i, (train_idx, test_idx) in enumerate(rkf.split(X=dataset, y=y)):
        repeat_i = split_i // n_splits
        fold_i = split_i % n_splits
        splits.setdefault(repeat_i, {})[fold_i] = (train_idx.tolist(), test_idx.tolist())
    return splits


def get_recommended_grouped_splits(
    *,
    dataset: pd.DataFrame,
    n_repeats: int,
    n_splits: int,
    group_on: str,
    group_labels: GroupLabelTypes | str,
    test_size: int | None,
    stratify_on: str | None,
    random_state: int = SPLIT_RANDOM_STATE,
) -> Splits:
    """Group-disjoint splits of ``dataset``, by one of two pathways.

    With ``group_labels="per_group"`` every group carries one value of the stratification column, and the groups
    themselves are split IID (group sizes are ignored). With ``"per_sample"`` sklearn's ``GroupKFold`` or
    ``StratifiedGroupKFold`` splits the rows, which accounts for group sizes but stratifies only approximately.

    Args:
        dataset: The rows to split, with a ``RangeIndex`` from 0.
        n_repeats: Number of repeats.
        n_splits: Number of folds per repeat; with ``n_repeats == n_splits == 1`` one train/test split.
        group_on: Column holding the group of each row.
        group_labels: ``"per_group"`` or ``"per_sample"``.
        test_size: Approximate number of test rows of the single train/test split; None for k-fold.
        stratify_on: Column to stratify on, or None.
        random_state: Seed of the splitter (repeat ``i`` of the per-sample path uses ``random_state + i``).

    Returns:
        ``{repeat: {fold: (train_indices, test_indices)}}``.
    """
    if not dataset.index.equals(pd.RangeIndex(start=0, stop=len(dataset))):
        raise ValueError("Dataset index must be a RangeIndex starting from 0 (do reset_index!).")

    if n_repeats == 1 and n_splits == 1 and (test_size is None or test_size <= 0):
        raise ValueError("test_size must be a positive integer for single train-test split!")

    if (group_labels == "per_group") and (stratify_on is not None):
        # nunique() ignores NaN, as the replicated check did.
        is_per_group = (dataset.groupby(group_on, observed=True)[stratify_on].nunique() == 1).all()  # noqa: PD101
        if not is_per_group:
            raise ValueError(
                "group_labels is set to 'per_group', but not all groups have "
                "a single unique value for the stratify column!"
            )

    split_fn = (
        _get_grouped_splits_via_groupkfold if group_labels == "per_sample" else _get_grouped_splits_via_index_split
    )
    return split_fn(
        dataset=dataset,
        n_repeats=n_repeats,
        n_splits=n_splits,
        group_on=group_on,
        test_size=test_size,
        stratify_on=stratify_on,
        random_state=random_state,
    )


def _get_grouped_splits_via_index_split(
    *,
    dataset: pd.DataFrame,
    n_repeats: int,
    n_splits: int,
    group_on: str,
    test_size: int | None,
    stratify_on: str | None,
    random_state: int = SPLIT_RANDOM_STATE,
) -> Splits:
    """Split the groups IID and map each split back to their rows; group sizes play no part."""
    group_values: list[str | int] = []
    group_samples: list[list[int]] = []
    group_stratify: list[object] = []
    multi_label_group_found = False

    for group_value, group_df in dataset.groupby(group_on, sort=False, observed=True):
        group_values.append(group_value)
        group_samples.append(group_df.index.tolist())
        if stratify_on is not None:
            unique_values = list(group_df[stratify_on].unique())
            group_stratify.append(unique_values[0])
            if len(unique_values) > 1:
                multi_label_group_found = True

    if multi_label_group_found:
        raise ValueError("Multi-label group found but: group_labels='per_group'!")

    group_dataset = pd.DataFrame({group_on: group_values})
    if stratify_on is not None:
        group_dataset[stratify_on] = group_stratify
    group_dataset = group_dataset.reset_index(drop=True)

    if test_size is not None:
        # Rescale the test size from rows to groups by the average group size.
        avg_samples_per_group = np.mean([len(samples) for samples in group_samples])
        test_size = int(test_size // avg_samples_per_group)

    group_splits = get_recommended_iid_splits(
        dataset=group_dataset,
        n_repeats=n_repeats,
        n_splits=n_splits,
        test_size=test_size,
        stratify_on=stratify_on,
        random_state=random_state,
    )

    def map_group_indices(indices: list[int]) -> list[int]:
        mapped: list[int] = []
        for group_idx in indices:
            mapped.extend(group_samples[group_idx])
        return mapped

    mapped_splits: Splits = {}
    all_groups = set(dataset[group_on].unique())
    for repeat_i, folds in group_splits.items():
        mapped_splits[repeat_i] = {}
        for fold_i, (train_group_idxs, test_group_idxs) in folds.items():
            train_idxs = map_group_indices(train_group_idxs)
            test_idxs = map_group_indices(test_group_idxs)
            train_groups = set(dataset.iloc[train_idxs][group_on].unique())
            test_groups = set(dataset.iloc[test_idxs][group_on].unique())
            assert train_groups.isdisjoint(test_groups)
            assert train_groups.union(test_groups) == all_groups
            mapped_splits[repeat_i][fold_i] = (train_idxs, test_idxs)
    return mapped_splits


def _get_grouped_splits_via_groupkfold(
    *,
    dataset: pd.DataFrame,
    n_repeats: int,
    n_splits: int,
    group_on: str,
    test_size: int | None,
    stratify_on: str | None,
    random_state: int = SPLIT_RANDOM_STATE,
) -> Splits:
    """Split the rows with sklearn's (stratified) group k-fold; one fold of a size-matched k-fold for one split."""
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
    from sklearn.utils.multiclass import type_of_target

    y = dataset[stratify_on] if stratify_on is not None else None
    group = dataset[group_on]
    splitter_cls = StratifiedGroupKFold if stratify_on is not None else GroupKFold

    if stratify_on:
        target_type = type_of_target(y)
        if target_type not in ("binary", "multiclass"):
            raise ValueError(
                f"StratifiedGroupKFold only supports binary and multiclass targets, but got {target_type}!",
                "Solution: Cast your target to 'category' dtype with 2 or more unique values for stratification.",
            )

    if n_repeats == 1 and n_splits == 1:
        if test_size is None or test_size <= 0:
            raise ValueError("test_size must be a positive integer for single train-test split!")
        n_groups = group.nunique()
        group_test_size = test_size / group.value_counts().mean()
        approximate_splits = int(max(2, min(n_groups, round(n_groups / group_test_size))))
        splitter_inst = splitter_cls(n_splits=approximate_splits, shuffle=True, random_state=random_state)
        train_index, test_index = next(splitter_inst.split(X=dataset, y=y, groups=group))
        return {0: {0: (train_index.tolist(), test_index.tolist())}}

    splits: Splits = {}
    for repeat_i in range(n_repeats):
        splits[repeat_i] = {}
        splitter_inst = splitter_cls(n_splits=n_splits, random_state=random_state + repeat_i, shuffle=True)
        for fold_idx, (train_index, test_index) in enumerate(splitter_inst.split(X=dataset, y=y, groups=group)):
            splits[repeat_i][fold_idx] = (train_index.tolist(), test_index.tolist())
    return splits
