from __future__ import annotations

import copy
from collections import defaultdict
from typing import Literal
from functools import reduce

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype
from typing_extensions import Self

from .abstract_repository import AbstractRepository
from .ensemble_mixin import EnsembleMixin
from .ground_truth_mixin import GroundTruthMixin
from .evaluation_repository import EvaluationRepository
from ..contexts.utils import prune_zeroshot_gt
from ..simulation.ground_truth import GroundTruth
from ..simulation.simulation_context import ZeroshotSimulatorContext


# TODO: Improve error message for overlap
class EvaluationRepositoryCollection(AbstractRepository, EnsembleMixin, GroundTruthMixin):
    """
    Repository collection class that implements core functionality related to
    fetching model predictions, available datasets, folds, etc.

    This class allows to merge a list of other repositories together into a single object.
    The merge operation is low-overhead and does not create any expensive objects.

    For example:
        repo1 has models A, B on datasets F, G
        repo2 has models B. C on datasets H
    Creating a repo collection with `repos=[repo1, repo2]` would then have access to:
        model A on datasets F, G
        model B on datasets F, G, H
        model C on datasets H

    Parameters
    ----------
    repos: list[EvaluationRepository | "EvaluationRepositoryCollection"]
        List of repos to merge.
    config_fallback: str, default None
        If specified, when a result is requested for a config on a task that has no result,
        instead of crashing, it will use the result from `config_fallback`.
        Recommended to use configs such as a constant predictor or a default random forest (cheap, robust, weak),
        otherwise weak models that fail on many tasks will actually benefit
        from crashing in terms of their rank during evaluation, which isn't ideal.
    overlap: Literal["raise", "first", "last"], default "raise"
        Determines the logic for handling (dataset, fold, config) overlaps in results in the specified `repos`.
        If "raise", will raise an exception if any overlap exists.
        If "first", will use the result from the repo earlier in the `repos` list.
        If "last", will use the result from the repo later in the `repos` list.
    """
    def __init__(
        self,
        repos: list[EvaluationRepository | "EvaluationRepositoryCollection"],
        config_fallback: str = None,
        overlap: Literal["raise", "first", "last"] = "raise",
    ):
        self.overlap = overlap
        self.repos: list[EvaluationRepository | "EvaluationRepositoryCollection"] = repos

        zeroshot_context = merge_zeroshot([repo._zeroshot_context for repo in self.repos])
        self._ground_truth: GroundTruth = merge_ground_truth([repo._ground_truth for repo in self.repos])
        self._mapping = self._compute_repo_mapping()
        super().__init__(zeroshot_context=zeroshot_context, config_fallback=config_fallback)

    def _compute_repo_mapping(self) -> dict[tuple[str, int, str], int]:
        repo_result_combinations = self._generate_dataset_fold_config_combinations(repos=self.repos)
        return self._combination_mapping_to_repo_index(repo_result_combinations=repo_result_combinations, overlap=self.overlap)

    def get_result_to_repo_idx(self, dataset: str, fold: int, config: str) -> int | None:
        """
        Returns the repo idx in `self.repos` containing the specified (dataset, fold, config) result.
        Returns None if no such repo exists.
        """
        return self._mapping.get((dataset, fold, config), None)

    def get_result_to_repo(self, dataset: str, fold: int, config: str) -> EvaluationRepository | "EvaluationRepositoryCollection" | None:
        """
        Returns the repo in `self.repos` containing the specified (dataset, fold, config) result.
        Returns None if no such repo exists.
        """
        repo_idx = self._mapping.get((dataset, fold, config), None)
        if repo_idx is None:
            return repo_idx
        else:
            return self.repos[repo_idx]

    def predict_test_multi(self, dataset: str, fold: int, configs: list[str] = None, binary_as_multiclass: bool = False) -> np.ndarray:
        return self._predict_multi(predict_func="predict_test_multi", dataset=dataset, fold=fold, configs=configs, binary_as_multiclass=binary_as_multiclass)

    def predict_val_multi(self, dataset: str, fold: int, configs: list[str] = None, binary_as_multiclass: bool = False) -> np.ndarray:
        return self._predict_multi(predict_func="predict_val_multi", dataset=dataset, fold=fold, configs=configs, binary_as_multiclass=binary_as_multiclass)

    def _predict_multi(self, predict_func: str, dataset: str, fold: int, configs: list[str] = None, binary_as_multiclass: bool = False) -> np.ndarray:
        if configs is None:
            configs = self.configs()

        repo_map = defaultdict(list)
        repo_idx_map = defaultdict(list)
        predict_map = dict()
        config_idx_lst = []
        for i, config in enumerate(configs):
            repo_idx = self.get_result_to_repo_idx(dataset=dataset, fold=fold, config=config)
            if repo_idx is None:
                if self._config_fallback is None:
                    raise ValueError(f"The following combination is not present in this repository: (dataset='{dataset}', fold={fold}, config='{config}')")
                # get fallback
                repo_idx = self.get_result_to_repo_idx(dataset=dataset, fold=fold, config=self._config_fallback)
                if repo_idx is None:
                    raise ValueError(
                        f"The following combination is not present in this repository: (dataset='{dataset}', fold={fold}, config='{config}')"
                        f"\nAdditionally, the fallback config is not present in (dataset='{dataset}', fold={fold}, config='{self._config_fallback}')"
                    )
                repo_map[repo_idx].append(self._config_fallback)
            else:
                repo_map[repo_idx].append(config)
            repo_idx_map[repo_idx].append(i)
            config_idx_lst.append(repo_idx)

        for repo_idx, config_lst in repo_map.items():
            f = getattr(self.repos[repo_idx], predict_func)
            predict_map[repo_idx] = f(dataset=dataset, fold=fold, configs=config_lst, binary_as_multiclass=binary_as_multiclass)

        # predict_map[repo_idx] has shape
        #  (n_configs, n_rows, n_classes) if multiclass classification
        #  (n_configs, n_rows, n_classes) if binary classification and binary_as_multiclass=True
        #  (n_configs, n_rows) otherwise
        # We ignore the first dimension because we need all configs in the final result,
        # and each repo could only have a subset of the configs.
        shape = predict_map[repo_idx].shape[1:]
        predict_multi = np.zeros(shape=(len(configs),) + shape, dtype=predict_map[repo_idx].dtype)

        for repo_idx, predict in predict_map.items():
            predict_multi[repo_idx_map[repo_idx], :] = predict

        return predict_multi

    def _subset_folds(self, folds: list[int]):
        super()._subset_folds(folds=folds)
        if self._ground_truth is not None:
            self._ground_truth.restrict_folds(folds=folds)
        for repo in self.repos:
            repo_folds = [f for f in repo.folds if f in folds]
            repo._subset_folds(folds=repo_folds)
        self._mapping = self._compute_repo_mapping()

    def _subset_datasets(self, datasets: list[str]):
        super()._subset_datasets(datasets=datasets)
        if self._ground_truth is not None:
            self._ground_truth.restrict_datasets(datasets=datasets)
        for repo in self.repos:
            repo_datasets = [d for d in repo.datasets() if d in datasets]
            repo._subset_datasets(datasets=repo_datasets)
        self._mapping = self._compute_repo_mapping()

    def force_to_dense(self, inplace: bool = False, verbose: bool = True) -> Self:
        """
        Method to force the repository to a dense representation inplace.

        The following operations will be applied in order:
        1. subset to only datasets that contain at least one result for all folds (self.n_folds())
        2. subset to only configs that have results in all tasks (configs that have results in every fold of every dataset)

        This will ensure that all datasets contain the same folds, and all tasks contain the same models.
        Calling this method when already in a dense representation will result in no changes.

        If you have different folds for different datasets or different configs for different datasets,
        this may result in an empty repository. Consider first calling `subset()` in this scenario.

        Parameters
        ----------
        inplace: bool, default = False
            If True, will perform logic inplace.
        verbose: bool, default = True
            Whether to log verbose details about the force to dense operation.

        Returns
        -------
        Return dense repo if inplace=False or self after inplace updates in this call.
        """
        if not inplace:
            return copy.deepcopy(self).force_to_dense(inplace=True, verbose=verbose)

        datasets = self.datasets(union=True)
        n_folds_per_dataset = pd.Series({dataset: len(dataset_folds) for dataset, dataset_folds in self._zeroshot_context.dataset_to_folds_dict.items()})

        n_folds = self.n_folds()
        datasets_dense = list(n_folds_per_dataset[n_folds_per_dataset == n_folds].index)

        # subset to only datasets that contain at least one result for all folds
        if set(datasets) != set(datasets_dense):
            self.subset(datasets=datasets_dense, inplace=inplace, force_to_dense=False)

        # subset to only configs that have results in all tasks
        configs = self.configs(union=True)
        configs_dense = self.configs(union=False)
        if set(configs) != set(configs_dense):
            self.subset(configs=configs_dense, inplace=inplace, force_to_dense=False)

        # subset all repos in collection so they do not contain invalid results
        for repo in self.repos:
            repo.subset(
                datasets=self.datasets(),
                configs=self.configs(),
                folds=self.folds,
                inplace=inplace,
                force_to_dense=False,
            )

        self._mapping = self._compute_repo_mapping()
        self._ground_truth = prune_zeroshot_gt(
            zeroshot_pred_proba=None,
            zeroshot_gt=self._ground_truth,
            dataset_to_tid_dict=self._dataset_to_tid_dict,
            verbose=verbose,
        )
        return self

    @staticmethod
    def _generate_dataset_fold_config_combinations(repos: list[EvaluationRepository]) -> list[list[tuple[str, int, str]]]:
        """
        Returns the combinations (dataset, fold, config) for each repository
        """
        return [repo.dataset_fold_config_pairs() for repo in repos]

    @staticmethod
    def _combination_mapping_to_repo_index(
        repo_result_combinations: list[list[tuple[str, int, str]]],
        overlap: Literal["raise", "first", "last"] = "raise",
    ) -> dict[tuple[str, int, str], int]:
        """
        Returns a dictionary mapping each (dataset, fold, config) to the repository index
        """
        if overlap == "first":
            len_combinations = len(repo_result_combinations)
            # traverse the repositories in reverse order to match `overlap` order
            mapping = {
                (dataset, fold, config): repo_index
                for repo_index in range(len_combinations-1, -1, -1)
                for (dataset, fold, config) in repo_result_combinations[repo_index]
            }
        elif overlap in ["last", "raise"]:
            mapping = {
                (dataset, fold, config): repo_index
                for repo_index, repo_combinations in enumerate(repo_result_combinations)
                for (dataset, fold, config) in repo_combinations
            }
            if overlap == "raise":
                len_combinations_total = 0
                for c in repo_result_combinations:
                    len_combinations_total += len(c)
                if len_combinations_total != len(mapping):
                    # TODO: Improve error message
                    raise AssertionError(f"Overlap detected in provided repositories! (overlap='{overlap}')")
        else:
            raise ValueError(f"Unknown overlap value: '{overlap}'")
        return mapping


def merge_zeroshot(zeroshot_contexts: list[ZeroshotSimulatorContext], require_matching_flags: bool = False) -> ZeroshotSimulatorContext:
    assert isinstance(zeroshot_contexts, list)

    df_baselines_lst = [z.df_baselines for z in zeroshot_contexts]
    df_baselines_lst = [df_baselines for df_baselines in df_baselines_lst if len(df_baselines) > 0]
    if df_baselines_lst:
        df_baselines = pd.concat(df_baselines_lst, ignore_index=True)
        df_baselines = df_baselines.drop_duplicates(ignore_index=True)
    else:
        df_baselines = None

    df_configs_lst = [z.df_configs for z in zeroshot_contexts]
    df_configs_lst = [df_configs for df_configs in df_configs_lst if len(df_configs) > 0]
    if df_configs_lst:
        df_configs = pd.concat(df_configs_lst, ignore_index=True)
        df_configs = df_configs.drop_duplicates(ignore_index=True)
    else:
        df_configs = None

    df_metadata_lst = [z.df_metadata for z in zeroshot_contexts]
    df_metadata_lst = [df_metadata for df_metadata in df_metadata_lst if df_metadata is not None and len(df_metadata) > 0]
    if df_metadata_lst:
        df_metadata = merge_metadata_frames(df_list=df_metadata_lst)
    else:
        df_metadata = None

    configs_hyperparameters = {}
    for z in zeroshot_contexts:
        if z.configs_hyperparameters is not None:
            intersection = set(configs_hyperparameters.keys()).intersection(set(z.configs_hyperparameters.keys()))
            for k in intersection:
                assert configs_hyperparameters[k] == z.configs_hyperparameters[k], (f"Inconsistent hyperparameters for config={k} found!\n"
                                                                                    f"Hyperparameters 1:\n"
                                                                                    f"\t{configs_hyperparameters[k]}\n"
                                                                                    f"Hyperparameters 2:\n"
                                                                                    f"\t{z.configs_hyperparameters[k]}")
            configs_hyperparameters.update(z.configs_hyperparameters)
    if len(configs_hyperparameters) == 0:
        configs_hyperparameters = None

    pct = zeroshot_contexts[0].pct
    score_against_only_baselines = zeroshot_contexts[0].score_against_only_baselines

    if require_matching_flags:
        for z in zeroshot_contexts[1:]:
            assert pct == z.pct, f"Inconsistent `pct` value found! ({pct}, {z.pct})"
        for z in zeroshot_contexts[1:]:
            assert score_against_only_baselines == z.score_against_only_baselines, f"Inconsistent `score_against_only_baselines` value found! ({score_against_only_baselines}, {z.score_against_only_baselines})"

    folds = None
    if any(z.folds is None for z in zeroshot_contexts):
        folds = None
    else:
        folds = []
        for z in zeroshot_contexts:
            folds += [f for f in z.folds if f not in folds]

    zeroshot_context_merged = ZeroshotSimulatorContext(
        df_configs=df_configs,
        df_baselines=df_baselines,
        df_metadata=df_metadata,
        configs_hyperparameters=configs_hyperparameters,
        folds=folds,
        pct=pct,
        score_against_only_baselines=score_against_only_baselines,
    )

    return zeroshot_context_merged


# TODO: Does not yet verify equivalence
def merge_ground_truth(ground_truths: list[GroundTruth]) -> GroundTruth:
    assert isinstance(ground_truths, list)
    ground_truths = [gt for gt in ground_truths if gt is not None]
    if len(ground_truths) == 0:
        return None

    label_test_dict = copy.copy(ground_truths[0]._label_test_dict)
    label_val_dict = copy.copy(ground_truths[0]._label_val_dict)
    for gt in ground_truths[1:]:
        datasets_gt = gt.datasets
        for d in datasets_gt:
            if d not in label_test_dict:
                label_test_dict[d] = {}
                label_val_dict[d] = {}
            label_test_dict[d].update(gt._label_test_dict[d])
            label_val_dict[d].update(gt._label_val_dict[d])
    ground_truth_merged = GroundTruth(label_test_dict=label_test_dict, label_val_dict=label_val_dict)
    return ground_truth_merged


def merge_metadata_frames(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge a list of metadata DataFrames such that:
    - Only well-defined instance key columns are used as join keys.
    - For non-key columns:
        * If one side is NaN and the other has a value, the value is used.
        * If both have the same non-NaN value, that value is used.
        * If both have different non-NaN values, this is treated as a conflict.
          (By default we raise; you can change to keep multiple rows if you want.)

    Result:
        - There is at most one row per instance unless there is a true conflict.
        - NaN vs value does NOT create duplicate rows.
    """
    if not df_list:
        return pd.DataFrame()

    # Decide your instance keys here.
    # This is often something like ['dataset', 'task', 'seed'] or similar.
    # For now, we use columns shared by all frames:
    join_cols = list(set.intersection(*(set(df.columns) for df in df_list)))
    if not join_cols:
        raise ValueError("No common columns found to use as join keys.")

    def merge_pair(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        overlapping_non_keys = (set(left.columns) & set(right.columns)) - set(join_cols)

        merged = pd.merge(
            left,
            right,
            on=join_cols,
            how="outer",
            suffixes=("_left", "_right"),
            sort=False,
        )

        for col in overlapping_non_keys:
            col_l = f"{col}_left"
            col_r = f"{col}_right"

            has_l = col_l in merged.columns
            has_r = col_r in merged.columns

            # Was this column ever integer-like in any input DF?
            orig_int_like = False
            if col in left.columns and is_integer_dtype(left[col].dtype):
                orig_int_like = True
            if col in right.columns and is_integer_dtype(right[col].dtype):
                orig_int_like = True

            if has_l and has_r:
                s_l = merged[col_l]
                s_r = merged[col_r]

                # 1) detect conflicts (both non-null and different)
                mask_both_notnull = s_l.notna() & s_r.notna()
                mask_conflict = mask_both_notnull & (s_l != s_r)
                if mask_conflict.any():
                    conflict_examples = merged.loc[mask_conflict, join_cols + [col_l, col_r]]
                    raise ValueError(
                        f"Conflict detected in column '{col}' for some instances.\n"
                        f"Examples:\n{conflict_examples}"
                    )

                # 2) coalesce WITHOUT combine_first to avoid FutureWarning
                s_out = s_l.copy()
                mask_na = s_out.isna()
                # where left is NaN, take right
                s_out[mask_na] = s_r[mask_na]
                merged[col] = s_out

                merged = merged.drop(columns=[col_l, col_r])

            elif has_l:
                merged = merged.rename(columns={col_l: col})
            elif has_r:
                merged = merged.rename(columns={col_r: col})

            # 3) If it was an integer column originally, try to cast back to nullable Int64
            if orig_int_like and col in merged.columns:
                s = merged[col]
                # pandas will often have upcast to float because of NaNs; check if values are integer-like
                if is_float_dtype(s.dtype):
                    non_na = s.dropna()
                    if (non_na % 1 == 0).all():
                        merged[col] = s.astype("Int64")

        return merged

    df_metadata = reduce(merge_pair, df_list)

    # Remove any exact duplicates that might remain
    df_metadata = df_metadata.drop_duplicates(ignore_index=True)

    return df_metadata
