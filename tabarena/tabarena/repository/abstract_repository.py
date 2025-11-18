from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from typing_extensions import Self

from ..benchmark.task.openml import OpenMLTaskWrapper
from ..simulation.simulation_context import ZeroshotSimulatorContext
from ..simulation.single_best_config_scorer import SingleBestConfigScorer
from ..utils.cache import SaveLoadMixin


class AbstractRepository(ABC, SaveLoadMixin):
    def __init__(
        self,
        zeroshot_context: ZeroshotSimulatorContext,
        config_fallback: str | None = None
    ):
        self._zeroshot_context = zeroshot_context
        self._config_fallback = None
        self.set_config_fallback(config_fallback)

    def set_config_fallback(self, config_fallback: str = None):
        if config_fallback:
            assert config_fallback in self.configs()
        self._config_fallback = config_fallback

    def print_info(self):
        self._zeroshot_context.print_info()

    @property
    def _dataset_to_tid_dict(self) -> Dict[str, int]:
        return self._zeroshot_context.dataset_to_tid_dict

    @property
    def _tid_to_dataset_dict(self) -> Dict[int, str]:
        return {v: k for k, v in self._dataset_to_tid_dict.items()}

    # TODO: tasks
    def subset(self,
               *,
               datasets: List[str] = None,
               folds: List[int] = None,
               configs: List[str] = None,
               baselines: List[str] = None,
               problem_types: List[str] = None,
               force_to_dense: bool = False,
               inplace: bool = False,
               verbose: bool = True,
               ) -> Self:
        """
        Method to subset the repository object and force to a dense representation.

        :param datasets: The list of datasets to subset. Ignored if unspecified.
        :param folds: The list of folds to subset. Ignored if unspecified.
        :param configs: The list of configs to subset. Ignored if unspecified.
        :param baselines: The list of baselines to subset. Ignored if unspecified.
        :param problem_types: The list of problem types to subset. Ignored if unspecified.
        :param force_to_dense: If True, forces the output to dense representation.
        :param inplace: If True, will perform subset logic inplace.
        :param verbose: Whether to log verbose details about the force to dense operation.
        :return: Return subsetted repo if inplace=False or self after inplace updates in this call.
        """
        if not inplace:
            return copy.deepcopy(self).subset(
                datasets=datasets,
                folds=folds,
                configs=configs,
                baselines=baselines,
                problem_types=problem_types,
                force_to_dense=force_to_dense,
                inplace=True,
                verbose=verbose,
            )
        if folds is not None:
            self._subset_folds(folds=folds)
        if configs is not None:
            self._zeroshot_context.subset_configs(configs=configs)
        if baselines is not None:
            self._zeroshot_context.subset_baselines(baselines=baselines)
        if datasets is not None:
            self._subset_datasets(datasets=datasets)
        if problem_types is not None:
            datasets_problem_type = self.datasets(problem_type=problem_types)
            self._subset_datasets(datasets=datasets_problem_type)
        if force_to_dense:
            self.force_to_dense(inplace=True, verbose=verbose)
        return self

    def _subset_folds(self, folds: list[int]):
        self._zeroshot_context.subset_folds(folds=folds)

    def _subset_datasets(self, datasets: list[str]):
        self._zeroshot_context.subset_datasets(datasets=datasets)

    @abstractmethod
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
        raise NotImplementedError

    @property
    def _df_metadata(self) -> pd.DataFrame:
        return self._zeroshot_context.df_metadata

    @property
    def task_metadata(self) -> pd.DataFrame:
        return self._df_metadata

    def tids(self, problem_type: str = None) -> List[int]:
        """
        Note: returns the taskid of the datasets rather than the string name.

        :param problem_type: If specified, only datasets with the given problem_type are returned.
        """
        return self._zeroshot_context.get_tids(problem_type=problem_type)

    def datasets(self, *, configs: list[str] = None, problem_type: str | list[str] = None, union: bool = True) -> list[str]:
        """repo_subset2.datasets()
        Return all valid datasets.
        By default, will return all datasets that appear in any config at least once.

        Parameters
        ----------
        problem_type : str | list[str], default = None
            If specified, will only consider datasets with the given problem type
        union: bool, default = True
            If True, will return the union of datasets present in each config.
            If False, will return the intersection of datasets present in each config.

        Returns
        -------
        A list of dataset names satisfying the above conditions.
        """
        return self._zeroshot_context.get_datasets(configs=configs, problem_type=problem_type, union=union)

    def tasks(self) -> list[tuple[str, int]]:
        dataset_folds = self._zeroshot_context.get_tasks(as_dataset_fold=True)
        return dataset_folds

    def configs(
        self,
        *,
        datasets: list[str] = None,
        tasks: list[tuple[str, int]] = None,
        config_types: list[str] = None,
        union: bool = True,
    ) -> list[str]:
        """
        Return all valid configs.
        By default, will return all configs that appear in any task at least once.

        Parameters
        ----------
        datasets: list[str], default = None
            If specified, will only consider the configs present in the given datasets.
        tasks: list[tuple[str, int]], default = None
            If specified, will only consider the configs present in the given tasks.
            Tasks are in the form `(dataset, fold)`.
            For example, `("abalone", 1)`.
        config_types: list[str], default = None
            If specified, will only consider the configs with a config type in `config_types`.
        union: bool, default = True
            If True, will return the union of configs present in each task.
            If False, will return the intersection of configs present in each task.

        Returns
        -------
        A list of config names satisfying the above conditions.
        """
        return self._zeroshot_context.get_configs(datasets=datasets, tasks=tasks, config_types=config_types, union=union)

    # TODO: unit test
    def baselines(self, *, datasets: list[str] = None, tasks: list[tuple[str, int]] = None, union: bool = True) -> list[str]:
        """
        Return all valid baselines.
        By default, will return all baselines that appear in any task at least once.

        Parameters
        ----------
        datasets : list[str], default = None
            If specified, will only consider the baselines present in the given datasets.
        tasks: list[tuple[str, int]], default = None
            If specified, will only consider the baselines present in the given tasks.
            Tasks are in the form `(dataset, fold)`.
            For example, `("abalone", 1)`.
        union: bool, default = True
            If True, will return the union of baselines present in each task.
            If False, will return the intersection of baselines present in each task.

        Returns
        -------
        A list of baseline names satisfying the above conditions.
        """
        return self._zeroshot_context.get_baselines(datasets=datasets, tasks=tasks, union=union)

    def dataset_to_tid(self, dataset: str) -> int:
        return self._dataset_to_tid_dict[dataset]

    def datasets_to_tids(self, datasets: list[str] = None) -> pd.Series:
        if datasets is None:
            datasets = self.datasets()
        return pd.Series({dataset: self._dataset_to_tid_dict[dataset] for dataset in datasets}, name="tid")

    def tid_to_dataset(self, tid: int) -> str:
        return self._tid_to_dataset_dict[tid]

    def metrics(
        self,
        datasets: List[str] = None,
        folds: List[int] = None,
        tasks: list[tuple[str, int]] = None,
        configs: List[str] = None,
        set_index: bool = True,
    ) -> pd.DataFrame:
        """
        :param datasets:
        :param folds:
        :param configs: list of configs to query metrics
        :return: pd.DataFrame of metrics for each dataset-fold-framework combination.

        Output is a multi-index pandas DataFrame ("dataset", "fold", "framework").
        Each row is a result for a particular config on a given task.
        If a config does not have a result for a given task, it will not have a row present in the DataFrame for that task.
        Columns:
            metric_error : Test error of the config
            metric_error_val : Validation error of the config
            time_train_s : Training time of the config
            time_infer_s : Inference time of the config
            rank : Rank of the config
        """
        df = self._zeroshot_context.df_configs_ranked
        if tasks is not None or configs is not None:
            df = self._zeroshot_context._filter_df_by_datasets(df=df, tasks=tasks, configs=configs)

        masks = []
        if datasets is not None:
            masks.append(df["dataset"].isin(datasets))
        if folds is not None:
            masks.append(df["fold"].isin(folds))
        if masks:
            mask = masks[0]
            for m in masks[1:]:
                mask &= m
            df = df[mask]

        if set_index:
            df = df.set_index(["dataset", "fold", "framework"], drop=True)[
                ["metric_error", "metric_error_val", "time_train_s", "time_infer_s", "rank"]
            ]

        return df

    def dataset_fold_config_pairs(self, datasets: List[str] = None, folds: List[int] = None, configs: List[str] = None) -> list:
        """
        Returns a list of all (dataset, fold, config) tuples available in the repo.

        Parameters
        ----------
        datasets: List[str], default None
            Filters the output to only contain datasets in the specified list.
        folds: List[int], default None
            Filters the output to only contain folds in the specified list.
        configs: List[str], default None
            Filters the output to only contain configs in the specified list.
        """
        return self.metrics(datasets=datasets, folds=folds, configs=configs).index.tolist()

    def predict_test(self, dataset: str, fold: int, config: str, binary_as_multiclass: bool = False) -> np.ndarray:
        """
        Returns the predictions on the test set for a given configuration on a given dataset and fold

        Parameters
        ----------
        dataset: str
            The dataset to get predictions from. Must be a value in `self.datasets()`.
        fold: int
            The fold of the dataset to get predictions from.
        config: str
            The model config to get predictions from. Must be a value in `self.configs()`.
        binary_as_multiclass: bool, default = False
            If True, will return binary predictions in shape (n_rows, n_classes).
            If False, will return binary predictions in shape (n_rows), with the value being class 1 (the positive class).

            You can convert from (n_rows, n_classes) -> (n_rows) via `predictions[:, 1]`.
            You can convert from (n_rows) -> (n_rows, n_classes) via `np.stack([1 - predictions, predictions], axis=predictions.ndim)`.

            The internal representation is of form (n_rows) as it requires less memory,
            so there is a conversion overhead introduced when `binary_as_multiclass=True`.

        Returns
        -------
        The model predictions on the test set with shape (n_rows, n_classes) for multiclass or (n_rows) in case of regression.
        For binary, shape depends on `binary_as_multiclass` value.
        """
        return self.predict_test_multi(dataset=dataset, fold=fold, configs=[config], binary_as_multiclass=binary_as_multiclass).squeeze()

    def predict_val(self, dataset: str, fold: int, config: str, binary_as_multiclass: bool = False) -> np.ndarray:
        """
        Parameters
        ----------
        dataset: str
            The dataset to get predictions from. Must be a value in `self.datasets()`.
        fold: int
            The fold of the dataset to get predictions from.
        config: str
            The model config to get predictions from. Must be a value in `self.configs()`.
        binary_as_multiclass: bool, default = False
            If True, will return binary predictions in shape (n_rows, n_classes).
            If False, will return binary predictions in shape (n_rows), with the value being class 1 (the positive class).

            You can convert from (n_rows, n_classes) -> (n_rows) via `predictions[:, 1]`.
            You can convert from (n_rows) -> (n_rows, n_classes) via `np.stack([1 - predictions, predictions], axis=predictions.ndim)`.

            The internal representation is of form (n_rows) as it requires less memory,
            so there is a conversion overhead introduced when `binary_as_multiclass=True`.

        Returns
        -------
        The model predictions on the validation set with shape (n_rows, n_classes) for multiclass or (n_rows) in case of regression.
        For binary, shape depends on `binary_as_multiclass` value.
        """
        return self.predict_val_multi(dataset=dataset, fold=fold, configs=[config], binary_as_multiclass=binary_as_multiclass).squeeze()

    @abstractmethod
    def predict_test_multi(self, dataset: str, fold: int, configs: List[str] = None, binary_as_multiclass: bool = False) -> np.ndarray:
        """
        Returns the predictions on the test set for a given list of configurations on a given dataset and fold

        Parameters
        ----------
        dataset: str
            The dataset to get predictions from. Must be a value in `self.datasets()`.
        fold: int
            The fold of the dataset to get predictions from.
        configs: List[str], default = None
            The model configs to get predictions from.
            If None, will use `self.configs()`.
        binary_as_multiclass: bool, default = False
            If True, will return binary predictions in shape (n_configs, n_rows, n_classes).
            If False, will return binary predictions in shape (n_configs, n_rows), with the value being class 1 (the positive class).

            You can convert from (n_configs, n_rows, n_classes) -> (n_configs, n_rows) via `predictions[:, :, 1]`.
            You can convert from (n_configs, n_rows) -> (n_configs, n_rows, n_classes) via `np.stack([1 - predictions, predictions], axis=predictions.ndim)`.

            The internal representation is of form (n_configs, n_rows) as it requires less memory,
            so there is a conversion overhead introduced when `binary_as_multiclass=True`.

        Returns
        -------
        The model predictions with shape (n_configs, n_rows, n_classes) for multiclass or (n_configs, n_rows) in case of regression.
        For binary, shape depends on `binary_as_multiclass` value.
        The output order will be the same order as `configs`.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_val_multi(self, dataset: str, fold: int, configs: List[str] = None, binary_as_multiclass: bool = False) -> np.ndarray:
        """
        Returns the predictions on the validation set for a given list of configurations on a given dataset and fold

        Parameters
        ----------
        dataset: str
            The dataset to get predictions from. Must be a value in `self.datasets()`.
        fold: int
            The fold of the dataset to get predictions from.
        configs: List[str], default = None
            The model configs to get predictions from.
            If None, will use `self.configs()`.
        binary_as_multiclass: bool, default = False
            If True, will return binary predictions in shape (n_configs, n_rows, n_classes).
            If False, will return binary predictions in shape (n_configs, n_rows), with the value being class 1 (the positive class).

            You can convert from (n_configs, n_rows, n_classes) -> (n_configs, n_rows) via `predictions[:, :, 1]`.
            You can convert from (n_configs, n_rows) -> (n_configs, n_rows, n_classes) via `np.stack([1 - predictions, predictions], axis=predictions.ndim)`.

            The internal representation is of form (n_configs, n_rows) as it requires less memory,
            so there is a conversion overhead introduced when `binary_as_multiclass=True`.

        Returns
        -------
        The model predictions with shape (n_configs, n_rows, n_classes) for multiclass or (n_configs, n_rows) in case of regression.
        For binary, shape depends on `binary_as_multiclass` value.
        The output order will be the same order as `configs`.
        """
        raise NotImplementedError

    def dataset_metadata(self, dataset: str) -> dict:
        metadata = self._df_metadata[self._df_metadata["dataset"] == dataset]
        return dict(zip(metadata.columns, metadata.values[0]))

    def dataset_info(self, dataset: str) -> dict:
        """
        Parameters
        ----------
        dataset: str

        Returns
        -------
        Dictionary with two keys:
            "metric": The evaluation metric name used for scoring on the dataset
            "problem_type": The problem type of the dataset
        """
        return self._zeroshot_context.df_metrics.loc[dataset].to_dict()

    def datasets_info(self, datasets: list[str] = None) -> pd.DataFrame:
        """
        Parameters
        ----------
        datasets: list[str]. default = None
            If None, uses all datasets

        Returns
        -------
        Pandas DataFrame with index "dataset" and two columns:
            "metric": The evaluation metric name used for scoring on the dataset
            "problem_type": The problem type of the dataset
        """
        if datasets is None:
            datasets = self.datasets()
        return self._zeroshot_context.df_metrics.loc[datasets]

    @property
    def folds(self) -> List[int]:
        """Folds with any result"""
        return sorted(self._zeroshot_context.folds)

    def n_folds(self) -> int:
        """Number of folds with any result"""
        return len(self.folds)

    def n_datasets(self) -> int:
        """Number of datasets with any result"""
        return len(self.datasets())

    def n_configs(self) -> int:
        """Number of configs with any result"""
        return len(self.configs())

    def task_name_from_tid(self, tid: int, fold: int) -> str:
        """Returns the task associated with a (tid, fold)"""
        return self._zeroshot_context.task_name_from_tid(tid=tid, fold=fold)

    def task_name(self, dataset: str, fold: int) -> str:
        """Returns the task associated with a (dataset, fold)"""
        return self.task_name_from_tid(tid=self.dataset_to_tid(dataset), fold=fold)

    def task_to_dataset(self, task: str) -> str:
        """Returns the dataset associated with a task"""
        return self._zeroshot_context.task_to_dataset_dict[task]

    def task_to_fold(self, task: str) -> int:
        """Returns the fold associated with a task"""
        return self._zeroshot_context.task_to_fold(task=task)

    def dataset_to_folds(self, dataset: str) -> list[int]:
        return self._zeroshot_context.dataset_to_folds(dataset=dataset)

    def config_hyperparameters(self, config: str, include_ag_args: bool = True) -> dict | None:
        """
        Returns config hyperparameters as defined in AutoGluon
        If no hyperparameters exist for the config, return None

        Parameters
        ----------
        config: str
            The config to get hyperparameters for
        include_ag_args: bool, default = True
            If True, includes the `ag_args` hyperparameter which is used in determining the name of the model in AutoGluon
        """
        config_hyperparameters = self._zeroshot_context.get_config_hyperparameters(config=config, include_ag_args=include_ag_args)
        if config_hyperparameters is not None:
            return config_hyperparameters["hyperparameters"]
        return None

    def configs_hyperparameters(self, configs: list[str] | None = None, include_ag_args: bool = True) -> dict[str, dict | None]:
        """
        Returns a dictionary mapping of config names to hyperparameters as defined in AutoGluon
        If no hyperparameters exist for a config, its value will be None

        Note that this is not the same as the `hyperparameters` argument to AutoGluon's `TabularPredictor.fit()`.
        To get this, refer to `self.autogluon_hyperparameters_dict()`.

        Parameters
        ----------
        configs: list[str], default = None
            The list of configs to get hyperparameters for
            If None, uses all configs
        include_ag_args: bool, default = True
            If True, includes the `ag_args` hyperparameter which is used in determining the name of the model in AutoGluon
        """
        configs_hyperparameters = self._zeroshot_context.get_configs_hyperparameters(configs=configs, include_ag_args=include_ag_args)
        configs_hyperparameters = {k: v["hyperparameters"] if v is not None else v for k, v in configs_hyperparameters.items()}
        return configs_hyperparameters

    def config_type(self, config: str) -> str | None:
        """
        Returns the AutoGluon `hyperparameters` type string for a given config.
        If no type string exists, the value will be None

        For example:
            "LightGBM_c1_BAG_L1" -> "GBM"
            "RandomForest_c1_BAG_L1" -> "RF"
        """
        return self.configs_type(configs=[config])[config]

    def configs_type(self, configs: list[str] | None = None) -> dict[str, str | None]:
        """
        Returns the AutoGluon `hyperparameters` type strings for a given config list, returned as a dict of config -> type
        If no type string exists, the value will be None

        For example:
            "LightGBM_c1_BAG_L1" -> "GBM"
            "RandomForest_c1_BAG_L1" -> "RF"
        """
        configs_type = self._zeroshot_context.configs_type
        if configs is not None:
            configs_type = {c: configs_type[c] for c in configs}
        return configs_type

    def config_types(self, configs: list[str] | None = None) -> list[str | None]:
        """
        Returns the AutoGluon `hyperparameters` type strings for a given config list, returned as a list of config types.

        Parameters
        ----------
        configs: list[str], optional
            If specified, will return the types of configs present in the list.
            Otherwise, will consider all configs.

        Returns
        -------
        config_types: list[str | None]
            The list of config types present in `configs`.

        """
        configs_type = self.configs_type(configs=configs)
        config_types = set()
        for config_type in configs_type.values():
            if config_type not in config_types:
                config_types.add(config_type)
        return list(config_types)

    def autogluon_hyperparameters_dict(self, configs: list[str], ordered: bool = True, include_ag_args: bool = True) -> dict[str, list[dict]]:
        """
        Returns the AutoGluon hyperparameters dict to fit the given list of configs in AutoGluon.

        The output `hyperparameters` would be passed to AutoGluon via:

        hyperparameters = repo.autogluon_hyperparameters_dict(configs=configs)
        predictor = TabularPredictor(...).fit(..., hyperparameters=hyperparameters)

        Parameters
        ----------
        configs : list[str]
            List of configs available in this repo (must be present in self.configs())
        ordered : bool, default True
            If True, will add a `priority` hyperparameter to each config so that AutoGluon fits them in the order specified in `configs`.
        include_ag_args : bool, default True
            If True, will include the `ag_args` hyperparameters for the configs. This determines the name suffix for the model.
        """
        configs_hyperparameters = self.configs_hyperparameters(configs=configs, include_ag_args=include_ag_args)
        configs_type = self.configs_type(configs=configs)

        self._verify_autogluon_hyperparameters(configs=configs, configs_hyperparameters=configs_hyperparameters, configs_type=configs_type)

        ordered_priority = -1
        hyperparameters = {}
        for config in configs:
            config_type = configs_type[config]
            config_hyperparameters = copy.deepcopy(configs_hyperparameters[config])
            if ordered:
                if "ag_args" not in config_hyperparameters:
                    config_hyperparameters["ag_args"] = {}
                config_hyperparameters["ag_args"]["priority"] = ordered_priority
                ordered_priority -= 1
            if config_type not in hyperparameters:
                hyperparameters[config_type] = []
            hyperparameters[config_type].append(config_hyperparameters)
        return hyperparameters

    @staticmethod
    def _verify_autogluon_hyperparameters(configs: list[str], configs_hyperparameters, configs_type):
        invalid_configs = set()
        for c in configs:
            if configs_hyperparameters[c] is None or configs_type[c] is None:
                invalid_configs.add(c)
        if invalid_configs:
            invalid_configs = list(invalid_configs)
            invalid_str = ""
            for c in invalid_configs:
                invalid_str += f"\t'{c}':\ttype={configs_type[c]}, hyperparameters={configs_hyperparameters[c]}\n"
            raise AssertionError(
                f"Cannot create AutoGluon hyperparameters dict using {len(configs)} configs={configs}...\n"
                f"Found {len(invalid_configs)} invalid configs: {invalid_configs}\n"
                f"These configs either have no hyperparameters or no type specified:\n"
                f"{invalid_str}"
            )

    def _construct_single_best_config_scorer(self, **kwargs) -> SingleBestConfigScorer:
        return SingleBestConfigScorer.from_repo(repo=self, **kwargs)

    def _convert_binary_to_multiclass(self, predictions: np.ndarray, dataset: str) -> np.ndarray:
        """
        Converts binary predictions in (n_rows) format to (n_rows, n_classes) format.
        Converts binary predictions in (n_configs, n_rows) format to (n_configs, n_rows, n_classes) format.
        Skips conversion if dataset's problem_type != "binary".

        Parameters
        ----------
        predictions: np.ndarray
            The predictions to convert.
        dataset: str
            The dataset the predictions originate from.

        Returns
        -------
        Returns converted predictions if binary, else returns original predictions.
        """
        if self.dataset_info(dataset)["problem_type"] == "binary":
            return np.stack([1 - predictions, predictions], axis=predictions.ndim)
        else:
            return predictions

    # TODO: repo.reproduce(config, dataset, fold)
    def get_openml_task(self, dataset: str) -> OpenMLTaskWrapper:
        """
        Fetch an OpenML task given a dataset name

        Parameters
        ----------
        dataset: str
            The dataset name used to fetch the OpenML task.
            Must be part of `repo.datasets`

        Returns
        -------
        OpenMLTaskWrapper object
        """
        tid = self.dataset_to_tid(dataset=dataset)
        task = OpenMLTaskWrapper.from_task_id(task_id=tid)
        return task
