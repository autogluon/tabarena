from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, List, Literal

import numpy as np
import pandas as pd
from typing_extensions import Self

from .abstract_repository import AbstractRepository
from .ensemble_mixin import EnsembleMixin
from .ground_truth_mixin import GroundTruthMixin
from .. import repository
from ..predictions.tabular_predictions import TabularModelPredictions
from ..simulation.configuration_list_scorer import ConfigurationListScorer
from ..simulation.ground_truth import GroundTruth
from ..simulation.simulation_context import ZeroshotSimulatorContext


class EvaluationRepository(AbstractRepository, EnsembleMixin, GroundTruthMixin):
    """
    Simple Repository class that implements core functionality related to
    fetching model predictions, available datasets, folds, etc.
    """
    def __init__(
            self,
            zeroshot_context: ZeroshotSimulatorContext,
            tabular_predictions: TabularModelPredictions,
            ground_truth: GroundTruth,
            config_fallback: str = None,
    ):
        """
        :param zeroshot_context:
        :param tabular_predictions:
        :param ground_truth:
        :param config_fallback: if specified, used to replace the result of a configuration that is missing, if not
        specified an error is thrown when querying a config that does not exist. A cheap baseline such as the result
        of a mean predictor can be used for the fallback.
        """
        super().__init__(zeroshot_context=zeroshot_context, config_fallback=config_fallback)
        self._tabular_predictions: TabularModelPredictions = tabular_predictions
        self._ground_truth = ground_truth
        if self._tabular_predictions is not None:
            assert all(self._zeroshot_context.dataset_to_tid_dict[x] in self._tid_to_dataset_dict for x in self._tabular_predictions.datasets)

    def to_zeroshot(self) -> repository.EvaluationRepositoryZeroshot:
        """
        Returns a version of the repository as an EvaluationRepositoryZeroshot object.

        :return: EvaluationRepositoryZeroshot object
        """
        from .evaluation_repository_zeroshot import EvaluationRepositoryZeroshot
        self_zeroshot = copy.copy(self)  # Shallow copy so that the class update does not alter self
        self_zeroshot.__class__ = EvaluationRepositoryZeroshot
        return self_zeroshot

    def _subset_folds(self, folds: list[int]):
        super()._subset_folds(folds=folds)
        if self._tabular_predictions is not None:
            self._tabular_predictions.restrict_folds(folds=folds)
        if self._ground_truth is not None:
            self._ground_truth.restrict_folds(folds=folds)

    def _subset_datasets(self, datasets: list[str]):
        super()._subset_datasets(datasets=datasets)
        if self._tabular_predictions is not None:
            self._tabular_predictions.restrict_datasets(datasets=datasets)
        if self._ground_truth is not None:
            self._ground_truth.restrict_datasets(datasets=datasets)

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

        # TODO: Move these util functions to simulations or somewhere else to avoid circular imports
        from tabarena.contexts.utils import intersect_folds_and_datasets, prune_zeroshot_gt
        # keep only dataset whose folds are all present
        intersect_folds_and_datasets(self._zeroshot_context, self._tabular_predictions, self._ground_truth)

        self.subset(configs=self._tabular_predictions.models, inplace=inplace, force_to_dense=False)

        datasets = [d for d in self._tabular_predictions.datasets if d in self._dataset_to_tid_dict]
        self.subset(datasets=datasets, inplace=inplace, force_to_dense=False)

        self._tabular_predictions.restrict_models(self.configs())
        self._ground_truth = prune_zeroshot_gt(zeroshot_pred_proba=self._tabular_predictions,
                                               zeroshot_gt=self._ground_truth,
                                               dataset_to_tid_dict=self._dataset_to_tid_dict,
                                               verbose=verbose, )
        return self

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
        predictions = self._tabular_predictions.predict_test(
            dataset=dataset,
            fold=fold,
            models=configs,
            model_fallback=self._config_fallback,
        )
        return self._convert_binary_to_multiclass(dataset=dataset, predictions=predictions) if binary_as_multiclass else predictions

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
        predictions = self._tabular_predictions.predict_val(
            dataset=dataset,
            fold=fold,
            models=configs,
            model_fallback=self._config_fallback,
        )
        return self._convert_binary_to_multiclass(dataset=dataset, predictions=predictions) if binary_as_multiclass else predictions

    def _construct_config_scorer(self,
                                 config_scorer_type: str = 'ensemble',
                                 **config_scorer_kwargs) -> ConfigurationListScorer:
        if config_scorer_type == 'ensemble':
            return self._construct_ensemble_selection_config_scorer(**config_scorer_kwargs)
        elif config_scorer_type == 'single':
            return self._construct_single_best_config_scorer(**config_scorer_kwargs)
        else:
            raise ValueError(f'Invalid config_scorer_type: {config_scorer_type}')

    @classmethod
    def from_context(
        cls,
        version: str = None,
        cache: bool = False,
        load_predictions: bool = True,
        prediction_format: str = "memmap",
        use_s3: bool = True,
        verbose: bool = True,
    ) -> Self:
        return load_repository(
            version=version,
            cache=cache,
            load_predictions=load_predictions,
            prediction_format=prediction_format,
            use_s3=use_s3,
            verbose=verbose,
        )

    # TODO: 1. Cleanup results_lst_simulation_artifacts, 2. Make context work with tasks instead of datasets x folds
    # TODO: Get raw data from repo method (X, y)
    # TODO: Score task repo method?
    # TODO: Remove score_vs_only_baselines and pct args from zeroshot_context?
    # TODO: unit test
    # TODO: docstring
    # FIXME: Support memmap directly, without needing full `results_lst_simulation_artifacts` in-memory
    @classmethod
    def from_raw(
        cls,
        df_configs: pd.DataFrame,
        results_lst_simulation_artifacts: list[dict[str, dict[int, dict]]],
        df_baselines: pd.DataFrame = None,
        task_metadata: pd.DataFrame = None,
        configs_hyperparameters: dict[str, dict[str, Any]] = None,
        pct: bool = False,
        score_against_only_baselines: bool = False,
    ) -> Self:
        from tabarena.predictions import TabularPredictionsInMemory
        from tabarena.simulation.ground_truth import GroundTruth
        from tabarena.simulation.simulation_context import ZeroshotSimulatorContext
        from autogluon.common.utils.simulation_utils import convert_simulation_artifacts_to_tabular_predictions_dict

        required_columns = [
            "dataset",
            "fold",
            "framework",
            "metric_error",
            "metric",
            "problem_type",
            "time_train_s",
            "time_infer_s",
        ]

        if df_configs is not None:
            for column in required_columns:
                if column not in df_configs:
                    raise AssertionError(f"Missing required column in df_configs: {column}\ndf_configs columns: {list(df_configs.columns)}")

        if results_lst_simulation_artifacts is not None:
            simulation_artifacts_full = cls._convert_sim_artifacts(results_lst_simulation_artifacts=results_lst_simulation_artifacts)

            zeroshot_pp, zeroshot_gt = convert_simulation_artifacts_to_tabular_predictions_dict(simulation_artifacts=simulation_artifacts_full)

            predictions = TabularPredictionsInMemory.from_dict(zeroshot_pp)
            ground_truth = GroundTruth.from_dict(zeroshot_gt)
        else:
            predictions = None
            ground_truth = None

        zeroshot_context = ZeroshotSimulatorContext(
            df_configs=df_configs,
            df_baselines=df_baselines,
            df_metadata=task_metadata,
            configs_hyperparameters=configs_hyperparameters,
            pct=pct,
            score_against_only_baselines=score_against_only_baselines,
        )

        repo = cls(
            zeroshot_context=zeroshot_context,
            tabular_predictions=predictions,
            ground_truth=ground_truth,
        )

        return repo

    def to_dir(self, path: str | Path):
        from tabarena.contexts.context import BenchmarkContext, construct_context

        path = os.path.abspath(path) + os.path.sep
        path_data_dir = path + "model_predictions/"

        # FIXME: use tasks rather than datasets and folds separately
        datasets = self.datasets()
        folds = self.folds
        if folds is not None:
            # make list serializable to json
            folds = [int(f) for f in folds]

        self._tabular_predictions.to_data_dir(data_dir=path_data_dir)
        self._ground_truth.to_data_dir(data_dir=path_data_dir)

        dataset_fold_lst_pp = self._tabular_predictions.dataset_fold_lst()
        dataset_fold_lst_gt = self._ground_truth.dataset_fold_lst()

        metadata = self._zeroshot_context.to_dir(path=path)

        configs_hyperparameters = metadata["configs_hyperparameters"]
        if configs_hyperparameters is not None:
            configs_hyperparameters = [configs_hyperparameters]

        # FIXME: Make this a repo constructor method?
        # FIXME: s3_download_map doesn't work with is_relative yet
        context: BenchmarkContext = construct_context(
            name=None,
            datasets=datasets,
            folds=folds,
            local_prefix=path,
            local_prefix_is_relative=False,  # TODO: Set to False by default and rename
            has_baselines=metadata["df_baselines"] is not None,
            task_metadata=metadata["df_metadata"],
            configs_hyperparameters=configs_hyperparameters,
            is_relative=True,
            config_fallback=self._config_fallback,
            dataset_fold_lst_pp=dataset_fold_lst_pp,
            dataset_fold_lst_gt=dataset_fold_lst_gt,
        )

        context.to_json(path=str(Path(path) / "context.json"))

    @classmethod
    def from_dir(
        cls,
        path: str | Path,
        prediction_format: Literal["memmap", "memopt", "mem"] = "memmap",
        update_relative_path: bool = True,
        verbose: bool = True,
    ) -> Self:
        from tabarena.contexts.context import BenchmarkContext

        path_context = str(Path(path) / "context.json")
        context = BenchmarkContext.from_json(path=path_context)
        if update_relative_path:
            context.benchmark_paths.relative_path = str(Path(path))

        repo = context.load_repo(prediction_format=prediction_format, verbose=verbose)
        return repo

    @classmethod
    def _convert_sim_artifacts(cls, results_lst_simulation_artifacts: list[dict[str, dict[int, dict[str, Any]]]]) -> dict[str, dict[int, dict[str, Any]]]:
        # FIXME: Don't require all results in memory at once
        simulation_artifacts_full = {}
        for simulation_artifacts in results_lst_simulation_artifacts:
            for k in simulation_artifacts.keys():
                if k not in simulation_artifacts_full:
                    simulation_artifacts_full[k] = {}
                for f in simulation_artifacts[k]:
                    if f not in simulation_artifacts_full[k]:
                        simulation_artifacts_full[k][f] = copy.deepcopy(simulation_artifacts[k][f])
                    else:
                        for method in simulation_artifacts[k][f]["pred_proba_dict_val"]:
                            if method in simulation_artifacts_full[k][f]["pred_proba_dict_val"]:
                                raise AssertionError(f"Two results exist for dataset {k}, fold {f}, method {method}!")
                            else:
                                simulation_artifacts_full[k][f]["pred_proba_dict_val"][method] = simulation_artifacts[k][f]["pred_proba_dict_val"][method]
                                simulation_artifacts_full[k][f]["pred_proba_dict_test"][method] = simulation_artifacts[k][f]["pred_proba_dict_test"][method]
        return simulation_artifacts_full


def load_repository(
    version: str,
    *,
    load_predictions: bool = True,
    cache: bool | str = False,
    prediction_format: str = "memmap",
    use_s3: bool = True,
    verbose: bool = True,
) -> EvaluationRepository:
    """
    Load the specified EvaluationRepository. Will automatically download all required inputs if they do not already exist on local disk.

    Parameters
    ----------
    version: str
        The name of the context to load.
    load_predictions: bool, default = True
        If True, loads the config predictions.
        If False, does not load the config predictions (disabling the ability to simulate ensembling).
    cache: bool | str, default = False
        Valid values: [True, False, "overwrite"]
        If True, will load directly from a cached repository or cache the loaded evaluation repository to accelerate future load calls.
        Setting to True may lead to incompatibility in loading repositories from different versions of the codebase.
        If "overwrite", will overwrite the existing cache and cache the new version.
    prediction_format: str, default = "memmap"
        Options: ["memmap", "memopt", "mem"]
        Determines the way the predictions are represented in the repo.
        It is recommended to keep the value as "memmap" for optimal performance.
    use_s3: bool, default = True
        Whether to use S3 to download tabrepo files, if False uses HuggingFace instead.
    Returns
    -------
    EvaluationRepository object for the given context.
    """
    from tabarena.contexts import get_subcontext
    if cache is not False:
        kwargs = dict()
        if isinstance(cache, str) and cache == "overwrite":
            kwargs["ignore_cache"] = True
            kwargs["exists"] = "overwrite"
        repo = get_subcontext(version).load(load_predictions=load_predictions, prediction_format=prediction_format, use_s3=use_s3, verbose=verbose, **kwargs)
    else:
        repo = get_subcontext(version).load_from_parent(load_predictions=load_predictions, prediction_format=prediction_format, use_s3=use_s3, verbose=verbose)
    return repo
