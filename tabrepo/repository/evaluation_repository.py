from __future__ import annotations

import copy
from typing import List

import numpy as np
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
        from tabrepo.contexts.utils import intersect_folds_and_datasets, prune_zeroshot_gt
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
    def from_context(cls, version: str = None, prediction_format: str = "memmap"):
        return load_repository(version=version, prediction_format=prediction_format)


def load_repository(version: str, *, load_predictions: bool = True, cache: bool | str = False, prediction_format: str = "memmap") -> EvaluationRepository:
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

    Returns
    -------
    EvaluationRepository object for the given context.
    """
    from tabrepo.contexts import get_subcontext
    if cache is not False:
        kwargs = dict()
        if isinstance(cache, str) and cache == "overwrite":
            kwargs["ignore_cache"] = True
            kwargs["exists"] = "overwrite"
        repo = get_subcontext(version).load(load_predictions=load_predictions, prediction_format=prediction_format, **kwargs)
    else:
        repo = get_subcontext(version).load_from_parent(load_predictions=load_predictions, prediction_format=prediction_format)
    return repo
