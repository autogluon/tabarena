from __future__ import annotations

import copy
import itertools
from typing import Literal, Tuple, Type

import numpy as np
import pandas as pd

from .time_utils import filter_configs_by_runtime, get_runtime
from ..simulation.ensemble_selection_config_scorer import EnsembleScorer, EnsembleScorerMaxModels, EnsembleSelectionConfigScorer
from ..utils.parallel_for import parallel_for


# FIXME: Type hints for AbstractRepository, how to do? Protocol?
class EnsembleMixin:
    # TODO: rank=False by default?
    # TODO: ensemble_size remove, put into ensemble_kwargs?
    # TODO: rename to fit_ensemble?
    #  Ditto for ensemble_weights
    def evaluate_ensemble(
        self,
        dataset: str,
        fold: int,
        configs: list[str] = None,
        *,
        time_limit: float = None,
        ensemble_cls: Type[EnsembleScorer] = EnsembleScorerMaxModels,
        ensemble_kwargs: dict = None,
        ensemble_size: int = 100,
        rank: bool = False,
        fit_order: Literal["original", "random"] = "original",
        seed: int = 0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluates an ensemble of a list of configs on a given task (dataset, fold).

        Parameters
        ----------
        dataset: str
            The dataset to evaluate
        fold: int
            The fold of the dataset to evaluate
        configs: list[str], default = None
            The list of configs to consider for ensembling.
            If None, will use all configs.
            Models will be simulated as being fit in the order specified in `fit_order`.
        time_limit: float, default = None
            The time limit of the ensemble.
            Will only consider the first N models in `configs` whose cumulative time limit is less than `time_limit`.
        ensemble_cls: Type[EnsembleScorer], default = EnsembleScorerMaxModels
            The ensemble method to use.
        ensemble_kwargs: dict, default = None
            The ensemble method kwargs.
        ensemble_size: int, default = 100
            The number of ensemble iterations.
        rank: bool, default = False
            If True, additionally calculates the rank of the ensemble result.
        fit_order: Literal["original", "random"], default = "original"
            Whether to simulate the models being fit in their original order sequentially or randomly.
        seed: int, default = 0
            The random seed used to shuffle `configs` if `fit_order="random"`.

        Returns
        -------
        result: pd.DataFrame
            A single-row multi-index (dataset, fold) DataFrame with the following columns:
                metric_error: float
                    The ensemble's metric test error.
                metric: str
                    The target evaluation metric.
                time_train_s: float
                    The training time of the ensemble in seconds (the sum of all considered models' time_train_s)
                time_infer_s: float
                    The inference time of the ensemble in seconds (the sum of all non-zero weight models' time_infer_s)
                problem_type: str
                    The problem type of the task.
                metric_error_val: float
                    The ensemble's metric validation error.
        ensemble_weights: pd.DataFrame
            A single-row multi-index (dataset, fold) DataFrame with column names equal to `configs`.
            Each config column's value is the weight given to it by the ensemble model.
            This can be used for debugging purposes and for deeper analysis.

        """
        task = self.task_name(dataset=dataset, fold=fold)

        task_tuple = (dataset, fold)
        config_metrics = self.metrics(
            tasks=[task_tuple],
            configs=configs,
            set_index=False,
        )

        configs_all = sorted(list(config_metrics["framework"].unique()))
        if configs is None:
            configs = configs_all

        if time_limit is not None:
            if fit_order == "random":
                # randomly shuffle the configs
                rng = np.random.default_rng(seed=seed)
                configs_fit_order = list(rng.permuted(configs))
            else:
                configs_fit_order = configs

            # filter configs to the first N configs whose combined time_limit is less than the provided time_limit
            configs = filter_configs_by_runtime(
                repo=self,
                dataset=dataset,
                fold=fold,
                config_names=configs_fit_order,
                config_metrics=config_metrics,
                max_cumruntime=time_limit,
            )

        configs_available = [c for c in configs if c in set(configs_all)]

        if len(configs_available) == 0:
            # if not enough time to fit any model, use the fallback config if it exists, even if it would be over the time limit
            # if no config fallback was specified, then raise an AssertionError
            if self._config_fallback is None:
                if len(configs_fit_order) > 0:
                    raise AssertionError(
                        f"Can't fit an ensemble with no configs when self._config_fallback is None "
                        f"(No configs are trainable in the provided time_limit={time_limit}.)"
                    )
                else:
                    raise AssertionError(f"Can't fit an ensemble with no configs when self._config_fallback is None.")
            configs_to_use = [self._config_fallback]
            config_metrics = self.metrics(
                tasks=[task_tuple],
                configs=configs_to_use,
                set_index=False,
            )
        else:
            configs_to_use = copy.deepcopy(configs_available)
            config_metrics = config_metrics[config_metrics["framework"].isin(configs_to_use)]

        if len(configs_available) == 0:
            imputed = True
            impute_method = self._config_fallback
        else:
            imputed = False
            impute_method = np.nan

        scorer = self._construct_ensemble_selection_config_scorer(
            tasks=[task],
            ensemble_size=ensemble_size,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
            backend="native",
        )

        # fit the ensemble and retrieve the metric error and ensemble weights
        results = scorer.compute_errors(configs=configs_to_use)
        metric_error = results[task]["metric_error"]
        ensemble_weights = results[task]["ensemble_weights"]
        metric_error_val = results[task]["metric_error_val"]

        dataset_info = self.dataset_info(dataset=dataset)
        metric = dataset_info["metric"]
        problem_type = dataset_info["problem_type"]

        # select configurations used in the ensemble as infer time only depends on the models with non-zero weight.
        fail_if_missing = self._config_fallback is None

        # compute the ensemble time_train_s by summing all considered config's time_train_s
        runtimes = get_runtime(
            repo=self,
            dataset=dataset,
            fold=fold,
            config_names=configs_to_use,
            config_metrics=config_metrics,
            runtime_col='time_train_s',
            fail_if_missing=fail_if_missing,
        )
        time_train_s = sum(runtimes.values())

        # compute the ensemble time_infer_s by summing all considered config's time_infer_s that have non-zero weight
        config_selected_ensemble = [
            config for i, config in enumerate(configs_to_use) if ensemble_weights[i] != 0
        ]

        config_metrics_inference = config_metrics[config_metrics["framework"].isin(config_selected_ensemble)]

        latencies = get_runtime(
            repo=self,
            dataset=dataset,
            fold=fold,
            config_names=config_selected_ensemble,
            config_metrics=config_metrics_inference,
            runtime_col='time_infer_s',
            fail_if_missing=fail_if_missing,
        )
        time_infer_s = sum(latencies.values())

        output_dict = {
            "metric_error": [metric_error],
            "metric": [metric],
            "time_train_s": [time_train_s],
            "time_infer_s": [time_infer_s],
            "problem_type": [problem_type],
            "metric_error_val": [metric_error_val],
            "imputed": [imputed],
            "impute_method": [impute_method],
        }

        if rank:
            dict_ranks = scorer.compute_ranks(errors={task: metric_error})
            rank_list = dict_ranks[task]
            output_dict["rank"] = [rank_list]

        multiindex = pd.MultiIndex.from_tuples([(dataset, fold)], names=["dataset", "fold"])
        df_ensemble_weights = pd.DataFrame(data=[ensemble_weights], index=multiindex, columns=configs_to_use)
        df_out = pd.DataFrame(data=output_dict, index=multiindex)

        return df_out, df_ensemble_weights

    def evaluate_ensembles(
        self,
        datasets: list[str] = None,
        folds: list[int] = None,
        configs: list[str] = None,
        *,
        ensemble_cls: Type[EnsembleScorer] = EnsembleScorerMaxModels,
        ensemble_kwargs: dict = None,
        ensemble_size: int = 100,
        time_limit: float = None,
        fit_order: Literal["original", "random"] = "original",
        seed: int = 0,
        rank: bool = False,
        backend_group_folds: bool = False,
        backend: Literal["ray", "native"] = "ray",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluates an ensemble of a list of configs on a given set of tasks (datasets x folds).

        Identical to calling `evaluate_ensemble` once for each task and then concatenating the results,
        however this method will be much faster due to parallelization.

        Parameters
        ----------
        datasets: list[str], default = None
            The datasets to evaluate.
            If None, will use all datasets.
        folds: list[int], default = None
            The folds of the dataset to evaluate.
            If None, will use all folds.
        configs: list[str], default = None
            The list of configs to consider for ensembling.
            If None, will use all configs.
            Models will be simulated as being fit in the order specified in `fit_order`.
        time_limit: float, default = None
            The time limit of the ensemble.
            Will only consider the first N models in `configs` whose cumulative time limit is less than `time_limit`.
        ensemble_cls: Type[EnsembleScorer], default = EnsembleScorerMaxModels
            The ensemble method to use.
        ensemble_kwargs: dict, default = None
            The ensemble method kwargs.
        ensemble_size: int, default = 100
            The number of ensemble iterations.
        rank: bool, default = False
            If True, additionally calculates the rank of the ensemble result.
        fit_order: Literal["original", "random"], default = "original"
            Whether to simulate the models being fit in their original order sequentially or randomly.
        seed: int, default = 0
            The random seed used to shuffle `configs` if `fit_order="random"`.
        backend: Literal["ray", "native"], default = "ray"
            The backend to use when running the list of tasks.

        Returns
        -------
        result: pd.DataFrame
            A multi-index (dataset, fold) DataFrame where each row corresponds to a task, with the following columns:
                metric_error: float
                    The ensemble's metric test error.
                metric: str
                    The target evaluation metric.
                time_train_s: float
                    The training time of the ensemble in seconds (the sum of all considered models' time_train_s)
                time_infer_s: float
                    The inference time of the ensemble in seconds (the sum of all non-zero weight models' time_infer_s)
                problem_type: str
                    The problem type of the task.
                metric_error_val: float
                    The ensemble's metric validation error.
        ensemble_weights: pd.DataFrame
            A multi-index (dataset, fold) DataFrame with column names equal to `configs`. Each row corresponds to a task.
            Each config column's value is the weight given to it by the ensemble model.
            This can be used for debugging purposes and for deeper analysis.

        """
        if backend == "native":
            backend = "sequential"
        if datasets is None:
            datasets = self.datasets()

        context = dict(
            self=self,
            configs=configs,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
            ensemble_size=ensemble_size,
            time_limit=time_limit,
            fit_order=fit_order,
            seed=seed,
            rank=rank,
        )

        if folds is None:
            inputs = []
            for dataset in datasets:
                folds_in_dataset = self.dataset_to_folds(dataset=dataset)
                for fold in folds_in_dataset:
                    inputs.append((dataset, fold))
        else:
            inputs = list(itertools.product(datasets, folds))

        if backend_group_folds:
            inputs_dict = {}
            for dataset, fold in inputs:
                if dataset not in inputs_dict:
                    inputs_dict[dataset] = []
                inputs_dict[dataset].append(fold)
            inputs = [{"datasets": [dataset], "folds": folds} for dataset, folds in inputs_dict.items()]
            context.update({"backend": "native", "backend_group_folds": False})
            par_func = self.__class__.evaluate_ensembles
        else:
            inputs = [{"dataset": dataset, "fold": fold} for dataset, fold in inputs]
            par_func = self.__class__.evaluate_ensemble

        list_rows = parallel_for(
            par_func,
            inputs=inputs,
            context=context,
            engine=backend,
            # To reduce log spam in outer parallel mode.
            progress_bar=backend != "sequential",
        )

        df_out = pd.concat([l[0] for l in list_rows], axis=0)
        df_ensemble_weights = pd.concat([l[1] for l in list_rows], axis=0)  # FIXME: Is this guaranteed same columns in each?

        return df_out, df_ensemble_weights

    def evaluate_ensembles_per(
        self,
        df_info: pd.DataFrame | list[dict[str]],
        *,
        ensemble_cls: Type[EnsembleScorer] = EnsembleScorerMaxModels,
        ensemble_kwargs: dict = None,
        ensemble_size: int = 100,
        time_limit: float = None,
        fit_order: Literal["original", "random"] = "original",
        seed: int = 0,
        rank: bool = False,
        backend_group_folds: bool = True,
        backend: Literal["ray", "native"] = "ray",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identical to calling `evaluate_ensemble` once for each row in df_info,
        however this method will be much faster due to parallelization.

        Parameters
        ----------
        df_info: pd.DataFrame, default = None
            A DataFrame with columns `configs` (list[str]), dataset (str), fold (int).
            Indicates which configs will be run for a given dataset and fold.
        time_limit: float, default = None
            The time limit of the ensemble.
            Will only consider the first N models in `configs` whose cumulative time limit is less than `time_limit`.
        ensemble_cls: Type[EnsembleScorer], default = EnsembleScorerMaxModels
            The ensemble method to use.
        ensemble_kwargs: dict, default = None
            The ensemble method kwargs.
        ensemble_size: int, default = 100
            The number of ensemble iterations.
        rank: bool, default = False
            If True, additionally calculates the rank of the ensemble result.
        fit_order: Literal["original", "random"], default = "original"
            Whether to simulate the models being fit in their original order sequentially or randomly.
        seed: int, default = 0
            The random seed used to shuffle `configs` if `fit_order="random"`.
        backend: Literal["ray", "native"], default = "ray"
            The backend to use when running the list of tasks.

        Returns
        -------
        result: pd.DataFrame
            A multi-index (dataset, fold) DataFrame where each row corresponds to a task, with the following columns:
                metric_error: float
                    The ensemble's metric test error.
                metric: str
                    The target evaluation metric.
                time_train_s: float
                    The training time of the ensemble in seconds (the sum of all considered models' time_train_s)
                time_infer_s: float
                    The inference time of the ensemble in seconds (the sum of all non-zero weight models' time_infer_s)
                problem_type: str
                    The problem type of the task.
                metric_error_val: float
                    The ensemble's metric validation error.
        ensemble_weights: pd.DataFrame
            A multi-index (dataset, fold) DataFrame with column names equal to `configs`. Each row corresponds to a task.
            Each config column's value is the weight given to it by the ensemble model.
            This can be used for debugging purposes and for deeper analysis.

        """
        if backend == "native":
            backend = "sequential"

        context = dict(
            self=self,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
            ensemble_size=ensemble_size,
            time_limit=time_limit,
            fit_order=fit_order,
            seed=seed,
            rank=rank,
        )

        if isinstance(df_info, pd.DataFrame):
            inputs = df_info[["dataset", "fold", "configs"]].values
            inputs = [{"dataset": dataset, "fold": fold, "configs": configs} for dataset, fold, configs in inputs]
        else:
            inputs = df_info

        if backend_group_folds:
            inputs_dict = {}
            for input_dict in inputs:
                dataset = input_dict["dataset"]
                if dataset not in inputs_dict:
                    inputs_dict[dataset] = []
                inputs_dict[dataset].append(input_dict)
            inputs = [{"df_info": inputs_dict[dataset]} for dataset in inputs_dict.keys()]
            context.update({"backend": "native", "backend_group_folds": False})
            par_func = self.__class__.evaluate_ensembles_per
        else:
            par_func = self.__class__.evaluate_ensemble

        list_rows = parallel_for(
            par_func,
            inputs=inputs,
            context=context,
            engine=backend,
            # To reduce log spam in outer parallel mode.
            progress_bar=backend != "sequential"
        )

        df_out = pd.concat([l[0] for l in list_rows], axis=0)
        df_ensemble_weights = pd.concat([l[1] for l in list_rows],
                                        axis=0)  # FIXME: Is this guaranteed same columns in each?

        return df_out, df_ensemble_weights

    def _construct_ensemble_selection_config_scorer(
        self,
        ensemble_size: int = 10,
        backend: str = 'ray',
        **kwargs
    ) -> EnsembleSelectionConfigScorer:
        config_scorer = EnsembleSelectionConfigScorer.from_repo(
            repo=self,
            ensemble_size=ensemble_size,  # 100 is better, but 10 allows to simulate 10x faster
            backend=backend,
            **kwargs,
        )
        return config_scorer
