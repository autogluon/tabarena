from __future__ import annotations

from typing import Tuple, Type

import pandas as pd

from ..simulation.ensemble_selection_config_scorer import EnsembleScorer, EnsembleScorerMaxModels, EnsembleSelectionConfigScorer


class EnsembleMixin:
    # TODO: rank=False by default, include way more information like fit time and infer time?
    # TODO: Add time_train_s
    # TODO: Add infer_limit
    def evaluate_ensemble(
        self,
        datasets: list[str],
        configs: list[str] = None,
        *,
        ensemble_cls: Type[EnsembleScorer] = EnsembleScorerMaxModels,
        ensemble_kwargs: dict = None,
        ensemble_size: int = 100,
        rank: bool = True,
        folds: list[int] | None = None,
        backend: str = "ray",
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        :param datasets: list of datasets to compute errors on.
        :param configs: list of config to consider for ensembling. Uses all configs if None.
        :param ensemble_size: number of members to select with Caruana.
        :param ensemble_cls: class used for the ensemble model.
        :param ensemble_kwargs: kwargs to pass to the init of the ensemble class.
        :param rank: whether to return ranks or raw scores (e.g. RMSE). Ranks are computed over all base models and
        automl framework.
        :param folds: list of folds that need to be evaluated, use all folds if not provided.
        :param backend: Options include ["native", "ray"].
        :return: Tuple:
            Pandas Series of ensemble test errors per task, with multi-index (dataset, fold).
            Pandas DataFrame of ensemble weights per task, with multi-index (dataset, fold). Columns are the names of each config.
        """
        if folds is None:
            folds = self.folds
        if configs is None:
            configs = self.configs()
        tasks = [
            self.task_name(dataset=dataset, fold=fold)
            for dataset in datasets
            for fold in folds
        ]
        scorer = self._construct_ensemble_selection_config_scorer(
            tasks=tasks,
            ensemble_size=ensemble_size,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
            backend=backend,
        )

        dict_errors, dict_ensemble_weights = scorer.compute_errors(configs=configs)

        if rank:
            dict_scores = scorer.compute_ranks(errors=dict_errors)
            out = dict_scores
        else:
            out = dict_errors

        dataset_folds = [(self.task_to_dataset(task=task), self.task_to_fold(task=task)) for task in tasks]
        ensemble_weights = [dict_ensemble_weights[task] for task in tasks]
        out_list = [out[task] for task in tasks]

        multiindex = pd.MultiIndex.from_tuples(dataset_folds, names=["dataset", "fold"])

        df_name = "rank" if rank else "error"
        df_out = pd.Series(data=out_list, index=multiindex, name=df_name)
        df_ensemble_weights = pd.DataFrame(data=ensemble_weights, index=multiindex, columns=configs)

        return df_out, df_ensemble_weights

    def _construct_ensemble_selection_config_scorer(self,
                                                    ensemble_size: int = 10,
                                                    backend='ray',
                                                    **kwargs) -> EnsembleSelectionConfigScorer:
        config_scorer = EnsembleSelectionConfigScorer.from_repo(
            repo=self,
            ensemble_size=ensemble_size,  # 100 is better, but 10 allows to simulate 10x faster
            backend=backend,
            **kwargs,
        )
        return config_scorer
