import copy
from typing import Dict, List

import numpy as np

from .task_predictions import TaskModelPredictionsOpt, ConfigPredictionsDict
from .tabular_predictions import TabularPredictionsInMemory, TabularPredictionsDict


class TabularPredictionsInMemoryOpt(TabularPredictionsInMemory):
    """
    A model predictions data representation optimized for `ray.put(self)` operations to minimize overhead.
    Ray has a large overhead when using a shared object with many numpy arrays (such as 500,000).
    This class converts many smaller numpy arrays into fewer larger numpy arrays,
    eliminating the vast majority of the overhead.
    """
    def __init__(self, pred_dict_opt: Dict[str, Dict[int, Dict[str, TaskModelPredictionsOpt]]], datasets: List[str] = None):
        super().__init__(pred_dict=pred_dict_opt, datasets=datasets)
        self.pred_dict = pred_dict_opt

    @classmethod
    def from_dict(cls, pred_dict: TabularPredictionsDict, output_dir: str = None, datasets: List[str] = None):
        pred_dict_opt = cls._stack_pred_dict(pred_dict=pred_dict)
        return cls(pred_dict_opt=pred_dict_opt, datasets=datasets)

    def to_dict(self) -> TabularPredictionsDict:
        model_available_dict = self.model_available_dict()
        return {
            dataset: {
                fold: {
                    "pred_proba_dict_val": {
                        model: self.predict_val(dataset, fold, [model]).squeeze() for model in models
                    },
                    "pred_proba_dict_test": {
                        model: self.predict_test(dataset, fold, [model]).squeeze() for model in models
                    }
                } for fold, models in fold_dict.items()
            } for dataset, fold_dict in model_available_dict.items()
        }

    def restrict_models(self, models: List[str]):
        task_names = self.datasets
        for t in task_names:
            available_folds = list(self.pred_dict[t].keys())
            for f in available_folds:
                available_splits = list(self.pred_dict[t][f].keys())
                for s in available_splits:
                    self.pred_dict[t][f][s] = self.pred_dict[t][f][s].subset(models=models, inplace=True)
                    if not self.pred_dict[t][f][s].models:
                        # If no models, then pop the entire task and go to the next task
                        self.pred_dict[t].pop(f)
                        break
            if not self.pred_dict[t]:
                # If no folds, then pop the entire dataset
                self.pred_dict.pop(t)

    @classmethod
    def _stack_pred_dict(cls, pred_dict: TabularPredictionsDict) -> Dict[str, Dict[int, Dict[str, TaskModelPredictionsOpt]]]:
        pred_dict = copy.deepcopy(pred_dict)  # TODO: Avoid the deep copy, create from scratch to min mem usage
        datasets = list(pred_dict.keys())
        for dataset in datasets:
            folds = list(pred_dict[dataset].keys())
            for fold in folds:
                splits = list(pred_dict[dataset][fold].keys())
                for split in splits:
                    model_pred_probas: ConfigPredictionsDict = pred_dict[dataset][fold][split]
                    pred_dict[dataset][fold][split] = TaskModelPredictionsOpt.from_config_predictions(
                        config_predictions=model_pred_probas
                    )
        return pred_dict

    def _get_model_results(self, model: str, model_pred_probas: TaskModelPredictionsOpt) -> np.array:
        return model_pred_probas.get_model_predictions(model=model)

    def _model_available_dict(self) -> Dict[str, Dict[int, List[str]]]:
        return {
            dataset: {
                fold: list(fold_info['pred_proba_dict_val'].models) for fold, fold_info in fold_dict.items()
            }
            for dataset, fold_dict in self.pred_dict.items()
        }
