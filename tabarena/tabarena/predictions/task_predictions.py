from typing import List, Dict, Tuple, Optional
import numpy as np

# dictionary mapping the config name to predictions for a given dataset fold split
ConfigPredictionsDict = Dict[str, np.array]


class AbstractTaskModelPredictions:
    def get_model_predictions(self, model: str) -> np.array:
        raise NotImplementedError

    @property
    def models(self) -> List[str]:
        raise NotImplementedError

    def subset(self, models: List[str], inplace: bool = False):
        raise NotImplementedError


class TaskModelPredictionsEmpty(AbstractTaskModelPredictions):
    """
    Empty task with no models
    """
    def get_model_predictions(self, model: str) -> np.array:
        raise AssertionError(f'Cannot get predictions from an empty task... (model="{model}")')

    @property
    def models(self) -> List[str]:
        return []

    def subset(self, models: List[str], inplace: bool = False):
        if len(models) != 0:
            raise AssertionError(f'Trying to subset an empty task to models: {models}')
        if inplace:
            return self
        else:
            return self.__class__()


class TaskModelPredictionsOpt(AbstractTaskModelPredictions):
    """
    Optimized logic to store predictions for a given task for many models.
    This combines all model predictions into a single numpy array, eliminating overheads when working with Ray.
    """
    def __init__(self, config_predictions_opt: np.array, model_index: Dict[str, int]):
        self.config_predictions_opt = config_predictions_opt
        self.model_index = model_index

    @classmethod
    def from_config_predictions(cls, config_predictions: ConfigPredictionsDict):
        config_predictions_opt, model_index = cls._stack_pred_w_index(model_pred_probas=config_predictions)
        return cls(config_predictions_opt=config_predictions_opt, model_index=model_index)

    @property
    def models(self) -> List[str]:
        return list(self.model_index.keys())

    def get_model_predictions(self, model: str) -> np.array:
        index = self.model_index[model]
        return self.config_predictions_opt[index]

    def subset(self, models: List[str], inplace: bool = False):
        """
        Subset available models to `models`.
        If `inplace=True`, alters and returns self, otherwise returns a new TaskModelPredictionsOpt object.
        Raises an AssertionError if a model in `models` is missing from `self.models`.
        """
        cur_models = self.models
        cur_models_set = set(cur_models)
        if len(models) == 0:
            return TaskModelPredictionsEmpty()
        for m in models:
            assert m in cur_models_set, f"cannot restrict {m} which is not in available models {cur_models}."

        pred_dict = dict()
        models = sorted(models)
        for m in models:
            # TODO: Potentially optimize by avoiding for loop
            pred_dict[m] = self.get_model_predictions(model=m)
        config_predictions_opt, model_index = self._stack_pred_w_index(model_pred_probas=pred_dict, models=models)

        if inplace:
            self.config_predictions_opt = config_predictions_opt
            self.model_index = model_index
            return self
        else:
            return self.__class__(config_predictions_opt=config_predictions_opt, model_index=model_index)

    @classmethod
    def _stack_pred_w_index(cls,
                           model_pred_probas: Dict[str, np.array],
                           models: Optional[List[str]] = None) -> Tuple[np.array, Dict[str, int]]:
        if models is None:
            models = list(model_pred_probas.keys())
        res = cls._stack_pred(model_pred_probas=model_pred_probas, models=models)
        model_index_dict = cls._models_to_index_dict(models=models)
        return res, model_index_dict

    @staticmethod
    def _stack_pred(model_pred_probas: Dict[str, np.array], models: Optional[List[str]] = None) -> np.array:
        """
        :param fold_dict: dictionary mapping fold to split to config name to predictions
        :return:
        """
        if models is None:
            models = list(model_pred_probas.keys())
        num_samples = len(model_pred_probas[models[0]])
        output_dims = set(
            config_evals.shape[1] if config_evals.ndim > 1 else None
            for config_evals in model_pred_probas.values()
        )
        assert len(output_dims) == 1
        output_dim = next(iter(output_dims))
        n_models = len(models)
        if output_dim is None:
            res = np.zeros((n_models, num_samples), dtype=np.float32)
        else:
            res = np.zeros((n_models, num_samples, output_dim), dtype=np.float32)

        if output_dim is not None:
            for n_model, model in enumerate(models):
                res[n_model, :, :] = model_pred_probas[model]
        else:
            for n_model, model in enumerate(models):
                res[n_model, :] = model_pred_probas[model]
        return res

    @staticmethod
    def _models_to_index_dict(models: List[str]) -> Dict[str, int]:
        return {m: i for i, m in enumerate(models)}
