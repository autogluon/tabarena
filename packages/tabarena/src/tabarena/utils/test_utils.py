from math import prod
from typing import List

import numpy as np

from tabarena.predictions.tabular_predictions import TabularPredictionsDict


def generate_dummy(shape, models):
    return {
        model: np.arange(prod(shape)).reshape(shape) + int(model)
        for model in models
    }


def generate_artificial_dict(
        num_folds: int,
        models: List[str],
        dataset_shapes: dict = None,
) -> TabularPredictionsDict:
    if dataset_shapes is None:
        dataset_shapes = {
            "d1": ((20,), (50,)),
            "d2": ((10,), (5,)),
            "d3": ((4, 3), (8, 3)),
        }
    # dictionary mapping dataset to fold to split to config name to predictions
    pred_dict: TabularPredictionsDict = {
        dataset: {
            fold: {
                "pred_proba_dict_val": generate_dummy(val_shape, models),
                "pred_proba_dict_test": generate_dummy(test_shape, models),
            }
            for fold in range(num_folds)
        }
        for dataset, (val_shape, test_shape) in dataset_shapes.items()
    }
    return pred_dict
