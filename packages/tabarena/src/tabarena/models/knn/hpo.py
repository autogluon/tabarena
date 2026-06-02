from __future__ import annotations

from autogluon.common.space import Categorical

from tabarena.models.knn.model import KNNNewModel
from tabarena.utils.config_utils import ConfigGenerator


search_space = {
    "n_neighbors": Categorical(20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 30, 40, 50, 100, 200, 300, 400, 500),
    "weights": Categorical("distance", "uniform"),
    "p": Categorical(2, 1, 1.5),
    "scaler": Categorical("standard", "quantile"),
    "cat_threshold": Categorical(10, 0, 1, 5, 20, 30, 50, 100, 1000000),
}


gen_knn = ConfigGenerator(
    model_cls=KNNNewModel,
    manual_configs=[{}],
    search_space=search_space,
)


def generate_configs_knn(num_random_configs=200):
    config_generator = ConfigGenerator(
        name="KNeighbors_new", manual_configs=[{}], search_space=search_space
    )
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
