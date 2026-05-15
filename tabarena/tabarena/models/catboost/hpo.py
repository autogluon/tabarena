from __future__ import annotations

from autogluon.tabular.models import CatBoostModel
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator


def generate_configs_catboost(num_random_configs=200):
    search_space = ConfigurationSpace(
        space=[
            Float("learning_rate", (5e-3, 1e-1), log=True),
            Categorical(
                "bootstrap_type", ["Bernoulli"]
            ),  # this is a bit faster than 'Bayesian'
            Float("subsample", (0.7, 1.0)),
            Categorical("grow_policy", ["SymmetricTree", "Depthwise"]),
            Integer("depth", (4, 8)),  # not too large for compute/memory reasons
            # leaving this out for now because catboost complains when it's supplied in SymmetricTree mode
            # Integer('min_data_in_leaf', (1, 100), log=True),  # todo: this only works for Depthwise!
            Float("colsample_bylevel", (0.85, 1.0)),
            Float("l2_leaf_reg", (1e-4, 5.0), log=True),
            # could add random_strength here but leaving it out for now
            Integer("leaf_estimation_iterations", (1, 20), log=True),
            # categorical hyperparameters
            Integer("one_hot_max_size", (8, 100), log=True),
            Float("model_size_reg", (0.1, 1.5), log=True),
            Integer("max_ctr_complexity", (2, 5)),
            # make sure the GPU version uses the same settings
            # (at least these are the two problematic parameters that I know of)
            Categorical("boosting_type", ["Plain"]),
            Categorical("max_bin", [254]),  # could be tuned, in principle
            # could search max_bin but this is expensive
        ],
        seed=1234,
    )
    configs = search_space.sample_configuration(num_random_configs)
    if num_random_configs == 1:
        configs = [configs]
    configs = [dict(config) for config in configs]
    return [convert_numpy_dtypes(config) for config in configs]


gen_catboost = CustomAGConfigGenerator(
    model_cls=CatBoostModel,
    search_space_func=generate_configs_catboost,
    manual_configs=[{}],
)


if __name__ == "__main__":
    print(generate_configs_catboost(3))
