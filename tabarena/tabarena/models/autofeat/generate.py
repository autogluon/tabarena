from autogluon.common.space import Real, Int, Categorical
from autogluon.tabular.models import LinearModel

from ...utils.config_utils import ConfigGenerator

from tabarena.benchmark.models.prep_ag import AutoFeatLinearModel


# name = 'LinearModel'
manual_configs = [
    {},
    {"transformations": ()}
]
search_space = {
    # "C": Real(lower=0.001, upper=10, default=1, log=True),  # FIXME: log=True?
    "C": Categorical("auto"),
    "C_scale": Categorical(0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 4, 5, 6, 8, 10), # With many trials, could also use a continues scale
    # "C_scale": Real(lower=0.0001, upper=10, default=1, log=True), # With many trials, could also use a continues scale
    "proc.skew_threshold": Real(lower=0.001, upper=1., default=None), # Just added to bypass local random searcher getting stuck. Don't use anymore, instead search over scalers and use the same for all features
    "proc.impute_strategy": Categorical("median", "mean"),
    "penalty": Categorical("L2", "L1"),
    "scaler": Categorical("standard", "squashing", "quantile-normal"), # "quantile-uniform", "quantile-normal"

}

gen_autofeatlinear = ConfigGenerator(model_cls=AutoFeatLinearModel, manual_configs=manual_configs, search_space=search_space)


def generate_configs_lr(num_random_configs=200):
    config_generator = ConfigGenerator(model_cls=AutoFeatLinearModel, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
