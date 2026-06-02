from __future__ import annotations

from autogluon.common.space import Categorical, Real
from autogluon.tabular.models import LinearModel

from tabarena.utils.config_utils import ConfigGenerator

search_space = {
    "C": Real(lower=0.1, upper=1e3, default=1, log=True),  # FIXME: log=True?
    "proc.skew_threshold": Categorical(0.99, 0.9, 0.999, None),
    "proc.impute_strategy": Categorical("median", "mean"),
    "penalty": Categorical("L2", "L1"),
}


gen_linear = ConfigGenerator(
    model_cls=LinearModel,
    manual_configs=[{}],
    search_space=search_space,
)


def generate_configs_lr(num_random_configs=200):
    config_generator = ConfigGenerator(
        name="LinearModel",
        manual_configs=[{}],
        search_space=search_space,
    )
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
