from autogluon.common.space import Real, Int, Categorical
from tabarena.benchmark.models.prep_ag.prep_lr.prep_lr_model import PrepLinearModel

from ...utils.config_utils import PrepConfigGenerator


name = 'prep_LinearModel'
manual_configs = [
    {
        "use_arithmetic_preprocessor": True,
        "use_cat_fe": True,
    },
]
search_space = {
    "C": Real(lower=0.1, upper=1e3, default=1, log=True),  # FIXME: log=True?
    "proc.skew_threshold": Categorical(0.99, 0.9, 0.999, None),
    "proc.impute_strategy": Categorical("median", "mean"),
    "penalty": Categorical("L2", "L1"),
    "use_arithmetic_preprocessor": Categorical(True, False),
    "use_cat_fe": Categorical(True, False),

}

gen_linear = PrepConfigGenerator(model_cls=PrepLinearModel, 
                             manual_configs=manual_configs, 
                             search_space=search_space
                             )


def generate_configs_lr(num_random_configs=200):
    config_generator = PrepConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
