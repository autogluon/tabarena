from __future__ import annotations

from autogluon.common.space import Categorical, Int, Real

from tabarena.benchmark.models.ag.ebm.ebm_model import ExplainableBoostingMachineModel
from tabarena.utils.config_utils import ConfigGenerator

name = "EBM"
manual_configs = []
search_space = {
    "max_leaves": Int(2, 3, default=2),
    "smoothing_rounds": Int(0, 1000, default=200),
    "learning_rate": Real(0.0025, 0.2, default=0.02, log=True),
    "interactions": Categorical(
        0,
        "0.5x",
        "1x",
        "1.5x",
        "2x",
        "2.5x",
        "3x",
        "3.5x",
        "4x",
        "4.5x",
        "5x",
        "6x",
        "7x",
        "8x",
        "9x",
        "10x",
        "15x",
        "20x",
        "25x",
    ),
    "interaction_smoothing_rounds": Int(0, 200, default=90),
    "min_hessian": Real(1e-10, 1e-2, default=1e-4, log=True),
    "min_samples_leaf": Int(2, 20, default=4),
    "gain_scale": Real(0.5, 5.0, default=5.0, log=True),
    "min_cat_samples": Int(5, 20, default=10),
    "cat_smooth": Real(5.0, 100.0, default=10.0, log=True),
    "missing": Categorical("separate", "low", "high", "gain"),
}

gen_ebm = ConfigGenerator(
    model_cls=ExplainableBoostingMachineModel,
    search_space=search_space,
    manual_configs=[{}],
)


def generate_configs_ebm(num_random_configs=200):
    config_generator = ConfigGenerator(
        name=name, manual_configs=manual_configs, search_space=search_space
    )
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
