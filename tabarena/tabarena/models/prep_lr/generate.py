from autogluon.common.space import Real, Int, Categorical
from tabarena.benchmark.models.prep_ag.prep_lr.prep_lr_model import PrepLinearModel

from ...utils.config_utils import PrepConfigGenerator


name = 'prep_LinearModel'
manual_configs = [
    {
        "C": 'auto',
        "C_scale": 1,
        "scaler": "squashing",
    },
]

prep_manual_configs = [
    {
        "use_arithmetic_preprocessor": True,
        "use_cat_fe": True,
        "use_groupby": True,
        "use_rstafc": True,
        "use_select_spearman": True,
        "use_tafc": False,
        "use_neighbor_interactions": False,
        "use_neighbor_structure": False,        
    }]
prep_search_space = {
        # Preprocessing hyperparameters
        "use_arithmetic_preprocessor": Categorical(True, False),
        "use_cat_fe": Categorical(True, False),
        "use_rstafc": Categorical(True, False),
        "use_groupby": Categorical(True, False), 
        "use_select_spearman": Categorical(True), # Might rather tune no. of features, i.e. in {1000, 1500, 2000}
}       


search_space = {
    # "C": Real(lower=0.001, upper=10, default=1, log=True),  # FIXME: log=True?
    "C": Categorical("auto"),
    # "C_scale": Categorical(0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 4, 5, 6, 8, 10), # With many trials, could also use a continues scale
    "C_scale": Real(lower=0.0001, upper=10, default=1, log=True), # With many trials, could also use a continues scale
    "proc.skew_threshold": Categorical(None), # Don't use anymore, instead search over scalers and use the same for all features
    "proc.impute_strategy": Categorical("median", "mean"),
    "penalty": Categorical("L2", "L1"),
    "scaler": Categorical("standard", "squashing", "quantile-normal"), # "quantile-uniform", "quantile-normal"
}

gen_linear = PrepConfigGenerator(model_cls=PrepLinearModel, 
                             manual_configs=manual_configs, 
                             search_space=search_space,
                             prep_manual_configs=prep_manual_configs,
                             prep_search_space=prep_search_space
                             )


def generate_configs_lr(num_random_configs=200):
    config_generator = PrepConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
