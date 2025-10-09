from __future__ import annotations

from tabrepo.benchmark.models.ag.prep_lgb.prep_lgb_model import PrepLGBModel
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from tabrepo.benchmark.experiment import YamlExperimentSerializer
from tabrepo.models.utils import convert_numpy_dtypes
from tabrepo.utils.config_utils import CustomAGConfigGenerator

from tabprep.presets.lgb_presets import get_lgb_presets

def generate_configs_lightgbm(num_random_configs=200) -> list:
    search_space = ConfigurationSpace(
        space=[
            Float("learning_rate", (5e-3, 1e-1), log=True),
            Float("feature_fraction", (0.4, 1.0)),
            Float("bagging_fraction", (0.7, 1.0)),
            Categorical("bagging_freq", [1]),
            Integer("num_leaves", (2, 200), log=True),
            Integer("min_data_in_leaf", (1, 64), log=True),
            Categorical("extra_trees", [False, True]),
            # categorical hyperparameters
            Integer("min_data_per_group", (2, 100), log=True),
            Float("cat_l2", (5e-3, 2), log=True),
            Float("cat_smooth", (1e-3, 100), log=True),
            Integer("max_cat_to_onehot", (8, 100), log=True),
            # these seem to help a little bit but can also make things slower
            Float("lambda_l1", (1e-4, 1.0)),
            Float("lambda_l2", (1e-4, 2.0)),
            # could search max_bin but this is expensive
        ],
        seed=1234,
    )
    
    configs = search_space.sample_configuration(num_random_configs)
    if num_random_configs == 1:
        configs = [configs]
    configs = [dict(config) for config in configs]
    
    r_num = 1
    for i in range(len(configs)):
        if 'prep_params' not in configs[i]:
            configs[i]['prep_params'] = {}
        if 'preset_name' not in configs[i]:
            configs[i]['preset_name'] = f'r{r_num}'
            r_num += 1
    return [convert_numpy_dtypes(config) for config in configs]

presets = get_lgb_presets()
manual_configs = []

for k, v in presets.items():
    v.update({'preset_name': k})
    manual_configs.append(v)

gen_prep_lightgbm = CustomAGConfigGenerator(
    model_cls=PrepLGBModel,
    search_space_func=generate_configs_lightgbm,
    manual_configs=manual_configs,
)


if __name__ == "__main__":
    experiments = gen_prep_lightgbm.generate_all_bag_experiments(num_random_configs=25)
    YamlExperimentSerializer.to_yaml(
        experiments=experiments, path="configs_prep_lightgbm_alt.yaml"
    )
    print(f"Generated {len(experiments)} experiments.")