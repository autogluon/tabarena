from __future__ import annotations

from tabarena.benchmark.models.prep_ag.prep_xgb.prep_xgb_model import PrepXGBoostModel
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator


def generate_configs_xgboost(num_random_configs=200):
    search_space = ConfigurationSpace(
        space=[
            Float("learning_rate", (5e-3, 1e-1), log=True),
            Integer("max_depth", (4, 10), log=True),
            Float("min_child_weight", (1e-3, 5.0), log=True),
            Float("subsample", (0.6, 1.0)),
            Float("colsample_bylevel", (0.6, 1.0)),
            Float("colsample_bynode", (0.6, 1.0)),
            Float("reg_alpha", (1e-4, 5.0)),
            Float("reg_lambda", (1e-4, 5.0)),
            Categorical("grow_policy", ["depthwise", "lossguide"]),
            Integer("max_cat_to_onehot", (8, 100), log=True),
            Integer("max_leaves", (8, 1024), log=True),
            # todo: do we still need to set enable_categorical?
            # could search max_bin and num_parallel_tree but this is expensive

            Categorical('use_arithmetic_preprocessor', [True, False]),
            Categorical('use_cat_fe', [True, False]),
        ],
        seed=1234,
    )

    configs = search_space.sample_configuration(num_random_configs)
    if num_random_configs == 1:
        configs = [configs]
    configs = [dict(config) for config in configs]
    for c in configs:
        c["enable_categorical"] = True
    for i in range(len(configs)):
        if 'prep_params' not in configs[i]:
            configs[i]['prep_params'] = {}
        if configs[i].pop('use_arithmetic_preprocessor') == True:
            configs[i]['prep_params'].update({
                'ArithmeticFeatureGenerator': {}
            })

        if configs[i].pop('use_cat_fe') == True:
            configs[i]['prep_params'].update({
                'CategoricalInteractionFeatureGenerator': {}, 
                'OOFTargetEncodingFeatureGenerator': {}
                        })
    return [convert_numpy_dtypes(config) for config in configs]


gen_xgboost = CustomAGConfigGenerator(
    model_cls=PrepXGBoostModel,
    search_space_func=generate_configs_xgboost,
    manual_configs=[
        {
        'prep_params': {
            'ArithmeticFeatureGenerator': {},
            'CategoricalInteractionFeatureGenerator': {}, 
            'OOFTargetEncodingFeatureGenerator': {}
                    },
        },
    ],
)

if __name__ == "__main__":
    print(generate_configs_xgboost(3))
