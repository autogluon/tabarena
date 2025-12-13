

from __future__ import annotations

from tabarena.benchmark.models.prep_ag.prep_lgb.prep_lgb_model import PrepLGBModel
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from tabarena.benchmark.experiment import YamlExperimentSerializer
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator

import numpy as np

def normal_pdf_points(k, border_density=0.05):
    a = np.sqrt(-2*np.log(border_density*np.sqrt(2*np.pi)))
    x = np.linspace(-a, a, k)
    y = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)
    return list(y)

num_leaves = [8, 16, 20, 32, 64, 128, 200]

def generate_configs_lightgbm(num_random_configs=200) -> list:
    search_space = ConfigurationSpace(
        space=[
            Float("learning_rate", (5e-3, 1e-1), log=True),
            Float("feature_fraction", (0.4, 1.0)),
            Float("bagging_fraction", (0.7, 1.0)),
            Categorical("bagging_freq", [1]),
            # Integer("num_leaves", (2, 200), log=True),
            Categorical("num_leaves", num_leaves, weights=normal_pdf_points(len(num_leaves), border_density=0.05)),

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

            # Preprocessing hyperparameters
            Categorical("use_arithmetic_preprocessor", [True, False]),
            Categorical("use_cat_fe", [True, False]),
            Categorical("ag.use_residuals", [True, False]),
            Categorical("ag.residual_type", ['oof']),
            Categorical("ag.residual_init_kwargs", [{}]),
            Categorical("ag.max_dataset_size_for_residuals", [1000]), # NOTE: Currently: always only consider linear residuals for small datasets (N<1000)
        ],
        seed=1234,
    )

    configs = search_space.sample_configuration(num_random_configs)
    if num_random_configs == 1:
        configs = [configs]
    configs = [dict(config) for config in configs]

    for i in range(len(configs)):
        if 'ag.prep_params' not in configs[i]:
            configs[i]['ag.prep_params'] = []
        prep_params_stage_1 = []
        use_arithmetic_preprocessor = configs[i].pop('use_arithmetic_preprocessor')
        use_cat_fe = configs[i].pop('use_cat_fe')
        if use_arithmetic_preprocessor:
            _generator_params = {}
            prep_params_stage_1.append([
                ('ArithmeticFeatureGenerator', _generator_params),
            ])

        if use_cat_fe:
            prep_params_stage_1.append([
                ('CategoricalInteractionFeatureGenerator', {}),
                ('OOFTargetEncodingFeatureGenerator', {}),
            ])

        if prep_params_stage_1:
            configs[i]['ag.prep_params'].append(prep_params_stage_1)

    return [convert_numpy_dtypes(config) for config in configs]


gen_lightgbm = CustomAGConfigGenerator(
    model_cls=PrepLGBModel,
    search_space_func=generate_configs_lightgbm,
    manual_configs=[
        {
        'ag.prep_params': [
            [
                ('ArithmeticFeatureGenerator', {}),
                [
                    ('CategoricalInteractionFeatureGenerator', {"passthrough": True}),
                    ('OOFTargetEncodingFeatureGenerator', {}),
                ],
            ],
        ],
        'ag.prep_params.passthrough_types': {"invalid_raw_types": ["category", "object"]},
        'ag.use_residuals': True,
        'ag.residual_type': 'oof',
        'ag.max_dataset_size_for_residuals': 1000,
        'ag.residual_init_kwargs': {},
        },
    ],
)

if __name__ == "__main__":
    experiments = gen_lightgbm.generate_all_bag_experiments(num_random_configs=200)
    YamlExperimentSerializer.to_yaml(
        experiments=experiments, path="configs_lightgbm_alt.yaml"
    )
