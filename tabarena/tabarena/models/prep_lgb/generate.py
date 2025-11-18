from __future__ import annotations

from tabarena.benchmark.models.ag.prep_lgb.prep_lgb_model import PrepLGBModel
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from tabarena.benchmark.experiment import YamlExperimentSerializer
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator

import numpy as np

'''
Todos:
- Adjust num_leaves
- Check whether we still need extra_trees if we have linear init
- Add arithmetic
- Add linear init
- Add cat FE
- Add OOF TE
'''

cat_int_kwargs = {'max_order': 3, 'min_cardinality': 2, 'add_freq': False, 'use_filters': False, 'max_base_interactions': 100}
residual_init_kwargs = {'scaler': 'squashing', 'linear_model_type': 'lasso', 'lambda_': 'medium', 'cat_method': 'ohe'}

def normal_pdf_points(k, border_density=0.05):
    a = np.sqrt(-2*np.log(border_density*np.sqrt(2*np.pi)))
    x = np.linspace(-a, a, k)
    y = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)
    return list(y)

num_leaves = [8, 16, 32, 64, 128, 256, 512]

def generate_configs_lightgbm(num_random_configs=200) -> list:
    search_space = ConfigurationSpace(
        space=[
            Float("learning_rate", (5e-3, 1e-1), log=True),
            Float("feature_fraction", (0.4, 1.0)),
            Float("bagging_fraction", (0.7, 1.0)),
            Categorical("bagging_freq", [1]),
            # Integer("num_leaves", (2, 200), log=True),
            Categorical("num_leaves", num_leaves, weights=normal_pdf_points(len(num_leaves), border_density=0.10)),
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
            Categorical("use_linear_residuals", [True]),
            Categorical("use_cat_fe", [True, False]),
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
        if configs[i].pop('use_arithmetic_preprocessor') == True:
            configs[i]['prep_params'].update({
                'ArithmeticPreprocessor': {'cat_as_num': False}
            })

        # Currently: always train on linear residuals for small datasets (N<1000)
        if configs[i].pop('use_linear_residuals') == True:
            configs[i]['use_residuals'] = True
            configs[i]['residual_type'] = 'oof'
            configs[i]['max_dataset_size_for_residuals'] = 1000
            configs[i]['residual_init_kwargs'] = residual_init_kwargs

        if configs[i].pop('use_cat_fe') == True:
            configs[i]['prep_params'].update({
                'CatIntAdder': cat_int_kwargs, 
                'OOFTargetEncoder': {'alpha': 10}
                        })

    return [convert_numpy_dtypes(config) for config in configs]


gen_lightgbm = CustomAGConfigGenerator(
    model_cls=PrepLGBModel,
    search_space_func=generate_configs_lightgbm,
    manual_configs=[
        {
        'preset_name': 'arithmetic_catfe_oofte_linresoof',
        'prep_params': {
            'ArithmeticPreprocessor': {'cat_as_num': False},
            'CatIntAdder': cat_int_kwargs, 
            'OOFTargetEncoder': {'alpha': 10}
                    },
        'use_residuals': True,
        'residual_type': 'oof',
        'max_dataset_size_for_residuals': 1000,
        'residual_init_kwargs': residual_init_kwargs,
        },
    ],
)

if __name__ == "__main__":
    experiments = gen_lightgbm.generate_all_bag_experiments(num_random_configs=200)
    YamlExperimentSerializer.to_yaml(
        experiments=experiments, path="configs_lightgbm_alt.yaml"
    )

