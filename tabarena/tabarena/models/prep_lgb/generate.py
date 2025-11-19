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

            # Float("feature_fraction", (0.4, 1.0)),
            # NOTE: With feature engineering, we can probably add more aggressive feature bagging
            Categorical("feature_fraction", [0.1,0.25,0.5,0.75,1.0]),

            Float("bagging_fraction", (0.7, 1.0)),
            Categorical("bagging_freq", [1]),

            # Integer("num_leaves", (2, 200), log=True),
            # NOTE: Smaller num_leaves is not necessarily desirable and often hurts if a dataset has fine-granular patterns, i.e. high-cardinality numericals.
            # Hence, we shouldn't use log. Use a weighted grid instead.
            Categorical("num_leaves", num_leaves, weights=normal_pdf_points(len(num_leaves), border_density=0.10)),

            # Integer("min_data_in_leaf", (1, 64), log=True),
            # NOTE: Smaller min_data_in_leaf is not necessarily better, so use a weighted grid
            Categorical("min_data_in_leaf", [1,4,8,16,32,64,128], weights=normal_pdf_points(7, border_density=0.10)),

            # NOTE: Unsure whether we still need extra_trees if we have linear init
            Categorical("extra_trees", [False, True]),

            # categorical hyperparameters
            # NOTE: Likely not that important anymore since it mainly concerns high-cardinality categoricals which we expect to be handled via preprocessing
            Integer("min_data_per_group", (2, 100), log=True),
            Float("cat_l2", (5e-3, 2), log=True),
            Float("cat_smooth", (1e-3, 100), log=True),
            Integer("max_cat_to_onehot", (8, 100), log=True),

            # these seem to help a little bit but can also make things slower
            # Float("lambda_l1", (1e-4,  1.0)),
            # Float("lambda_l2", (1e-4,  2.0)),
            # NOTE: With more features added in the preprocessing, we can test stronger regularization
            Categorical("lambda_l1", [1e-4, 1e-2, 1, 2, 5, 10.0]),# 1.0)),
            Categorical("lambda_l2", [1e-4, 1e-2, 1, 2, 5, 10.0]),# 2.0)),
            # could search max_bin but this is expensive

            # Preprocessing hyperparameters
            Categorical("use_arithmetic_preprocessor", [True, False]),
            Categorical("use_cat_fe", [True, False]),
            Categorical("use_residuals", [True, False]),
            Categorical("residual_type", ['oof']),
            Categorical("residual_init_kwargs", [residual_init_kwargs]),
            Categorical("max_dataset_size_for_residuals", [1000]), # NOTE: Currently: always only consider linear residuals for small datasets (N<1000)

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
        'preset_name': 'all_preprocessors',
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
        {
        'preset_name': 'only_arithmetic',
        'prep_params': {
            'ArithmeticPreprocessor': {'cat_as_num': False},
                    },
        'use_residuals': False,
        'residual_type': 'oof',
        'max_dataset_size_for_residuals': 1000,
        'residual_init_kwargs': residual_init_kwargs,
        },
        {
        'preset_name': 'only_cat_fe',
        'prep_params': {
            'CatIntAdder': cat_int_kwargs, 
            'OOFTargetEncoder': {'alpha': 10}
                    },
        'use_residuals': False,
        'residual_type': 'oof',
        'max_dataset_size_for_residuals': 1000,
        'residual_init_kwargs': residual_init_kwargs,
        },        
        {
        'preset_name': 'only_linear_residuals',
        'prep_params': {},
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

