from __future__ import annotations

from autogluon.common.space import Categorical

from tabarena.benchmark.models.prep_ag.prep_tabpfnv2_5.prep_tabpfnv2_5_model import PrepRealTabPFNv25Model
from tabarena.utils.config_utils import CustomAGConfigGenerator
from tabarena.models.utils import convert_numpy_dtypes

import numpy as np

def _get_model_path_zip(model_cls):
    # Zip model paths to ensure configs are not generated that only differ in combination
    clf_models = model_cls.extra_checkpoints_for_tuning("classification")
    reg_models = model_cls.extra_checkpoints_for_tuning("regression")
    zip_model_paths = [
        [model_cls.default_classification_model, model_cls.default_regression_model],
    ]
    n_clf_models = len(clf_models)
    n_reg_models = len(reg_models)
    for i in range(max(n_clf_models, n_reg_models)):
        zip_model_paths.append(
            [clf_models[min(i, n_clf_models - 1)], reg_models[min(i, n_reg_models - 1)]]
        )

    return zip_model_paths

def generate_single_config_tabpfn(rng):
    # taken from
    search_space = {
        # Model Type
        "zip_model_path": Categorical(*_get_model_path_zip(PrepRealTabPFNv25Model)),
        "softmax_temperature": Categorical(
            0.25,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.25,
            1.5,
        ),
        "balance_probabilities": Categorical(True, False),
        "inference_config/OUTLIER_REMOVAL_STD": Categorical(3, 6, 12),
        "inference_config/POLYNOMIAL_FEATURES": Categorical("no", 25),
        "inference_config/REGRESSION_Y_PREPROCESS_TRANSFORMS": Categorical(
            [None],
            [None, "safepower"],
            ["safepower"],
            ["kdi_alpha_0.3"],
            ["kdi_alpha_1.0"],
            ["kdi_alpha_3.0"],
            ["quantile_uni"],
        ),
        # Preprocessing
        "preprocessing/scaling": Categorical(
            ["none"],
            ["quantile_uni_coarse"],
            ["quantile_norm_coarse"],
            ["kdi_uni"],
            ["kdi_alpha_0.3"],
            ["kdi_alpha_3.0"],
            ["safepower", "quantile_uni"],
            ["none", "quantile_uni_coarse"],
            ["squashing_scaler_default", "quantile_uni_coarse"],
            ["squashing_scaler_default"],
        ),
        "preprocessing/categoricals": Categorical(
            "numeric",
            "onehot",
            "none",
        ),
        "preprocessing/append_original": Categorical(True, False),
        "preprocessing/global": Categorical(None, "svd", "svd_quarter_components"),

        'use_arithmetic_preprocessor': rng.choice([True, False]),
        'use_cat_fe': rng.choice([True, False]),


    }
    return convert_numpy_dtypes(search_space) 

def generate_configs_tabpfn(num_random_configs=200, seed=1234):
    # note: this doesn't set val_metric_name, which should be set outside
    rng = np.random.default_rng(seed)
    configs = [generate_single_config_tabpfn(rng) for _ in range(num_random_configs)]
    
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
    
    return configs

gen_realtabpfnv25 = CustomAGConfigGenerator(
    model_cls=PrepRealTabPFNv25Model, search_space_func=generate_configs_tabpfn, 
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
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_realtabpfnv25.generate_all_bag_experiments(
                num_random_configs=200
            ),
        )
    )
