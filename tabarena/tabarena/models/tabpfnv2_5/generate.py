from __future__ import annotations

from autogluon.common.space import Categorical

from tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model import RealTabPFNv25Model
from tabarena.utils.config_utils import ConfigGenerator


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


search_space = {
    # Model Type
    "zip_model_path": Categorical(*_get_model_path_zip(RealTabPFNv25Model)),
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
}

gen_realtabpfnv25 = ConfigGenerator(
    model_cls=RealTabPFNv25Model,
    search_space=search_space,
    manual_configs=[{}],
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
