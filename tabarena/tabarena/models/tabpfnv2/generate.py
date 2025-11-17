from __future__ import annotations

import numpy as np
from hyperopt import hp
from hyperopt.pyll import stochastic

from tabarena.benchmark.models.ag.tabpfnv2.tabpfnv2_model import TabPFNV2Model
from tabarena.utils.config_utils import CustomAGConfigGenerator


def _enumerate_preprocess_transforms():
    """From: https://github.com/PriorLabs/tabpfn-extensions."""
    transforms = []

    names_list = [
        ["safepower"],
        ["quantile_uni_coarse"],
        ["quantile_norm_coarse"],
        ["norm_and_kdi"],
        ["quantile_uni"],
        ["none"],
        ["robust"],
        ["kdi_uni"],
        ["kdi_alpha_0.3"],
        ["kdi_alpha_3.0"],
        ["safepower", "quantile_uni"],
        ["kdi", "quantile_uni"],
        ["none", "power"],
    ]

    for names in names_list:
        for categorical_name in [
            "numeric",
            "ordinal_very_common_categories_shuffled",
            "onehot",
            "none",
        ]:
            for append_original in [True, False]:
                for subsample_features in [-1, 0.99, 0.95, 0.9]:
                    for global_transformer_name in [None, "svd"]:
                        transforms += [
                            [
                                {
                                    # Use "name" parameter as expected by TabPFN PreprocessorConfig
                                    "name": name,
                                    "global_transformer_name": global_transformer_name,
                                    "subsample_features": subsample_features,
                                    "categorical_name": categorical_name,
                                    "append_original": append_original,
                                }
                                for name in names
                            ],
                        ]
    return transforms


def get_unified_param_grid_hyperopt() -> dict:
    """Generate the full hyperopt search space for TabPFN optimization from the Nature paper.

    From: https://github.com/PriorLabs/tabpfn-extensions

    Args:
        task_type: Either "multiclass" or "regression"

    Returns:
        Hyperopt search space dictionary
    """
    search_space = {
        # Custom HPs
        "model_type": hp.choice(
            "model_type",
            ["single", "dt_pfn"],
        ),
        "n_ensemble_repeats": hp.choice("n_ensemble_repeats", [4]),
        # -- Model HPs
        "average_before_softmax": hp.choice("average_before_softmax", [True, False]),
        "softmax_temperature": hp.choice(
            "softmax_temperature",
            [
                0.75,
                0.8,
                0.9,
                0.95,
                1.0,
            ],
        ),
        # Inference config
        "inference_config/FINGERPRINT_FEATURE": hp.choice(
            "FINGERPRINT_FEATURE",
            [True, False],
        ),
        "inference_config/PREPROCESS_TRANSFORMS": hp.choice(
            "PREPROCESS_TRANSFORMS",
            _enumerate_preprocess_transforms(),
        ),
        "inference_config/POLYNOMIAL_FEATURES": hp.choice(
            "POLYNOMIAL_FEATURES",
            ["no", 50],
        ),
        "inference_config/OUTLIER_REMOVAL_STD": hp.choice(
            "OUTLIER_REMOVAL_STD",
            [None, 7.0, 9.0, 12.0],
        ),
        "inference_config/SUBSAMPLE_SAMPLES": hp.choice(
            "SUBSAMPLE_SAMPLES",
            [0.99, None],
        ),
    }

    search_space["classification_model_path"] = hp.choice(
        "classification_model_path",
        [
            "tabpfn-v2-classifier.ckpt",
            "tabpfn-v2-classifier-od3j1g5m.ckpt",
            "tabpfn-v2-classifier-gn2p4bpt.ckpt",
            "tabpfn-v2-classifier-znskzxi4.ckpt",
            "tabpfn-v2-classifier-llderlii.ckpt",
            "tabpfn-v2-classifier-vutqq28w.ckpt",
        ],
    )

    search_space["regression_model_path"] = hp.choice(
        "regression_model_path",
        [
            "tabpfn-v2-regressor-09gpqh39.ckpt",
            "tabpfn-v2-regressor.ckpt",
            "tabpfn-v2-regressor-2noar4o2.ckpt",
            "tabpfn-v2-regressor-wyl4o83o.ckpt",
            "tabpfn-v2-regressor-5wof9ojf.ckpt",
        ],
    )

    search_space["inference_config/REGRESSION_Y_PREPROCESS_TRANSFORMS"] = hp.choice(
        "REGRESSION_Y_PREPROCESS_TRANSFORMS",
        [
            [None],
            [None, "power"],
            ["power"],
            ["safepower"],
            ["kdi_alpha_0.3"],
            ["kdi_alpha_1.0"],
            ["kdi_alpha_1.5"],
            ["kdi_alpha_0.6"],
            ["kdi_alpha_3.0"],
            ["quantile_uni"],
        ],
    )

    return search_space


def prepare_tabpfnv2_config(raw_config: dict, *, refit_folds: bool = True) -> dict:
    """Set refit folds to True and convert tuples to lists."""
    raw_config = {
        k: list(v) if isinstance(v, tuple) else v for k, v in raw_config.items()
    }
    return raw_config


def search_space_func(num_random_configs: int = 200, seed=1234) -> list[dict]:
    search_space = get_unified_param_grid_hyperopt()
    rng = np.random.default_rng(seed)
    stochastic.sample(search_space, rng=rng)
    return [
        prepare_tabpfnv2_config(dict(stochastic.sample(search_space, rng=rng)))
        for _ in range(num_random_configs)
    ]


gen_tabpfnv2 = CustomAGConfigGenerator(
    model_cls=TabPFNV2Model,
    search_space_func=search_space_func,
    manual_configs=[{}],
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabpfnv2.generate_all_bag_experiments(
                num_random_configs=200
            ),
        ),
    )
