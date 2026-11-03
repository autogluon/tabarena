from __future__ import annotations

import numpy as np

from tabarena.models.iltm.model import ILTMModel
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator


def generate_single_config_iltm(rng):
    params = {
        "checkpoint": rng.choice(["cbrconcat", "xgbrconcat", "rtr"], p=[0.7, 0.2, 0.1]),
        "checkpoint_mix": rng.choice([None, "default"], p=[0.4, 0.6]),
        "n_ensemble": rng.choice([8, 12]),
        "batch_size": rng.choice([2048, 4096]),
        "finetuning": True,
        "finetuning_dropout": rng.choice([0.0, 0.15], p=[0.6, 0.4]),
        "finetuning_max_steps": 2048,
        "finetuning_batch_size": rng.choice([1024, 2048]),
        "finetuning_data": "entire_dataset",
        "finetuning_lr_adamw": np.exp(rng.uniform(np.log(1.7e-4), np.log(5.5e-4))),
        "finetuning_lr_lion": np.exp(rng.uniform(np.log(2e-4), np.log(1.5e-3))),
        "gradient_clip_norm": rng.uniform(0.5, 1.5),
        "finetuning_optimizer": rng.choice(["adamw", "lion"], p=[0.4, 0.6]),
        "max_train_batches_per_epoch": 128,
        "finetuning_subset_frac": None,
        "finetuning_subset_max_samples": 100_000,
        "val_max_samples": 25_000,
        "tree_data_split": "all",
        "tree_for_each_predictor": True,
        "tree_n_estimators": rng.choice([100, 125, 150, 200], p=[0.1, 0.25, 0.15, 0.5]),
        "tree_lr": rng.uniform(5e-2, 0.7),
        "tree_max_depth": rng.choice([4, 5], p=[0.3, 0.7]),
        "tree_min_samples_leaf": rng.choice([1, 4, 12], p=[0.1, 0.05, 0.85]),
        "tree_l2_leaf_reg": rng.choice([0.75, 1.0, 1.25, 2.0, 2.5, 3.0]),
        "tree_subsample": rng.choice([0.56, 0.63, 0.71], p=[0.1, 0.85, 0.05]),
        "tree_feature_fraction": rng.choice([0.65, 0.81, 0.99], p=[0.05, 0.1, 0.85]),
        "tree_gamma": rng.choice([0.0, 0.5], p=[0.9, 0.1]),
        "tree_bagging_temperature": None,
        "do_retrieval": True,
        "retrieval_alpha": 0.75,
        "retrieval_temperature": rng.uniform(1.0, 2.0),
        "retrieval_distance": "cosine",
        "retrieval_alpha_finetuning": False,
        "retrieval_alpha_adaptive": True,
        "retrieval_temperature_finetuning": False,
        "clip_data_value": 1_000_000,
        "rf_size": 32_768,
        "pca_sampling": "zeropad",
        "scheduler_min_lr": np.exp(rng.uniform(np.log(1e-7), np.log(2e-6))),
        "clip_predictions": rng.choice([False, True], p=[0.3, 0.7]),
        "corr_select_k": rng.choice([0, 300, 512, 1024, 2048], p=[0.6, 0.05, 0.1, 0.15, 0.1]),
    }
    return convert_numpy_dtypes(params)


def generate_configs_iltm(num_random_configs: int = 200, seed: int = 1234) -> list[dict]:
    if num_random_configs < 0:
        raise ValueError("num_random_configs must be non-negative")
    if num_random_configs == 0:
        return []

    rng = np.random.default_rng(seed)
    return [generate_single_config_iltm(rng) for _ in range(num_random_configs)]


gen_iltm = CustomAGConfigGenerator(
    model_cls=ILTMModel,
    search_space_func=generate_configs_iltm,
    manual_configs=[{}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_iltm.generate_all_bag_experiments(num_random_configs=200),
        ),
    )
