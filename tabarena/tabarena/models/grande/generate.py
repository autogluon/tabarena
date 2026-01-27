from __future__ import annotations

import numpy as np

from tabarena.benchmark.experiment import YamlExperimentSerializer
from tabarena.benchmark.models.ag.grande.grande_model import GRANDEModel
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import (
    CustomAGConfigGenerator,
    generate_bag_experiments,
)


def generate_single_config_grande(rng):
    # common search space
    params = {
        "depth": rng.choice([3, 5], p=[0.5, 0.5]),
        "n_estimators": 1024,
        "learning_rate_weights": 0.001,  # np.exp(rng.uniform(np.log(1e-4), np.log(1e-1))),
        "learning_rate_index": 0.01,  # np.exp(rng.uniform(np.log(5e-4), np.log(1e-1))),
        "learning_rate_values": np.exp(rng.uniform(np.log(1e-3), np.log(5e-2))),
        "learning_rate_leaf": np.exp(rng.uniform(np.log(1e-2), np.log(5e-1))),
        "temperature": 0.0,
        "use_class_weights": False,  # rng.choice([True, False], p=[0.5, 0.5]), #False,
        "dropout": rng.choice([0.0, 0.2]),
        "selected_variables": 0.8,  # rng.choice([0.5, 0.8], p=[0.5, 0.5]),
        "data_subset_fraction": 1.0,  # rng.choice([0.5, 0.8, 1.0], p=[0.25, 0.25, 0.5]),#1.0, #rng.choice([0.5, 0.8, 1.0], p=[0.25, 0.25, 0.5]),
        "bootstrap": False,  # rng.choice([True, False], p=[0.5, 0.5]),
        "verbose": 0,
        "batch_size": 256,
        "early_stopping_epochs": 50,
        "epochs": 250,
        "focal_loss": False,  # rng.choice([True, False], p=[0.5, 0.5]),
        "es_metric": True,
        "missing_values": False,
        "swa": False,
        "cosine_decay_restarts": rng.choice([True, False], p=[0.5, 0.5]),
        "optimizer": "adam",
        "reduce_on_plateau_scheduler": True,
        "learning_rate_embedding": np.exp(rng.uniform(np.log(1e-4), np.log(5e-2))),
        "use_category_embeddings": rng.choice([True, False], p=[0.5, 0.5]),
        "embedding_dim_cat": 8,
        "use_numeric_embeddings": rng.choice([True, False], p=[0.5, 0.5]),
        "embedding_dim_num": 8,
        "embedding_threshold": 1,
        "label_smoothing": 0.00,  # rng.choice([0.00, 0.1], p=[0.5, 0.5]), #rng.choice([0.00, 0.1], p=[0.5, 0.5]),
        "use_robust_scale_smoothing": False,
    }

    return convert_numpy_dtypes(params)


def generate_configs_grande(num_random_configs=200, seed=1234):
    # note: this doesn't set val_metric_name, which should be set outside
    rng = np.random.default_rng(seed)
    return [generate_single_config_grande(rng) for _ in range(num_random_configs)]


gen_grande = CustomAGConfigGenerator(
    model_cls=GRANDEModel,
    search_space_func=generate_configs_grande,
    manual_configs=[{}],
)

if __name__ == "__main__":
    configs_yaml = []
    config_defaults = [{}]
    configs = generate_configs_grande(100, seed=1234)

    experiments_grande_streamlined = gen_grande.generate_all_bag_experiments(100)

    experiments_default = generate_bag_experiments(
        model_cls=GRANDEModel,
        configs=config_defaults,
        time_limit=3600,
        name_id_prefix="c",
    )
    experiments_random = generate_bag_experiments(
        model_cls=GRANDEModel, configs=configs, time_limit=3600
    )
    experiments = experiments_default + experiments_random
    YamlExperimentSerializer.to_yaml(
        experiments=experiments, path="configs_grande.yaml"
    )
