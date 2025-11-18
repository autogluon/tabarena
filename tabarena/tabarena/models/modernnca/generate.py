from __future__ import annotations

import numpy as np

from tabarena.benchmark.experiment import YamlExperimentSerializer
from tabarena.benchmark.models.ag.modernnca.modernnca_model import ModernNCAModel
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator, generate_bag_experiments


def generate_single_config_modernnca(rng):
    # common search space
    params = {
        "dropout": rng.uniform(0.0, 0.5),
        "d_block": rng.integers(64, 1024, endpoint=True),
        "n_blocks": rng.choice([0, rng.integers(0, 2, endpoint=True)]),
        "dim": rng.integers(64, 1024, endpoint=True),
        "num_emb_n_frequencies": rng.integers(16, 96, endpoint=True),
        "num_emb_frequency_scale": np.exp(rng.uniform(np.log(0.005), np.log(10.0))),
        "num_emb_d_embedding": rng.integers(16, 64, endpoint=True),
        "sample_rate": rng.uniform(0.05, 0.6),
        "lr": np.exp(rng.uniform(np.log(1e-5), np.log(1e-1))),
        "weight_decay": rng.choice([0.0, np.exp(rng.uniform(np.log(1e-6), np.log(1e-3)))]),
        "temperature": 1.0,
        "num_emb_type": "plr",
        "num_emb_lite": True,
        "batch_size": "auto",
    }

    return convert_numpy_dtypes(params)


def generate_configs_modernnca(num_random_configs=200, seed=1234):
    # note: this doesn't set val_metric_name, which should be set outside
    rng = np.random.default_rng(seed)
    return [generate_single_config_modernnca(rng) for _ in range(num_random_configs)]


gen_modernnca = CustomAGConfigGenerator(
    model_cls=ModernNCAModel,
    search_space_func=generate_configs_modernnca,
    manual_configs=[{}],
)

if __name__ == "__main__":
    configs_yaml = []
    config_defaults = [{}]
    configs = generate_configs_modernnca(100, seed=1234)

    experiments_realmlp_streamlined = gen_modernnca.generate_all_bag_experiments(100)

    experiments_default = generate_bag_experiments(
        model_cls=ModernNCAModel,
        configs=config_defaults,
        time_limit=3600,
        name_id_prefix="c",
    )
    experiments_random = generate_bag_experiments(model_cls=ModernNCAModel, configs=configs, time_limit=3600)
    experiments = experiments_default + experiments_random
    YamlExperimentSerializer.to_yaml(experiments=experiments, path="configs_modernnca.yaml")
