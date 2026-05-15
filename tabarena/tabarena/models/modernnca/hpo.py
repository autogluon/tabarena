from __future__ import annotations

import numpy as np

from tabarena.benchmark.models.ag.modernnca.modernnca_model import ModernNCAModel
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator


def generate_single_config_modernnca(rng):
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
        "weight_decay": rng.choice(
            [0.0, np.exp(rng.uniform(np.log(1e-6), np.log(1e-3)))]
        ),
        "temperature": 1.0,
        "num_emb_type": "plr",
        "num_emb_lite": True,
        "batch_size": "auto",
    }

    return convert_numpy_dtypes(params)


def generate_configs_modernnca(num_random_configs=200, seed=1234):
    rng = np.random.default_rng(seed)
    return [generate_single_config_modernnca(rng) for _ in range(num_random_configs)]


gen_modernnca = CustomAGConfigGenerator(
    model_cls=ModernNCAModel,
    search_space_func=generate_configs_modernnca,
    manual_configs=[{}],
)
