from __future__ import annotations

import numpy as np

from tabarena.benchmark.models.ag.tabm.tabm_model import TabMModel
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator

name = "TabM"


def generate_single_config_tabm(rng):
    # taken from https://github.com/yandex-research/tabm/blob/main/exp/tabm-mini-piecewiselinear/adult/0-tuning.toml
    # discussed with the authors
    params = {
        "batch_size": "auto",
        "patience": 16,
        "amp": False,  # only for GPU, maybe we should change it to True?
        "arch_type": "tabm-mini",
        "tabm_k": 32,
        "gradient_clipping_norm": 1.0,
        # this makes it probably slower with numerical embeddings, and also more RAM intensive
        # according to the paper it's not very important but should be a bit better (?)
        "share_training_batches": False,
        "lr": np.exp(rng.uniform(np.log(1e-4), np.log(3e-3))),
        "weight_decay": rng.choice(
            [0.0, np.exp(rng.uniform(np.log(1e-4), np.log(1e-1)))]
        ),
        # removed n_blocks=1 according to Yury Gurishniy's advice
        "n_blocks": rng.choice([2, 3, 4, 5]),
        # increased lower limit from 64 to 128 according to Yury Gorishniy's advice
        "d_block": rng.choice([i for i in range(128, 1024 + 1) if i % 16 == 0]),
        "dropout": rng.choice([0.0, rng.uniform(0.0, 0.5)]),
        # numerical embeddings
        "num_emb_type": "pwl",
        "d_embedding": rng.choice([i for i in range(8, 32 + 1) if i % 4 == 0]),
        "num_emb_n_bins": rng.integers(2, 128, endpoint=True),
        # could reduce eval_batch_size in case of OOM
    }

    return convert_numpy_dtypes(params)


def generate_configs_tabm(num_random_configs=200, seed=1234):
    # note: this doesn't set val_metric_name, which should be set outside
    rng = np.random.default_rng(seed)
    return [generate_single_config_tabm(rng) for _ in range(num_random_configs)]


gen_tabm = CustomAGConfigGenerator(
    model_cls=TabMModel, search_space_func=generate_configs_tabm, manual_configs=[{}]
)
