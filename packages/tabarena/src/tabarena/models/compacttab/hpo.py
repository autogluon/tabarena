from __future__ import annotations

import numpy as np

from tabarena.models.compacttab.model import CompactTabHybridModel, CompactTabModel
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator


def generate_single_config_compacttab(rng: np.random.Generator) -> dict:
    params = {
        "learning_rate": float(np.exp(rng.uniform(np.log(1e-4), np.log(3e-3)))),
        "weight_decay": float(rng.choice([1e-5, 1e-4, 1e-3, 1e-2])),
        "dropout": float(rng.choice([0.05, 0.10, 0.15, 0.20, 0.25])),
        "num_layers": int(rng.choice([2, 3, 4, 6])),
        "hidden_dim": int(rng.choice([64, 128, 256])),
        "embed_dim": int(rng.choice([12, 16, 24, 32])),
        "mixup_alpha": float(rng.choice([0.0, 0.05, 0.10, 0.20])),
        "max_epochs": 40,
        "batch_size": int(rng.choice([64, 128, 256])),
        "patience": 10,
    }
    return convert_numpy_dtypes(params)


def generate_configs_compacttab(num_random_configs: int = 200, seed: int = 1234) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [generate_single_config_compacttab(rng) for _ in range(num_random_configs)]


gen_compacttab = CustomAGConfigGenerator(
    model_cls=CompactTabModel,
    search_space_func=generate_configs_compacttab,
    manual_configs=[{}],
)

gen_compacttab_hybrid = CustomAGConfigGenerator(
    model_cls=CompactTabHybridModel,
    search_space_func=generate_configs_compacttab,
    manual_configs=[{}],
)
