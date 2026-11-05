from __future__ import annotations

import numpy as np

from tabarena.models.agentic_tabm.model import AgenticTabMModel
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator

#: Module index choices per stage, as registered in the package's ``bin/tabm/modules``. The fit keeps the
#: ``_reg`` or ``_clf`` draw of its task.
_INDEX_CHOICES = {
    "numerical_preprocess_idx": tuple(range(9)),
    "categorical_preprocess_idx": (0, 1, 2),
    "target_preprocess_idx_reg": (0, 1, 2, 3),
    "target_preprocess_idx_clf": (0,),
    "num_embedding_idx": tuple(range(9)),
    "cat_embedding_idx": (0, 1, 2, 3, 4),
    "model_idx": tuple(range(15)),
    "train_idx": tuple(range(9)),
    "optimizer_idx": (0, 1),
    "inference_idx_reg": tuple(range(8)),
    "inference_idx_clf": (0, 1, 2),
    "loss_idx_reg": (0, 1),
    "loss_idx_clf": (0, 1),
}


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def generate_single_config_agentic_tabm(rng: np.random.Generator) -> dict:
    # TabArena's TabM search space (``tabarena.models.tabm.hpo``) in its draw order, plus the stage indices.
    params = {
        "gradient_clip": 1.0,
        "max_epochs": 256,
        "patience": 16,
        "eval_batch_size": 32768,
        "k": 32,
        "lr": _log_uniform(rng, 1e-4, 3e-3),
        "weight_decay": rng.choice([0.0, _log_uniform(rng, 1e-4, 1e-1)]),
        "n_blocks": rng.choice([2, 3, 4, 5]),
        "d_block": rng.choice(np.arange(128, 1024 + 1, 16)),
        "dropout": rng.choice([0.0, rng.uniform(0.0, 0.5)]),
        "d_embedding": rng.choice(np.arange(8, 32 + 1, 4)),
        "n_bins": rng.integers(2, 128, endpoint=True),
    }
    for key, choices in _INDEX_CHOICES.items():
        params[key] = rng.choice(choices)
    return convert_numpy_dtypes(params)


def generate_configs_agentic_tabm(num_random_configs: int = 200, seed: int = 1234) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [generate_single_config_agentic_tabm(rng) for _ in range(num_random_configs)]


gen_agentic_tabm = CustomAGConfigGenerator(
    model_cls=AgenticTabMModel,
    search_space_func=generate_configs_agentic_tabm,
    manual_configs=[{}],
)
