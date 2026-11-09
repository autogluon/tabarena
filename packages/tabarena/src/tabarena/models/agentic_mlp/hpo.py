from __future__ import annotations

import numpy as np

from tabarena.models.agentic_mlp.model import AgenticMLPModel
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator

#: Module index choices per stage, the keys of ``bin.mlp.hparams.IMPLEMENTATION_REGISTRIES`` (gaps are
#: variants dropped from the package's search space). The fit keeps the ``_reg`` or ``_clf`` draw of its task.
_INDEX_CHOICES = {
    "numerical_preprocess_idx": (0, 1, 2, 3, 4, 5, 6, 7),
    "categorical_preprocess_idx": (0, 1, 2, 3),
    "target_preprocess_idx_reg": (0, 1, 2, 3, 4),
    "target_preprocess_idx_clf": (0,),
    "num_embedding_idx": (0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13),
    "cat_embedding_idx": (0, 1, 2, 3, 4, 5, 6, 7, 8),
    "model_idx": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    "train_idx": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12),
    "optimizer_idx": (0, 1),
    "inference_idx_reg": (0, 1, 2),
    "inference_idx_clf": (0, 1, 2, 3),
    "loss_idx_reg": (0, 1, 2),
    "loss_idx_clf": (0, 1),
}


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def generate_single_config_agentic_mlp(rng: np.random.Generator) -> dict:
    # Ranges follow ``bin.mlp.hparams.sample_hps``. Dropout and weight decay are drawn either way, so later
    # configs do not depend on the on/off switches.
    n_bins = rng.choice(np.arange(16, 129, 4))
    use_dropout = bool(rng.choice([True, False]))
    dropout = rng.uniform(0.0, 0.5)
    use_weight_decay = bool(rng.choice([True, False]))
    weight_decay = _log_uniform(rng, 1e-4, 1e-1)
    params = {
        "n_bins": n_bins,
        "dropout": dropout if use_dropout else 0.0,
        "weight_decay": weight_decay if use_weight_decay else 0.0,
        "lr": _log_uniform(rng, 3e-5, 1e-3),
        "n_blocks": rng.integers(1, 6, endpoint=True),
        "d_block": rng.choice(np.arange(64, 1025, 16)),
        "d_embedding": rng.choice(np.arange(8, 129, 4)),
    }
    for key, choices in _INDEX_CHOICES.items():
        params[key] = int(rng.choice(choices))
    return convert_numpy_dtypes(params)


def generate_configs_agentic_mlp(num_random_configs: int = 200, seed: int = 1234) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [generate_single_config_agentic_mlp(rng) for _ in range(num_random_configs)]


gen_agentic_mlp = CustomAGConfigGenerator(
    model_cls=AgenticMLPModel,
    search_space_func=generate_configs_agentic_mlp,
    manual_configs=[{}],
)
