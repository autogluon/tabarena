from __future__ import annotations

import numpy as np

from tabarena.models.agentic_realmlp.model import AgenticRealMLPModel
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator

#: Module index choices per stage, as in the package's ``bin/realmlp/search_space.toml``.
_INDEX_CHOICES = {
    "numerical_preprocess_idx": (0, 1, 2),
    "categorical_preprocess_idx": (0, 1, 2),
    "target_preprocess_idx": (0,),
    "num_embedding_idx": (0, 1),
    "cat_embedding_idx": (0, 1),
    "model_idx": (0, 1, 2),
    "train_idx": (0, 1, 2),
    "optimizer_idx": (0, 1),
    "inference_idx": (0,),
    "loss_idx": (0, 1),
}


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def generate_single_config_agentic_realmlp(rng: np.random.Generator) -> dict:
    # The large-config values are drawn either way, so later configs do not depend on ``is_large``.
    params = {
        "n_hidden_layers": rng.integers(2, 4, endpoint=True),
        "hidden_sizes": "rectangular",
        "hidden_width": rng.choice([256, 384, 512]),
        "p_drop": rng.uniform(0.0, 0.5),
        "act": "mish",
        "plr_sigma": _log_uniform(rng, 1e-2, 50.0),
        "plr_lr_factor": _log_uniform(rng, 5e-2, 3e-1),
        "scale_lr_factor": _log_uniform(rng, 2.0, 10.0),
        "first_layer_lr_factor": _log_uniform(rng, 0.3, 1.5),
        "ls_eps_sched": "coslog4",
        "ls_eps": _log_uniform(rng, 5e-3, 1e-1),
        "p_drop_sched": "flat_cos",
        "lr": _log_uniform(rng, 2e-2, 3e-1),
        "wd": _log_uniform(rng, 1e-3, 5e-2),
        "use_ls": rng.choice([False, True]),
        "embedding_size": rng.choice([4, 8, 16]),
        "n_ens": 8,
        "ens_av_before_softmax": False,
        "one_minus_sq_mom": _log_uniform(rng, 5e-3, 5e-2),
        "max_one_hot_cat_size_raw": _log_uniform(rng, 4.0, 33.0),
    }
    is_large = bool(rng.choice([False, True]))
    large_plr_hidden_1 = rng.choice([8, 16, 32, 64])
    large_plr_hidden_2 = rng.choice([8, 16, 32, 64])
    large_n_epochs = rng.choice([256, 512])
    params.update(
        is_large=is_large,
        plr_hidden_1=large_plr_hidden_1 if is_large else 16,
        plr_hidden_2=large_plr_hidden_2 if is_large else 4,
        n_epochs=large_n_epochs if is_large else 256,
        use_early_stopping=is_large,
        early_stopping_multiplicative_patience=3,
        early_stopping_additive_patience=40,
    )
    for key, choices in _INDEX_CHOICES.items():
        params[key] = rng.choice(choices)
    return convert_numpy_dtypes(params)


def generate_configs_agentic_realmlp(num_random_configs: int = 200, seed: int = 1234) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [generate_single_config_agentic_realmlp(rng) for _ in range(num_random_configs)]


gen_agentic_realmlp = CustomAGConfigGenerator(
    model_cls=AgenticRealMLPModel,
    search_space_func=generate_configs_agentic_realmlp,
    manual_configs=[{}],
)
