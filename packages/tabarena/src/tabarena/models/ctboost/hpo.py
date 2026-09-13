"""Frozen TabArena search portfolio for CTBoost."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from tabarena.models.ctboost.model import CTBoostModel
from tabarena.utils.config_utils import CustomAGConfigGenerator

TABARENA_SEARCH_PORTFOLIO_SIZE = 200
_SEARCH_PORTFOLIO_SEED = 1234
_PAIR_BUDGET_PARAM = "tabarena_categorical_pair_budget"


def _adaptive_training_budget(learning_rate: float) -> tuple[int, int]:
    """Return a bounded tree cap and validation patience for a learning rate."""
    resolved = float(learning_rate)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    iterations = int(round(min(1600.0, max(400.0, 50.0 / resolved)) / 50.0) * 50)
    patience = round(min(80.0, max(30.0, 0.075 * iterations)))
    return iterations, patience


def _finalize_search_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve internal conditional knobs into valid CTBoost parameters."""
    resolved = {name: value.item() if isinstance(value, np.generic) else value for name, value in config.items()}
    leaf_fraction = resolved.pop("__leaf_fraction", None)
    if leaf_fraction is None:
        resolved["max_leaves"] = 0
    else:
        full_leaf_count = 1 << int(resolved["max_depth"])
        resolved["max_leaves"] = max(
            4,
            min(full_leaf_count - 1, round(float(leaf_fraction) * full_leaf_count)),
        )

    pair_budget = int(resolved.pop("__categorical_pair_budget", 0))
    if pair_budget and resolved.get("ordered_ctr", False):
        resolved[_PAIR_BUDGET_PARAM] = pair_budget

    learning_rate = resolved.get("learning_rate")
    if learning_rate is not None:
        iterations, early_stopping_rounds = _adaptive_training_budget(float(learning_rate))
        resolved.setdefault("iterations", iterations)
        resolved.setdefault("early_stopping_rounds", early_stopping_rounds)
    return resolved


def _stratified_unit_samples(count: int, dimensions: int) -> np.ndarray:
    """Create a progressively ordered deterministic Latin-hypercube design."""
    rng = np.random.default_rng(_SEARCH_PORTFOLIO_SEED)
    samples = np.empty((count, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        strata = rng.permutation(count)
        samples[:, dimension] = (strata + rng.random(count)) / count

    remaining = np.ones(count, dtype=bool)
    minimum_distance = np.sum(np.square(samples - 0.5), axis=1)
    order: list[int] = []
    for _ in range(count):
        candidate_scores = np.where(remaining, minimum_distance, -1.0)
        selected = int(np.argmax(candidate_scores))
        order.append(selected)
        remaining[selected] = False
        distances = np.sum(np.square(samples - samples[selected]), axis=1)
        minimum_distance = np.minimum(minimum_distance, distances)
    return samples[order]


def _linear_sample(value: float, lower: float, upper: float) -> float:
    return float(lower * (1.0 - value) + upper * value)


_FROZEN_LOG_SAMPLE_OVERRIDES = {
    # Windows and glibc libm differ by one ULP for these two frozen design
    # points. Canonicalize only them so the portfolio hash is cross-platform.
    (
        "0x1.4dd35420fef52p-4",
        "0x1.999999999999ap-4",
        "0x1.4000000000000p+3",
    ): float.fromhex("0x1.2a141a19178d4p-3"),
    (
        "0x1.61f66383ab081p-1",
        "0x1.47ae147ae147bp-8",
        "0x1.0000000000000p-1",
    ): float.fromhex("0x1.ee4e4f97673a6p-4"),
}


def _log_sample(value: float, lower: float, upper: float) -> float:
    canonical = _FROZEN_LOG_SAMPLE_OVERRIDES.get((float(value).hex(), float(lower).hex(), float(upper).hex()))
    if canonical is not None:
        return canonical
    return float(math.exp(math.log(lower) * (1.0 - value) + math.log(upper) * value))


def _integer_sample(value: float, lower: int, upper: int) -> int:
    return min(upper, lower + int(value * (upper - lower + 1)))


def _log_integer_sample(value: float, lower: int, upper: int) -> int:
    return min(upper, max(lower, round(_log_sample(value, lower, upper))))


def _categorical_sample(value: float, choices: Sequence[Any]) -> Any:
    return choices[min(len(choices) - 1, int(value * len(choices)))]


def generate_configs_ctboost(num_random_configs: int = 200) -> list[dict[str, Any]]:
    """Generate the frozen, deterministic 200-config CTBoost portfolio."""
    count = int(num_random_configs)
    if count < 0:
        raise ValueError("num_random_configs must be non-negative")
    if count == 0:
        return []
    if count > TABARENA_SEARCH_PORTFOLIO_SIZE:
        raise ValueError(
            f"num_random_configs cannot exceed the frozen {TABARENA_SEARCH_PORTFOLIO_SIZE}-config portfolio"
        )

    samples = _stratified_unit_samples(TABARENA_SEARCH_PORTFOLIO_SIZE, dimensions=17)[:count]
    configs: list[dict[str, Any]] = []
    for values in samples:
        ordered_ctr = bool(values[12] >= 0.25)
        grow_policy = str(_categorical_sample(values[6], ["DepthWise", "LeafWise"]))
        config: dict[str, Any] = {
            "learning_rate": _log_sample(values[0], 2e-2, 2e-1),
            "max_depth": _integer_sample(values[1], 3, 8),
            "alpha": _log_sample(values[2], 5e-3, 5e-1),
            "lambda_l2": _log_sample(values[3], 1e-4, 10.0),
            "subsample": _linear_sample(values[4], 0.6, 1.0),
            "colsample_bytree": _linear_sample(values[5], 0.6, 1.0),
            "grow_policy": grow_policy,
            "min_data_in_leaf": _log_integer_sample(values[8], 1, 64),
            "min_child_weight": _categorical_sample(values[9], [0.0, 0.01, 0.1, 1.0, 5.0]),
            "one_hot_max_size": _categorical_sample(values[10], [2, 4, 16, 64]),
            "max_cat_threshold": _categorical_sample(values[11], [16, 64, 256]),
            "ordered_ctr": ordered_ctr,
            "random_strength": _categorical_sample(values[14], [0.0, 0.01, 0.1, 1.0]),
            "max_bins": _categorical_sample(values[15], [128, 256]),
            "bootstrap_type": "Bernoulli",
        }
        if grow_policy == "LeafWise":
            config["__leaf_fraction"] = _categorical_sample(values[7], [0.25, 0.5, 0.75])
        if ordered_ctr:
            config["ctr_prior_strength"] = _log_sample(values[13], 0.1, 10.0)
            config["__categorical_pair_budget"] = _categorical_sample(values[16], [0, 0, 0, 2, 4])
        configs.append(_finalize_search_config(config))
    return configs


gen_ctboost = CustomAGConfigGenerator(
    model_cls=CTBoostModel,
    search_space_func=generate_configs_ctboost,
    manual_configs=[{}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_ctboost.generate_all_bag_experiments(num_random_configs=0),
        )
    )
