from __future__ import annotations

import numpy as np

from tabarena.models.denselight.model import DenseLightModel
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator

# Search space mirrors LightAutoML's own denselight tuning (lr / weight_decay / batch size; see
# TorchModel._get_default_search_spaces) plus the obvious architecture knobs (dropout, layer sizes).
_HIDDEN_SIZE_CHOICES = [
    [512, 256],
    [512, 512, 512],
    [256, 128],
    [1024, 512, 256],
]


def generate_single_config_denselight(rng):
    weight_decay_bin = rng.integers(0, 2)
    weight_decay = 0.0 if weight_decay_bin == 0 else np.exp(rng.uniform(np.log(1e-6), np.log(1e-2)))
    params = {
        "lr": np.exp(rng.uniform(np.log(1e-5), np.log(1e-1))),
        "weight_decay": weight_decay,
        "bs": rng.choice([64, 128, 256, 512, 1024]),
        "drop_rate": rng.uniform(0.0, 0.5),
        "hidden_size": _HIDDEN_SIZE_CHOICES[rng.integers(0, len(_HIDDEN_SIZE_CHOICES))],
    }
    return convert_numpy_dtypes(params)


def generate_configs_denselight(num_random_configs=200, seed=1234):
    rng = np.random.default_rng(seed)
    return [generate_single_config_denselight(rng) for _ in range(num_random_configs)]


gen_denselight = CustomAGConfigGenerator(
    model_cls=DenseLightModel,
    search_space_func=generate_configs_denselight,
    manual_configs=[{}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_denselight.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
