from __future__ import annotations

from copy import deepcopy

from tabarena.models.iltm.model import ILTMModel, _isolate_iltm_global_state
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator


def generate_configs_iltm(num_random_configs: int = 25) -> list[dict]:
    if num_random_configs < 0:
        raise ValueError("num_random_configs must be non-negative")
    if num_random_configs == 0:
        return []

    with _isolate_iltm_global_state():
        from iltm import get_hyperparameter_configs

        configs = get_hyperparameter_configs()

    if num_random_configs > len(configs):
        raise ValueError(f"At most {len(configs)} iLTM configurations are available")

    configs = deepcopy(configs[:num_random_configs])
    for config in configs:
        config.pop("device", None)
    return [convert_numpy_dtypes(config) for config in configs]


gen_iltm = CustomAGConfigGenerator(
    model_cls=ILTMModel,
    search_space_func=generate_configs_iltm,
    manual_configs=[{}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_iltm.generate_all_bag_experiments(num_random_configs=25),
        ),
    )
