from __future__ import annotations

from tabarena.benchmark.models.ag.limix.limix_model import LimiXModel
from tabarena.utils.config_utils import ConfigGenerator


gen_limix = ConfigGenerator(
    model_cls=LimiXModel,
    search_space={},
    manual_configs=[{}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_limix.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
