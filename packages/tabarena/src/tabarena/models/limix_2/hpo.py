from __future__ import annotations

from tabarena.models.limix_2.model import LimiX2Model
from tabarena.utils.config_utils import ConfigGenerator

gen_limix_2 = ConfigGenerator(
    model_cls=LimiX2Model,
    search_space={},
    manual_configs=[{}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_limix_2.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
