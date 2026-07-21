from __future__ import annotations

from tabarena.models.nori.model import Nori30MModel, NoriModel
from tabarena.utils.config_utils import ConfigGenerator

gen_nori = ConfigGenerator(
    model_cls=NoriModel,
    search_space={},
    manual_configs=[{}],
)

gen_nori30m = ConfigGenerator(
    model_cls=Nori30MModel,
    search_space={},
    manual_configs=[{}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_nori.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
