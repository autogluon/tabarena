from __future__ import annotations

from tabarena.benchmark.models.ag.tabpfnwide.tabpfnwide_model import TabPFNWideModel
from tabarena.utils.config_utils import ConfigGenerator

gen_tabpfnwide = ConfigGenerator(
    model_cls=TabPFNWideModel,
    search_space={},
    manual_configs=[{}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabpfnwide.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
