from __future__ import annotations

from tabarena.models.tabldm.model import TabLDMModel
from tabarena.utils.config_utils import ConfigGenerator

gen_tabldm = ConfigGenerator(
    model_cls=TabLDMModel,
    search_space={},
    manual_configs=[{}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabldm.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
