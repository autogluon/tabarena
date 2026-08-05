from __future__ import annotations

from tabarena.models.tabfm.model import TabFMModel
from tabarena.utils.config_utils import ConfigGenerator

gen_tabfm = ConfigGenerator(
    model_cls=TabFMModel,
    manual_configs=[{}],
    search_space={},
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabfm.generate_all_bag_experiments(
                num_random_configs=0,
            ),
        ),
    )
