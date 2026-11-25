from __future__ import annotations

from tabarena.models.tabsil.model import TabSilModel
from tabarena.utils.config_utils import ConfigGenerator

#: The default configuration only, like the other foundation models without a search space.
gen_tabsil = ConfigGenerator(
    model_cls=TabSilModel,
    search_space={},
    manual_configs=[{}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabsil.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
