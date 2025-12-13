from __future__ import annotations

from tabarena.benchmark.models.ag.tabstar.tabstar_model import TabStarModel
from tabarena.utils.config_utils import ConfigGenerator

# TODO: add search space
gen_tabstar = ConfigGenerator(
    model_cls=TabStarModel,
    manual_configs=[{}],
    search_space={},
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabstar.generate_all_bag_experiments(
                num_random_configs=0
            ),
        ),
    )
