from __future__ import annotations

from tabarena.models.tabh2o.model import TabH2OModel
from tabarena.utils.config_utils import ConfigGenerator

gen_tabh2o = ConfigGenerator(
    model_cls=TabH2OModel,
    search_space={},
    manual_configs=[{}],
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabh2o.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
