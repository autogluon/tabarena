from __future__ import annotations

from tabarena.models.tabstar_v1p1.model import TabSTARv1p1Model
from tabarena.utils.config_utils import ConfigGenerator

# The default configuration is the method: v1.1 is submitted as a single tuning-free entrant.
gen_tabstar_v1p1 = ConfigGenerator(
    model_cls=TabSTARv1p1Model,
    manual_configs=[{}],
    search_space={},
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabstar_v1p1.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
