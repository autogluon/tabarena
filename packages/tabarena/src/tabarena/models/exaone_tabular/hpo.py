from __future__ import annotations

from tabarena.models.exaone_tabular.model import EXAONETabularModel
from tabarena.utils.config_utils import ConfigGenerator

gen_exaone_tabular = ConfigGenerator(
    model_cls=EXAONETabularModel,
    search_space={},
    manual_configs=[{}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_exaone_tabular.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
