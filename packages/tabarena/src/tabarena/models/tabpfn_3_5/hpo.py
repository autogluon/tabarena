from __future__ import annotations

from tabarena.models.tabpfn_3_5.model import TabPFN35FastModel, TabPFN35Model
from tabarena.utils.config_utils import ConfigGenerator

gen_tabpfn_3_5 = ConfigGenerator(
    model_cls=TabPFN35Model,
    search_space={},
    manual_configs=[{}],
)

gen_tabpfn_3_5_fast = ConfigGenerator(
    model_cls=TabPFN35FastModel,
    search_space={},
    manual_configs=[{}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    for gen in (gen_tabpfn_3_5, gen_tabpfn_3_5_fast):
        print(
            YamlExperimentSerializer.to_yaml_str(
                experiments=gen.generate_all_bag_experiments(num_random_configs=0),
            ),
        )
