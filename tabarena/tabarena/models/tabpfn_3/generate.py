from __future__ import annotations

from tabarena.benchmark.models.ag.tabpfnv3.tabpfn_3_model import TabPFN3Model
from tabarena.utils.config_utils import ConfigGenerator

gen_tabpfn_3 = ConfigGenerator(
    model_cls=TabPFN3Model,
    search_space={},
    manual_configs=[{}],
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabpfn_3.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
