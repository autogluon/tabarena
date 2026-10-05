from __future__ import annotations

from tabarena.systems.chakra_tab.system import ChakraTabSystemModel
from tabarena.utils.config_utils import SystemConfigGenerator

# One configuration per API preset; the API does its own validation, bagging and ensembling.
gen_chakra_tab = SystemConfigGenerator(
    model_cls=ChakraTabSystemModel,
    name="Chakra-Tab",
    manual_configs=[{"preset": "medium"}, {"preset": "full"}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_chakra_tab.generate_all_system_experiments(num_random_configs=0),
        ),
    )
