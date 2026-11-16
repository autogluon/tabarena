from __future__ import annotations

from tabarena.systems.tabpfn_3_5_thinking.system import TabPFN35ThinkingSystemModel
from tabarena.utils.config_utils import SystemConfigGenerator

# One configuration, the thinking effort xhigh.
gen_tabpfn_3_5_thinking = SystemConfigGenerator(
    model_cls=TabPFN35ThinkingSystemModel,
    name="TabPFN-3.5-Thinking",
    manual_configs=[{"effort": "xhigh"}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabpfn_3_5_thinking.generate_all_system_experiments(num_random_configs=0),
        ),
    )
