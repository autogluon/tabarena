from __future__ import annotations

from tabarena.models.tabfm.model import TabFMModel
from tabarena.models.tabfm.system import TabFMPlusSystemModel
from tabarena.utils.config_utils import ConfigGenerator, SystemConfigGenerator

gen_tabfm = ConfigGenerator(
    model_cls=TabFMModel,
    manual_configs=[{}],
    search_space={},
)

# TabFM+ is benchmarked as a self-contained system (no AutoGluon bagging), so it uses a
# SystemConfigGenerator; the single default config runs TabFM's ``ensemble`` interface (the
# system model's default).
gen_tabfm_plus = SystemConfigGenerator(
    model_cls=TabFMPlusSystemModel,
    name="TabFM+",
    manual_configs=[{}],
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

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabfm_plus.generate_all_system_experiments(
                num_random_configs=0,
            ),
        ),
    )
