from __future__ import annotations

from tabarena.systems.tabfm_plus.system import TabFMPlusSystemModel
from tabarena.utils.config_utils import SystemConfigGenerator

# TabFM+ runs as a self-contained system (no AutoGluon bagging), so it uses a
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
            experiments=gen_tabfm_plus.generate_all_system_experiments(
                num_random_configs=0,
            ),
        ),
    )
