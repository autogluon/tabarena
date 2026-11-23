from __future__ import annotations

from tabarena.systems.val_select.system import ValSelectSystemModel
from tabarena.utils.config_utils import SystemConfigGenerator

# ValSelect runs as a self-contained system (no AutoGluon bagging): one default config, the
# LimiX-2-1016 + Kumo-Tabular holdout selection with its defaults.
gen_val_select = SystemConfigGenerator(
    model_cls=ValSelectSystemModel,
    name="ValSelect",
    manual_configs=[{}],
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_val_select.generate_all_system_experiments(
                num_random_configs=0,
            ),
        ),
    )
