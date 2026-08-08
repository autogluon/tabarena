from __future__ import annotations

from tabarena.systems.autogluon.system import AutoGluonSystemModel
from tabarena.utils.config_utils import SystemConfigGenerator

# The two presets TabArena reports. `best_quality` is the CPU workhorse; `extreme_quality`
# is the GPU preset that folds in foundation models. Each config becomes one benchmarked
# AutoGluon variant; the time budget comes from the run's resources, not from here, so the
# same generator serves the 5m / 1h / 4h suites.
gen_autogluon = SystemConfigGenerator(
    model_cls=AutoGluonSystemModel,
    name="AutoGluon",
    manual_configs=[
        {"preset": "best_quality"},
        {"preset": "extreme_quality"},
    ],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_autogluon.generate_all_system_experiments(
                num_random_configs=0,
            ),
        ),
    )
