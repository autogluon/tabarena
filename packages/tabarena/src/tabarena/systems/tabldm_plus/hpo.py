from __future__ import annotations

from tabarena.systems.tabldm_plus.system import TabLDMPlusSystemModel
from tabarena.utils.config_utils import SystemConfigGenerator

# Xiaomi-TabLDM+ runs as a self-contained system (no AutoGluon bagging), so it uses a
# SystemConfigGenerator. The single default config is the upstream enhanced inference path: the
# candidate/ensemble weights are fit on a hold-out split the estimator carves out of the training
# data itself. Both switches are spelled out rather than left to upstream's defaults, because this
# hold-out split is what makes the method a system rather than a model (see the class docstring).
gen_tabldm_plus = SystemConfigGenerator(
    model_cls=TabLDMPlusSystemModel,
    name="Xiaomi-TabLDM+",
    manual_configs=[{"enhance_candidates": True, "validation": True}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabldm_plus.generate_all_system_experiments(
                num_random_configs=0,
            ),
        ),
    )
