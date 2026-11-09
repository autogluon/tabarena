from __future__ import annotations

from tabarena.models.tabpack.model import TabPackModel
from tabarena.utils.config_utils import ConfigGenerator

# TabPack is a hyperparameter ensemble that the paper evaluates without tuning: one fit already
# samples 64 members from the library's default distributions, so the default config is the model.
gen_tabpack = ConfigGenerator(
    model_cls=TabPackModel,
    search_space={},
    manual_configs=[{}],
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabpack.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
