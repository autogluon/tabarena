from __future__ import annotations

from tabarena.benchmark.models.ag.tabstar.tabstar_model import TabStarModel
from tabarena.utils.config_utils import ConfigGenerator
from autogluon.common.space import Categorical

gen_tabstar = ConfigGenerator(
    model_cls=TabStarModel,
    manual_configs=[{}],
    search_space={
        "lora_lr": Categorical(0.0001, 0.0002, 0.0005, 0.001, 0.002),
        "lora_wd": Categorical(0, 0.00001, 0.0001, 0.001),
        "lora_r": Categorical(16, 32, 64),
        "lora_alpha": Categorical(1, 2),
        "lora_dropout": Categorical(0, 0.05, 0.1),
        "global_batch": Categorical(64, 128, 256),
    },
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabstar.generate_all_bag_experiments(
                num_random_configs=0
            ),
        ),
    )
