from __future__ import annotations

from tabarena.benchmark.models.ag.tabstar.tabstar_model import TabStarModel
from tabarena.utils.config_utils import ConfigGenerator
from autogluon.common.space import Categorical

gen_tabstar = ConfigGenerator(
    model_cls=TabStarModel,
    manual_configs=[{}],
    # TODO: more hyperparameter?
    search_space={
        "lora_lr": Categorical(0.0005, 0.001, 0.002, 0.005, 0.01),
        "lora_r": Categorical(8, 16, 32, 64),
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
