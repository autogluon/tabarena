from __future__ import annotations

from tabarena.benchmark.models.ag.iltm.iltm_model import ILTMModel
from tabarena.utils.config_utils import ConfigGenerator

gen_iltm = ConfigGenerator(
    model_cls=ILTMModel,
    manual_configs=[{}],
    search_space={},  # No search space for now as for other foundation models.
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_iltm.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
