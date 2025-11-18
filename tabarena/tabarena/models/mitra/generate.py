from __future__ import annotations

from autogluon.tabular.models import MitraModel
from tabarena.utils.config_utils import ConfigGenerator

name = "Mitra"
manual_configs = [
    {},
]

gen_mitra = ConfigGenerator(
    model_cls=MitraModel, manual_configs=manual_configs, search_space={}
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_mitra.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
