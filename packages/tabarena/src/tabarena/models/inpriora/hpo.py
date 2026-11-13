from __future__ import annotations

from tabarena.models.inpriora.model import InprioraModel
from tabarena.utils.config_utils import ConfigGenerator


# One released checkpoint per task and no training hyperparameters to tune: the default configuration only.
def default_config() -> dict:
    """Cache the encoded context only in the refit model; each fold model predicts its single
    out-of-fold split once, so its cache would be built and dropped without being reused.
    """
    return {"cache": False, "ag.refit_hyperparameters": {"cache": True}}


gen_inpriora = ConfigGenerator(
    model_cls=InprioraModel,
    search_space={},
    manual_configs=[default_config()],
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_inpriora.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
