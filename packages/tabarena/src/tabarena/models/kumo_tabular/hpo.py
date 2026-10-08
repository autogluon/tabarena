from __future__ import annotations

from tabarena.models.kumo_tabular.model import KumoTabularMediumModel, KumoTabularModel, KumoTabularSmallModel
from tabarena.utils.config_utils import ConfigGenerator


def default_config() -> dict:
    """The default config: the context KV cache only for the refit model.

    The bagged fold models predict their out-of-fold rows once and are then dropped (``refit_folds``), so
    they run without the cache; the refit model, which serves every later prediction, builds it.
    """
    return {"cache_context": False, "ag.refit_hyperparameters": {"cache_context": True}}


gen_kumo_tabular = ConfigGenerator(
    model_cls=KumoTabularModel,
    search_space={},
    manual_configs=[default_config()],
)

gen_kumo_tabular_medium = ConfigGenerator(
    model_cls=KumoTabularMediumModel,
    search_space={},
    manual_configs=[default_config()],
)

gen_kumo_tabular_small = ConfigGenerator(
    model_cls=KumoTabularSmallModel,
    search_space={},
    manual_configs=[default_config()],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    for gen in (gen_kumo_tabular, gen_kumo_tabular_medium, gen_kumo_tabular_small):
        print(
            YamlExperimentSerializer.to_yaml_str(
                experiments=gen.generate_all_bag_experiments(num_random_configs=0),
            ),
        )
