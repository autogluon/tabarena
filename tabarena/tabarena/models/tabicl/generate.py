from __future__ import annotations

from copy import deepcopy

from autogluon.common.space import Categorical, Real

from tabarena.benchmark.models.ag.tabicl.tabicl_model import (
    TabICLModel,
    TabICLModelBase,
    TabICLv2Model,
)
from tabarena.utils.config_utils import ConfigGenerator

# Unofficial search space
base_search_space = {
    "norm_methods": Categorical(
        "none", "power", "robust", "quantile_rtdl", ["none", "power"]
    ),
    # just in case, tuning between TabICL and TabPFN defaults
    "outlier_threshold": Real(4.0, 12.0),
    "average_logits": Categorical(False, True),
    # if average_logits=True this is equivalent to temperature scaling
    "softmax_temperature": Real(0.7, 1.0),
    # Hack to integrate refitting into the search space
    "ag_args_ensemble": Categorical({"refit_folds": True}),
}


def get_gen_function(model_cls: TabICLModelBase):
    search_space = deepcopy(base_search_space)
    search_space["checkpoint_version"] = Categorical(
        *model_cls.checkpoint_search_space()
    )
    return ConfigGenerator(
        model_cls=model_cls, manual_configs=[{}], search_space=search_space
    )


gen_tabicl = get_gen_function(TabICLModel)

gen_tabiclv2 = get_gen_function(TabICLv2Model)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabicl.generate_all_bag_experiments(num_random_configs=0),
        ),
    )

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabiclv2.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
