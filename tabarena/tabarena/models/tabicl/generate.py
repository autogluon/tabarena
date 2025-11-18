from __future__ import annotations

from autogluon.common.space import Categorical, Real

from tabarena.benchmark.models.ag.tabicl.tabicl_model import TabICLModel
from tabarena.utils.config_utils import ConfigGenerator

name = "TabICL"
manual_configs = [
    # Default config with refit after cross-validation.
    {"ag_args_ensemble": {"refit_folds": True}},
]

# Unofficial search space
search_space = {
    "checkpoint_version": Categorical("tabicl-classifier-v1.1-0506.ckpt", "tabicl-classifier-v1-0208.ckpt"),
    "norm_methods": Categorical("none", "power", "robust", "quantile_rtdl", ["none", "power"]),
    # just in case, tuning between TabICL and TabPFN defaults
    "outlier_threshold": Real(4.0, 12.0),
    "average_logits": Categorical(False, True),
    # if average_logits=True this is equivalent to temperature scaling
    "softmax_temperature": Real(0.7, 1.0),
    # Hack to integrate refitting into the search space
    "ag_args_ensemble": Categorical({"refit_folds": True}),
}

gen_tabicl = ConfigGenerator(
    model_cls=TabICLModel, manual_configs=manual_configs, search_space=search_space
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabicl.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
