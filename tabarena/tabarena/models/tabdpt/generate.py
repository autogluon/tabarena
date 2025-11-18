from __future__ import annotations

from autogluon.common.space import Categorical

from tabarena.benchmark.models.ag.tabdpt.tabdpt_model import TabDPTModel
from tabarena.utils.config_utils import ConfigGenerator

name = "TabDPT"
search_space = {
    "temperature": Categorical(
        0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0, 1.25, 1.5
    ),
    "context_size": Categorical(2048, 768, 256),
    "permute_classes": Categorical(True, False),
    "normalizer": Categorical(
        "standard",
        None,
        "minmax",
        "robust",
        "power",
        "quantile-uniform",
        "quantile-normal",
        "log1p",
    ),
    "missing_indicators": Categorical(False, True),
    "clip_sigma": Categorical(4, 2, 6, 8),
    "feature_reduction": Categorical("pca", "subsample"),
    "faiss_metric": Categorical("l2", "ip"),
}

gen_tabdpt = ConfigGenerator(
    model_cls=TabDPTModel, manual_configs=[{}], search_space=search_space
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabdpt.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
