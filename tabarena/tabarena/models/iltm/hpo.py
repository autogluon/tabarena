from __future__ import annotations

from ConfigSpace import Categorical, ConfigurationSpace, Float

from tabarena.models.iltm.model import ILTMModel
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator


def generate_configs_iltm(num_random_configs: int = 200) -> list[dict]:
    # Mirrors iLTM's recommended search space (iltm/hyperparameter_search_space.py),
    # including its `probs=...` weights. Constants matching iLTM defaults are
    # omitted; `device` is omitted because the wrapper passes it explicitly to
    # the iLTM predictor and a duplicate kwarg would raise TypeError.
    search_space = ConfigurationSpace(
        space=[
            Categorical(
                "checkpoint",
                ["xgbrconcat", "cbrconcat", "r128bn", "rnobn", "xgb", "catb", "rtr", "rtrcb"],
            ),
            Categorical("n_ensemble", [4, 8, 12, 16, 32, 64]),
            Categorical("batch_size", [2048, 4096]),
            Categorical("finetuning_dropout", [0.0, 0.15]),
            Categorical("finetuning_max_steps", [2048, 4096]),
            Categorical("finetuning_batch_size", [64, 128, 256, 512, 1024, 2048, 4096]),
            Float("finetuning_lr", (1e-4, 3e-3), log=True),
            Float("gradient_clip_norm", (0.5, 1.5)),
            Categorical("finetuning_optimizer", ["adamw", "lion"]),
            Categorical("tree_data_split", ["dynamic", "all"]),
            Categorical("tree_n_estimators", [100, 125, 150, 200, 300]),
            Float("tree_lr", (1e-3, 1.0), log=True),
            Categorical(
                "tree_max_depth",
                [4, 5, 6],
                weights=[0.20, 0.65, 0.15],
            ),
            Categorical("tree_min_samples_leaf", [1, 2, 4, 8, 12, 16]),
            Float("tree_subsample", (0.5, 1.0)),
            Float("tree_feature_fraction", (0.6, 1.0)),
            Categorical(
                "tree_gamma",
                [0.0, 0.05, 0.1, 0.25, 0.5],
                weights=[0.6, 0.1, 0.1, 0.1, 0.1],
            ),
            Categorical(
                "tree_l2_leaf_reg",
                [0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0],
            ),
            Float("tree_bagging_temperature", (0.1, 1.0)),
            Categorical("do_retrieval", [True, False], weights=[0.65, 0.35]),
            Float("retrieval_alpha", (0.0, 1.0)),
            Float("retrieval_temperature", (1.0, 2.5)),
            Categorical("retrieval_distance", ["cosine", "euclidean"]),
            Float("scheduler_min_lr", (1e-7, 3e-4), log=True),
            Categorical("clip_predictions", [False, True]),
            Categorical(
                "corr_select_k",
                [0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 512, 1024, 2048, 4096],
                weights=[
                    0.20,
                    0.02,
                    0.02,
                    0.02,
                    0.03,
                    0.03,
                    0.05,
                    0.10,
                    0.15,
                    0.15,
                    0.08,
                    0.08,
                    0.03,
                    0.02,
                    0.02,
                ],
            ),
        ],
        seed=1234,
    )

    configs = search_space.sample_configuration(num_random_configs)
    if num_random_configs == 1:
        configs = [configs]
    configs = [dict(config) for config in configs]
    return [convert_numpy_dtypes(config) for config in configs]


gen_iltm = CustomAGConfigGenerator(
    model_cls=ILTMModel,
    search_space_func=generate_configs_iltm,
    manual_configs=[{}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_iltm.generate_all_bag_experiments(num_random_configs=25),
        ),
    )
