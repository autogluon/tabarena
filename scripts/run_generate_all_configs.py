from autogluon.core.models import DummyModel

from tabarena.benchmark.experiment import AGModelBagExperiment, YamlExperimentSerializer
from tabarena.utils.config_utils import ConfigGenerator
from tabarena.models.lightgbm.generate import gen_lightgbm
from tabarena.models.tabicl.generate import gen_tabicl
from tabarena.models.tabdpt.generate import gen_tabdpt
from tabarena.models.knn.generate import gen_knn
from tabarena.models.fastai.generate import gen_fastai
from tabarena.models.extra_trees.generate import gen_extratrees
from tabarena.models.catboost.generate import gen_catboost
from tabarena.models.xgboost.generate import gen_xgboost
from tabarena.models.nn_torch.generate import gen_nn_torch
from tabarena.models.random_forest.generate import gen_randomforest
from tabarena.models.lr.generate import gen_linear
from tabarena.models.realmlp.generate import gen_realmlp
from tabarena.models.ebm.generate import gen_ebm
from tabarena.models.tabm.generate import gen_tabm
from tabarena.models.modernnca.generate import gen_modernnca


if __name__ == '__main__':
    n_random_configs = 200
    n_random_configs_baselines = 200

    # Original Search Space
    experiments_linear = gen_linear.generate_all_bag_experiments(num_random_configs=n_random_configs_baselines)
    experiments_knn = gen_knn.generate_all_bag_experiments(num_random_configs=n_random_configs_baselines)
    experiments_nn_torch = gen_nn_torch.generate_all_bag_experiments(num_random_configs=n_random_configs)

    # New Search Space
    experiments_randomforest = gen_randomforest.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_extratrees = gen_extratrees.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_lightgbm = gen_lightgbm.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_xgboost = gen_xgboost.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_catboost = gen_catboost.generate_all_bag_experiments(num_random_configs=n_random_configs)

    # New methods
    experiments_fastai = gen_fastai.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_ebm = gen_ebm.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_realmlp = gen_realmlp.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_tabm = gen_tabm.generate_all_bag_experiments(num_random_configs=n_random_configs)

    experiments_tabicl = gen_tabicl.generate_all_bag_experiments(num_random_configs=0)
    experiments_tabdpt = gen_tabdpt.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_modernnca = gen_modernnca.generate_all_bag_experiments(num_random_configs=n_random_configs)

    # Dummy (constant predictor)
    experiments_dummy = ConfigGenerator(model_cls=DummyModel, search_space={}, manual_configs=[{}]).generate_all_bag_experiments(num_random_configs=0)

    experiments_lst = [
        experiments_linear,
        experiments_knn,
        experiments_randomforest,
        experiments_extratrees,
        experiments_lightgbm,
        experiments_xgboost,
        experiments_catboost,
        experiments_nn_torch,

        experiments_fastai,
        experiments_ebm,
        experiments_realmlp,
        experiments_tabicl,
        # experiments_tabpfnv2,
        experiments_tabdpt,
        experiments_tabm,
        experiments_modernnca,

        experiments_dummy,
    ]

    experiments_all: list[AGModelBagExperiment] = [exp for exp_family_lst in experiments_lst for exp in exp_family_lst]

    # Verify no duplicate names
    experiment_names = set()
    for experiment in experiments_all:
        if experiment.name not in experiment_names:
            experiment_names.add(experiment.name)
        else:
            raise AssertionError(f"Found multiple instances of experiment named {experiment.name}. All experiment names must be unique!")

    YamlExperimentSerializer.to_yaml(experiments=experiments_all, path="configs_all.yaml")

    from tabarena.models.automl import generate_autogluon_experiments
    experiments_autogluon = generate_autogluon_experiments()

    YamlExperimentSerializer.to_yaml(experiments=experiments_autogluon, path="configs_autogluon.yaml")
