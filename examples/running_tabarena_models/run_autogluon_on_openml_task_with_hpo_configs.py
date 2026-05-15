from tabarena.benchmark.task.openml import OpenMLTaskWrapper
from tabarena.export.autogluon import AutoGluonExporter
from tabarena.models.lightgbm.generate import gen_lightgbm
from tabarena.models.catboost.generate import gen_catboost

from autogluon.tabular import TabularPredictor

# run the default config + 5 random configurations of lightgbm and the default catboost config
experiments_lightgbm = gen_lightgbm.generate_all_bag_experiments(num_random_configs=5)
experiments_catboost = gen_catboost.generate_all_bag_experiments(num_random_configs=0)

experiments = [
    *experiments_lightgbm,
    *experiments_catboost,
]
exporter = AutoGluonExporter(experiments)
ag_hyperparameters = exporter.export_hyperparameters()

# supports any task on OpenML
task_id = 363614  # anneal
task = OpenMLTaskWrapper.from_task_id(task_id=task_id)

train_data, test_data = task.get_train_test_split_combined(fold=0)

predictor = TabularPredictor(
    label=task.label,
    problem_type=task.problem_type,
    eval_metric=task.eval_metric,
)

predictor = predictor.fit(
    train_data=train_data,
    hyperparameters=ag_hyperparameters,
    num_bag_folds=8,
)

leaderboard = predictor.leaderboard(test_data, display=True)
