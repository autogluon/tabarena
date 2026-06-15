from __future__ import annotations

from autogluon.tabular import TabularPredictor

from tabarena.benchmark.task.openml import OpenMLTaskWrapper

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
    # presets="best",  # uncomment for a longer run
)

leaderboard = predictor.leaderboard(test_data, display=True)
