from __future__ import annotations

import logging

import openml
import pandas as pd
import pytest
from autogluon.core.data import LabelCleaner
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import ProxyModelConfig
from tabarena.benchmark.feature_selection_methods.feature_selection_methods_register import (
    FEATURE_SELECTION_METHODS,
    get_feature_selector_from_name,
)

logger = logging.getLogger(__name__)

# Parametrization: Combine three small datasets of different problem types (including missing values) with all implemented FS methods
DATASET_CONFIGS: list[tuple[int, str, str]] = [
    (55, "binary", "roc_auc"),
    (10, "multiclass", "log_loss"),
    (46964, "regression", "rmse"),
]
test_params = []
for dataset_id, problem_type, evaluation_metric in DATASET_CONFIGS:
    for method_name in FEATURE_SELECTION_METHODS:
        test_params.append((dataset_id, problem_type, evaluation_metric, method_name))

# TODO:
#   - Add testing for edge cases in the data state and feature count
#   - Test time limit here somehow
@pytest.mark.parametrize(
    ("dataset_id", "problem_type", "evaluation_metric", "method_name"),
    test_params,
    ids=[f"{method_name}_{dataset_id}" for dataset_id, problem_type, evaluation_metric, method_name in test_params],
)
def test_feature_selector_dataset_combo(dataset_id, problem_type, evaluation_metric, method_name, verbosity=0):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
    y = label_cleaner.transform(y)
    data = pd.concat([X, y.rename("class")], axis=1)
    train_data, test_data = train_test_split(data, test_size=0.33, random_state=0)

    proxy_config = ProxyModelConfig(
        problem_type=problem_type,
        eval_metric=evaluation_metric,
        model_hyperparameters={"num_boost_round": 1},
    )

    max_features = 5
    feature_selector = get_feature_selector_from_name(
        name=method_name,
    )
    feature_selector = feature_selector(max_features=max_features, proxy_config=proxy_config)

    print(f"\n####### Testing {method_name} on dataset {dataset_id} ({problem_type})")

    predictor = TabularPredictor(
        label="class",
        default_base_path=f"/tmp/ag_out_ds{dataset_id}_{method_name}",
        eval_metric=evaluation_metric,
        problem_type=problem_type,
    ).fit(
        train_data=train_data,
        hyperparameters={"GBM": {"num_boost_round": 10}},
        num_bag_folds=2,
        num_bag_sets=1,
        verbosity=verbosity,
        dynamic_stacking=False,
        fit_weighted_ensemble=False,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
        _feature_generator_kwargs={
            "post_generators": [feature_selector],
        },
    )

    leaderboard = predictor.leaderboard(data=test_data, silent=True)
    assert not leaderboard.empty, f"Leaderboard empty for {method_name} on dataset {dataset_id}"

    assert feature_selector._selected_features is not None
    assert len(feature_selector._selected_features) <= max_features
    assert all(f in train_data.columns for f in feature_selector._selected_features)
