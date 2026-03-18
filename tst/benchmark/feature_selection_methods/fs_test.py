from __future__ import annotations

import logging
from typing import List, Tuple

import openml
import pandas as pd
import pytest
from autogluon.core.data import LabelCleaner
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import ProxyModelConfig

logger = logging.getLogger(__name__)

# Parametrization: Combine three small datasets of different problem types (including missing values) with all implemented FS methods
DATASET_CONFIGS: List[Tuple[int, str, str]] = [
    (55, "binary", "roc_auc"),
    (10, "multiclass", "log_loss"),
    (46964, "regression", "rmse"),
]

METHOD_NAMES = [
    "AccuracyFeatureSelector",
    "RandomFeatureSelector",
    "ANOVAFeatureSelector",
    "CARTFeatureSelector",
    "CFSFeatureSelector",
    "Chi2FeatureSelector",
    "CMIMFeatureSelector",
    "ConsistencyFeatureSelector",
    "DISRFeatureSelector",
    "ElasticNetFeatureSelector",
    "GainRatioFeatureSelector",
    "GiniFeatureSelector",
    "ImpurityFeatureSelector",
    "InformationGainFeatureSelector",
    "INTERACTFeatureSelector",
    "JMIFeatureSelector",
    "LaplacianScoreFeatureSelector",
    "LassoFeatureSelector",
    "MIFeatureSelector",
    "mRMRFeatureSelector",
    "OneRFeatureSelector",
    "PearsonCorrelationFeatureSelector",
    "ReliefFFeatureSelector",
    "RFImportanceFeatureSelector",
    "SequentialBackwardEliminationFeatureSelector",
    "SequentialForwardSelectionFeatureSelector",
    "SymmetricalUncertaintyFeatureSelector",
    "tTestFeatureSelector"
]

test_params = []
for dataset_id, problem_type, evaluation_metric in DATASET_CONFIGS:
    for method_name in METHOD_NAMES:
        test_params.append((dataset_id, problem_type, evaluation_metric, method_name))


def _instantiate_selector(method_name: str, max_features: int, proxy_config) -> any:
    import_map = {
        "AccuracyFeatureSelector": "experimental.feature_selection_benchmark.accuracy.accuracy",
        "RandomFeatureSelector": "experimental.feature_selection_benchmark.random.random",
        "ANOVAFeatureSelector": "experimental.feature_selection_benchmark.anova.anova",
        "CARTFeatureSelector": "experimental.feature_selection_benchmark.cart.cart",
        "CFSFeatureSelector": "experimental.feature_selection_benchmark.cfs.cfs",
        "Chi2FeatureSelector": "experimental.feature_selection_benchmark.chi2.chi2",
        "CMIMFeatureSelector": "experimental.feature_selection_benchmark.cmim.cmim",
        "ConsistencyFeatureSelector": "experimental.feature_selection_benchmark.consistency.consistency",
        "DISRFeatureSelector": "experimental.feature_selection_benchmark.disr.disr",
        "ElasticNetFeatureSelector": "experimental.feature_selection_benchmark.elastic_net.elastic_net",
        "GainRatioFeatureSelector": "experimental.feature_selection_benchmark.gain_ratio.gain_ratio",
        "GiniFeatureSelector": "experimental.feature_selection_benchmark.gini.gini",
        "ImpurityFeatureSelector": "experimental.feature_selection_benchmark.impurity.impurity",
        "InformationGainFeatureSelector": "experimental.feature_selection_benchmark.information_gain.information_gain",
        "INTERACTFeatureSelector": "experimental.feature_selection_benchmark.interact.interact",
        "JMIFeatureSelector": "experimental.feature_selection_benchmark.jmi.jmi",
        "LaplacianScoreFeatureSelector": "experimental.feature_selection_benchmark.laplacian_score.laplacian_score",
        "LassoFeatureSelector": "experimental.feature_selection_benchmark.lasso.lasso",
        "MIFeatureSelector": "experimental.feature_selection_benchmark.mi.mi",
        "mRMRFeatureSelector": "experimental.feature_selection_benchmark.mrmr.mrmr",
        "OneRFeatureSelector": "experimental.feature_selection_benchmark.one_r.one_r",
        "PearsonCorrelationFeatureSelector": "experimental.feature_selection_benchmark.pearson_correlation.pearson_correlation",
        "ReliefFFeatureSelector": "experimental.feature_selection_benchmark.relief_f.relief_f",
        "RFImportanceFeatureSelector": "experimental.feature_selection_benchmark.rf_importance.rf_importance",
        "SequentialBackwardEliminationFeatureSelector": "experimental.feature_selection_benchmark.sbe.sbe",
        "SequentialForwardSelectionFeatureSelector": "experimental.feature_selection_benchmark.sfs.sfs",
        "SymmetricalUncertaintyFeatureSelector": "experimental.feature_selection_benchmark.symmetrical_uncertainty.symmetrical_uncertainty",
        "tTestFeatureSelector": "experimental.feature_selection_benchmark.t_test.t_test",
    }
    module_path = import_map[method_name]
    module = __import__(module_path.replace('/', '.'), fromlist=[method_name])
    SelectorClass = getattr(module, method_name)

    kwargs = {"max_features": max_features}
    if method_name in ["AccuracyFeatureSelector", "SequentialBackwardEliminationFeatureSelector", "SequentialForwardSelectionFeatureSelector"]:
        kwargs["proxy_mode_config"] = proxy_config

    return SelectorClass(**kwargs)


@pytest.mark.parametrize(
    "dataset_id, problem_type, evaluation_metric, method_name",
    test_params,
    ids=[f"{method_name}_{dataset_id}" for dataset_id, problem_type, evaluation_metric, method_name in test_params]
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
    feature_selector = _instantiate_selector(method_name, max_features, proxy_config)

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

    if hasattr(feature_selector, 'feature_scoring_method') and feature_selector.feature_scoring_method:
        assert feature_selector._feature_scores is not None
        for f in feature_selector._selected_features:
            assert f in feature_selector._feature_scores
