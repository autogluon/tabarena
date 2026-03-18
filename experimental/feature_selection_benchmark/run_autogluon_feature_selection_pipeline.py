"""Example of a feature selector in tabular data."""

from __future__ import annotations

import openml
import pandas as pd
from autogluon.core.data import LabelCleaner
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import ProxyModelConfig
from tabarena.benchmark.feature_selection_methods.ag.accuracy.accuracy import AccuracyFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.anova.anova import ANOVAFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.cart.cart import CARTFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.cfs.cfs import CFSFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.chi2.chi2 import Chi2FeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.cmim.cmim import CMIMFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.consistency.consistency import ConsistencyFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.disr.disr import DISRFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.elastic_net.elastic_net import ElasticNetFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.gain_ratio.gain_ratio import GainRatioFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.gini.gini import GiniFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.impurity.impurity import ImpurityFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.information_gain.information_gain import (
    InformationGainFeatureSelector,
)
from tabarena.benchmark.feature_selection_methods.ag.interact.interact import INTERACTFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.jmi.jmi import JMIFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.laplacian_score.laplacian_score import (
    LaplacianScoreFeatureSelector,
)
from tabarena.benchmark.feature_selection_methods.ag.lasso.lasso import LassoFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.mi.mi import MIFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.mrmr.mrmr import mRMRFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.one_r.one_r import OneRFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.pearson_correlation.pearson_correlation import (
    PearsonCorrelationFeatureSelector,
)
from tabarena.benchmark.feature_selection_methods.ag.random.random import RandomFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.relief_f.relief_f import ReliefFFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.rf_importance.rf_importance import RFImportanceFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.sbe.sbe import SequentialBackwardEliminationFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.sfs.sfs import SequentialForwardSelectionFeatureSelector
from tabarena.benchmark.feature_selection_methods.ag.symmetrical_uncertainty.symmetrical_uncertainty import (
    SymmetricalUncertaintyFeatureSelector,
)
from tabarena.benchmark.feature_selection_methods.ag.t_test.t_test import tTestFeatureSelector


def run_example():
    dataset_id = 55  # 46964  # 10 or 55
    problem_type = "binary"  # "regression"  # "multiclass" or "binary"
    eval_metric = "roc_auc"  # "rmse"  # "log_loss" or "roc_auc"

    # Load OpenML dataset
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
    y = label_cleaner.transform(y)
    data = pd.concat([X, y.rename("class")], axis=1)
    train_data, test_data = train_test_split(data, test_size=0.33, random_state=0)

    max_features = 5
    proxy_model_config = ProxyModelConfig(
        problem_type=problem_type,
        eval_metric=eval_metric,
        model_hyperparameters={"num_boost_round": 1},
    )
    verbosity = 0

    for feature_selector in [
        AccuracyFeatureSelector(max_features=max_features, proxy_mode_config=proxy_model_config),
        RandomFeatureSelector(max_features=max_features),
        ANOVAFeatureSelector(max_features=max_features),
        CARTFeatureSelector(max_features=max_features),
        CFSFeatureSelector(max_features=max_features),
        Chi2FeatureSelector(max_features=max_features),
        CMIMFeatureSelector(max_features=max_features),
        ConsistencyFeatureSelector(max_features=max_features),
        DISRFeatureSelector(max_features=max_features),
        ElasticNetFeatureSelector(max_features=max_features),
        GainRatioFeatureSelector(max_features=max_features),
        GiniFeatureSelector(max_features=max_features),
        ImpurityFeatureSelector(max_features=max_features),
        InformationGainFeatureSelector(max_features=max_features),
        INTERACTFeatureSelector(max_features=max_features),
        JMIFeatureSelector(max_features=max_features),
        LaplacianScoreFeatureSelector(max_features=max_features),
        LassoFeatureSelector(max_features=max_features),
        MIFeatureSelector(max_features=max_features),
        mRMRFeatureSelector(max_features=max_features),
        OneRFeatureSelector(max_features=max_features),
        PearsonCorrelationFeatureSelector(max_features=max_features),
        ReliefFFeatureSelector(max_features=max_features),
        SequentialBackwardEliminationFeatureSelector(max_features=max_features, proxy_mode_config=proxy_model_config),
        SequentialForwardSelectionFeatureSelector(max_features=max_features, proxy_mode_config=proxy_model_config),
        SymmetricalUncertaintyFeatureSelector(max_features=max_features),
        RFImportanceFeatureSelector(max_features=max_features),
        tTestFeatureSelector(max_features=max_features),
    ]:
        print("\n####### Running feature selector:", feature_selector.name)
        predictor = TabularPredictor(
            label="class", default_base_path="/tmp/ag_out", eval_metric=eval_metric, problem_type=problem_type
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

        predictor.leaderboard(data=test_data, display=True)

        X, y = predictor.load_data_internal()
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.max_colwidth",
            None,
        ):
            print(X.head(n=1))

        print("\nOutcome of feature selection:")
        print(f"\t Selected features: {feature_selector._selected_features}")
        print(f"\t Feature scores: {feature_selector._feature_scores}")


if __name__ == "__main__":
    run_example()
