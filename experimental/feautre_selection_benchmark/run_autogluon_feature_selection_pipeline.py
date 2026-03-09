"""Example of a feature selector in tabular data."""

from __future__ import annotations

import pandas as pd
from autogluon.features.generators.selection import FeatureSelectionGenerator
from autogluon.tabular import TabularDataset, TabularPredictor


def run_example():
    train_data = TabularDataset(
        "https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification/train_data.csv"
    )
    test_data = TabularDataset(
        "https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification/test_data.csv"
    )
    predictor = TabularPredictor(label="class", default_base_path="/tmp/ag_out", eval_metric="roc_auc").fit(
        train_data=train_data,
        hyperparameters={"GBM": {"num_boost_round": 10}},
        num_bag_folds=2,
        num_bag_sets=1,
        verbosity=4,
        dynamic_stacking=False,
        fit_weighted_ensemble=False,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
        _feature_generator_kwargs={
            "post_generators": [FeatureSelectionGenerator()],
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
        print(X.head())
        print(y.head())


if __name__ == "__main__":
    run_example()
