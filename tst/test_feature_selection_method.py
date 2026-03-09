from __future__ import annotations

import time

import openml
from autogluon.core.data import LabelCleaner
from autogluon.core.models import BaggedEnsembleModel
from autogluon.features import AbstractFeatureSelector, AutoMLPipelineFeatureSelector
from tabarena.models.utils import get_configs_generator_from_name


def run_method(FeatureSelector, hyperparameters):
    """Runs a feature selector through the standard AutoGluon pipeline.

    Args:
        FeatureSelector: class implementing fit_transform(X, y, model, n_max_features, **kwargs)
        hyperparameters: dict with 'n_max_features', 'time_limit', 'dataset' (optional)

    Returns:
        pd.DataFrame: X after feature selection (transformed data)
    """
    dataset_id = hyperparameters.get("dataset_id")
    problem_type = hyperparameters.get("problem_type")

    # Load OpenML dataset
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")

    model_meta = get_configs_generator_from_name(model_name="LightGBM")
    model_config = model_meta.manual_configs[0]
    model = BaggedEnsembleModel(model_meta.model_cls(problem_type=problem_type **model_config))

    # FIXME: determine if this is needed
    # Clean labels (same as your script)
    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
    y = label_cleaner.transform(y)

    # Run feature selection
    if callable(FeatureSelector) and not isinstance(FeatureSelector, AbstractFeatureSelector):
        SelectorInstance = FeatureSelector()
    else:
        SelectorInstance = FeatureSelector
    feature_selector = AutoMLPipelineFeatureSelector(post_selectors=[SelectorInstance])

    return feature_selector.fit_transform(X, y, model, **hyperparameters)



def verify_method(FeatureSelector, hyperparameters, check_time_constraint: bool = False):
    start_time = time.time()
    df = run_method(FeatureSelector, hyperparameters)
    time_used = time.time() - start_time
    n_features = len(df.columns)
    n_max_features = hyperparameters["n_max_features"]

    assert n_features <= n_max_features, f"Expected at most {n_max_features} features, but got {n_features} features."
    print("Asserted number of features: ", n_features)

    if check_time_constraint:
        time_limit = hyperparameters["time_limit"]
        assert time_used <= time_limit, (
            f"Expected the time used to be maximal {time_limit} seconds, but the method took {time_used} seconds."
        )
        print("Asserted time limit: ", time_limit)

    return df
