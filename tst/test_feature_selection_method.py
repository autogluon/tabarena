import time

from autogluon.common import TabularDataset
from autogluon.core.data import LabelCleaner
from autogluon.features import AutoMLPipelineFeatureSelector, AbstractFeatureSelector


def run_method(FeatureSelector, hyperparameters):
    """
    Runs a feature selector through the standard AutoGluon pipeline.

    Args:
        FeatureSelector: class implementing fit_transform(X, y, model, n_max_features, **kwargs)
        hyperparameters: dict with 'n_max_features', 'time_limit', 'dataset' (optional)

    Returns:
        pd.DataFrame: X after feature selection (transformed data)
    """
    # Load dataset (same as your script)
    dataset = hyperparameters.get('dataset', 'AdultIncomeBinaryClassification')
    try:
        train_data = TabularDataset(f'https://autogluon.s3.amazonaws.com/datasets/{dataset}/train_data.csv')
        test_data = TabularDataset(f'https://autogluon.s3.amazonaws.com/datasets/{dataset}/test_data.csv')
    except:
        train_data = TabularDataset('train_data.csv')
        test_data = TabularDataset('test_data.csv')

    X_train = train_data.drop("class", axis=1)
    y_train = train_data["class"]

    # Prepare model (same as your script)
    from autogluon.core.models import BaggedEnsembleModel
    from tabarena.models.utils import get_configs_generator_from_name
    model_meta = get_configs_generator_from_name(model_name="CatBoost")
    model_config = model_meta.manual_configs[0]
    model = BaggedEnsembleModel(model_meta.model_cls(problem_type="binary", **model_config))

    # Clean labels (same as your script)
    label_cleaner = LabelCleaner.construct(problem_type="binary", y=y_train)
    y_train = label_cleaner.transform(y_train)

    # Run feature selection
    if callable(FeatureSelector) and not isinstance(FeatureSelector, AbstractFeatureSelector):
        SelectorInstance = FeatureSelector()
    else:
        SelectorInstance = FeatureSelector
    feature_selector = AutoMLPipelineFeatureSelector(post_selectors=[SelectorInstance])

    X_out = feature_selector.fit_transform(X_train, y_train, model, **hyperparameters)

    return X_out


def verify_method(FeatureSelector, hyperparameters):
    start_time = time.time()
    df = run_method(FeatureSelector, hyperparameters)
    end_time = time.time()
    time_used = end_time - start_time
    n_features = len(df.columns)

    verify_n_features(n_features, hyperparameters["n_max_features"])
    """
    # Uncomment this in order to use time constraint verification
    verify_time_limit(hyperparameters["time_limit"], time_used)
    """
    return df


def verify_n_features(n_features, n_max_features):
    assert n_features <= n_max_features, f"Expected at most {n_max_features} features, but got {n_features} features."
    print("Asserted number of features: ", n_features)

def verify_time_limit(time_limit, time_used):
    assert time_used <= time_limit, f"Expected the time used to be maximal {time_limit} seconds, but the method took {time_used} seconds."
    print("Asserted time limit: ", time_limit)
