"""Run a standalone TabArena Feature Selection Method using some model on any task."""

from __future__ import annotations

from urllib.error import URLError

from autogluon.common import TabularDataset
from autogluon.core.data import LabelCleaner
from autogluon.core.models import BaggedEnsembleModel
from autogluon.features.generators import AutoMLPipelineFeatureSelector
from autogluon.features.generators.selection import FeatureSelector

from tabarena.models.utils import get_configs_generator_from_name
import warnings
warnings.filterwarnings("ignore")

task_type = "binary"
cross_validation_bagging = True
refit_model = False
model_to_run = "CatBoost"
model_meta = get_configs_generator_from_name(model_name=model_to_run)
model_cls = model_meta.model_cls
model_config = model_meta.manual_configs[0]

try:
    train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification/train_data.csv')
    test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification/test_data.csv')
    # Save to local disk
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)
except URLError:
    train_data = TabularDataset('train_data.csv')
    test_data = TabularDataset('test_data.csv')

X_train = train_data.drop("class", axis=1)
y_train = train_data["class"]
X_test = test_data.drop("class", axis=1)
y_test = test_data["class"]

method = "MI"
n_max_features = 10
model = BaggedEnsembleModel(
        model_cls(problem_type=task_type, **model_config),
        hyperparameters=dict(refit_folds=refit_model),
    )

# --- Using a TabArena Model: Preprocessing, Train, and Predict:
print(f"Running TabArena Feature Selection Method: {method}")
print("Start: ", str(len(X_test.columns)) + " features")
label_cleaner = LabelCleaner.construct(problem_type=task_type, y=y_train)
y_train = label_cleaner.transform(y_train)
y_test = label_cleaner.transform(y_test)
feature_selector = AutoMLPipelineFeatureSelector(post_selectors=[FeatureSelector(method)])
X_train = feature_selector.fit_transform(X_train, y_train, model, n_max_features)
X_test = feature_selector.transform(X_test)

print("Result: ", str(len(X_test.columns)) + " features")

if cross_validation_bagging:
    model.params["fold_fitting_strategy"] = "sequential_local"
    model = model.fit(X=X_train, y=y_train, k_fold=8)
    print(f"Validation {model.eval_metric.name}:", model.score_with_oof(y=y_train))
else:
    model = model_cls(problem_type=task_type, **model_config)
    model = model.fit(X=X_train, y=y_train)
y_pred = model.predict(X=X_test)
