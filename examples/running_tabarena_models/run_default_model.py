"""Run a standalone TabArena model on any task."""

from __future__ import annotations

from autogluon.core.data import LabelCleaner
from autogluon.core.models import BaggedEnsembleModel
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from data_utils import (
    get_example_data_for_task_type,
    score_for_task_type,
)

from tabarena.models.utils import get_configs_generator_from_name

task_type = "binary"
"""Task type for the model to run on.
Either "binary", "multiclass", or "regression".
"""
cross_validation_bagging = True
"""If True, will use cross-validation bagging for the model.
This is the default on TabArena and recommended for most tasks.
"""
refit_model = False
"""If True, will refit the model on the full training data after
cross-validating. Recommended for tabular foundation models, such
 as TabPFNv2 and TabICL, otherwise not recommended!"""
model_to_run = "RealMLP"
"""Select a model to run, which we automatically load in the code below.

Note: not all models are available for all task types.

The recommended options are:
    - "RealMLP"
    - "TabM"
    - "LightGBM"
    - "CatBoost"
    - "XGBoost"
    - "ModernNCA"
    - "TabPFNv2"
    - "TabICL"
    - "TorchMLP"
    - "TabDPT"
    - "EBM"
    - "FastaiMLP"
    - "ExtraTrees
    - "RandomForest"
    - "KNN"
    - "Linear"

You can also import it manually from TabArena / AutoGluon, which we recommend
for practical applications, for example:
 - RealMLP: from tabarena.models.realmlp.model import RealMLPModel
 - Catboost: from autogluon.tabular.models.catboost.catboost_model import CatBoostModel
"""
model_meta = get_configs_generator_from_name(model_name=model_to_run)
model_cls = model_meta.model_cls
model_config = model_meta.manual_configs[0]

X_train, X_test, y_train, y_test = get_example_data_for_task_type(task_type=task_type)

# --- Using a TabArena Model: Preprocessing, Train, and Predict:
print(f"Running TabArena model {model_to_run} on task type {task_type}...")
feature_generator, label_cleaner = (
    AutoMLPipelineFeatureGenerator(),
    LabelCleaner.construct(problem_type=task_type, y=y_train),
)
X_train, y_train = (
    feature_generator.fit_transform(X_train),
    label_cleaner.transform(y_train),
)
X_test, y_test = feature_generator.transform(X_test), label_cleaner.transform(y_test)

if cross_validation_bagging:
    model = BaggedEnsembleModel(
        model_cls(problem_type=task_type, **model_config),
        hyperparameters=dict(refit_folds=refit_model),
    )
    model.params["fold_fitting_strategy"] = "sequential_local"
    model = model.fit(X=X_train, y=y_train, k_fold=8)
    print(f"Validation {model.eval_metric.name}:", model.score_with_oof(y=y_train))
else:
    model = model_cls(problem_type=task_type, **model_config)
    model = model.fit(X=X_train, y=y_train)
y_pred = model.predict(X=X_test)

score_for_task_type(y_test=y_test, y_pred=y_pred, task_type=task_type)
