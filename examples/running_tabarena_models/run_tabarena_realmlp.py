"""Minimal Example to use a TabArena model for a new dataset (without the TabArena benchmark).

As all models in TabArena are (custom) AutoGluon models, we can use the API from AutoGluon to train, evaluate,
and deploy them. For details, see https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-model.html
"""

from __future__ import annotations

from autogluon.core.data import LabelCleaner
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Import a TabArena model
from tabarena.models.realmlp.model import RealMLPModel

# Get Data
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Preprocessing
feature_generator, label_cleaner = (
    AutoMLPipelineFeatureGenerator(),
    LabelCleaner.construct(problem_type="binary", y=y_train),
)
X_train, y_train = (
    feature_generator.fit_transform(X_train),
    label_cleaner.transform(y_train),
)
X_test, y_test = feature_generator.transform(X_test), label_cleaner.transform(y_test)

# Train TabArena Model
clf = RealMLPModel(problem_type="binary")
clf.fit(X=X_train, y=y_train)

# Predict and score
prediction_probabilities = clf.predict_proba(X=X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities))
