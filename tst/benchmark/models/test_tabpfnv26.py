from __future__ import annotations

import pytest


def test_tabpfn26():
    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model import (
            TabPFNv26Model,
        )

        model_cls = TabPFNv26Model
        FitHelper.verify_model(
            model_cls=model_cls,
            model_hyperparameters={
                "n_estimators": 1,
            },
        )
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... Ensure you have the proper dependencies installed to run this test:\n{err}"
        )


def test_tabpfnv26_many_class():
    """Test that RealTabPFNv25 handles >10 classes via ManyClassClassifier."""
    try:
        import numpy as np
        import pandas as pd
        from autogluon.core.data import LabelCleaner
        from autogluon.features.generators import AutoMLPipelineFeatureGenerator
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model import (
            TabPFNv26Model,
        )

        n_classes = 15
        X_np, y_np = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=8,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=42,
        )
        X = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(X_np.shape[1])])
        y = pd.Series(y_np, name="target")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        task_type = "multiclass"
        feature_generator = AutoMLPipelineFeatureGenerator()
        label_cleaner = LabelCleaner.construct(problem_type=task_type, y=y_train)

        X_train = feature_generator.fit_transform(X_train)
        y_train = label_cleaner.transform(y_train)
        X_test = feature_generator.transform(X_test)
        y_test = label_cleaner.transform(y_test)

        model = TabPFNv26Model(
            problem_type=task_type,
            hyperparameters={"n_estimators": 1},
        )
        model = model.fit(X=X_train, y=y_train)

        # Verify ManyClassClassifier was used
        from tabpfn_extensions.many_class import ManyClassClassifier

        assert isinstance(model.model, ManyClassClassifier), (
            f"Expected ManyClassClassifier wrapper for {n_classes} classes, got {type(model.model)}"
        )

        # Predict and verify output shapes
        y_pred = model.predict(X=X_test)
        assert len(y_pred) == len(y_test)

        y_pred_proba = model.predict_proba(X=X_test)
        assert y_pred_proba.shape == (len(y_test), n_classes)

        # Probabilities should sum to ~1
        row_sums = np.sum(y_pred_proba, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... Ensure you have the proper dependencies installed to run this test:\n{err}"
        )