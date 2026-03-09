"""Example of a custom TabArena Model for a Random Forest model.

Note: due to the pickle protocol used in TabArena, the model class must be in a separate
file and not in the main script running the experiments!
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from autogluon.core.models import AbstractModel
from autogluon.features import AutoMLPipelineFeatureGenerator
from autogluon.features.generators.selection import FeatureSelectionGenerator

if TYPE_CHECKING:
    import pandas as pd


class RandomFeatureSelectionModel(AbstractModel):
    """Minimal implementation of a model compatible with the scikit-learn API.
    For more details on how to implement an abstract model, see https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-model.html
    and compare to implementations of models under tabarena.benchmark/models/ag/.
    """

    ag_key = "Boruta_LGBM"
    ag_name = "Boruta_LightGBM"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._feature_generator_fitted = False
        self._y = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        """Model-specific preprocessing of the input data."""
        """ # X = super()._preprocess(X, **kwargs)
        X = X.fillna(X.mean(numeric_only=True))
        X = X.fillna(X.mode().iloc[0])

        # Fit feature selection HERE on full training data
        self._feature_selection_generator = AutoMLPipelineFeatureGenerator(
            post_generators=[FeatureSelectionGenerator("SelectKBest")])
        X = self._feature_selection_generator.fit_transform(X=X, y=self._y)
        self.features = list(X.columns)"""
        """if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        if is_train:
            self._feature_generator = FillNaFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.fillna(0).to_numpy(dtype=np.float32)"""
        X = X.fillna(X.mean(numeric_only=True))  # For numeric columns
        X = X.fillna(X.mode().iloc[0])  # For categorical columns
        # Do Feature Selection
        if is_train:
            # Only create and fit once
            if self._feature_generator is None:
                self._feature_generator = AutoMLPipelineFeatureGenerator(
                    post_generators=[FeatureSelectionGenerator("Boruta")]
                )

            if not self._feature_generator_fitted:
                X = X.reset_index(drop=True)
                y = self._y.reset_index(drop=True)
                X = self._feature_generator.fit_transform(X, y, is_train=True)
                self._feature_generator_fitted = True
                self._feature_generator.features_in = list(X.columns)
                self.features = list(X.columns)
            else:
                # If already fitted, just transform
                X = self._feature_generator.transform(X)
        else:
            # During prediction/validation
            if self._feature_generator is not None:
                X = self._feature_generator.transform(X)
        return X

    def _fit(
        self,
        X: pd.DataFrame,  # training data
        y: pd.Series,  # training labels
        num_cpus: int = 1,  # number of CPUs to use for training
        **kwargs,  # kwargs includes many other potential inputs, refer to AbstractModel documentation for details
    ):
        self._y = y
        if self.problem_type in ["regression"]:
            from sklearn.ensemble import HistGradientBoostingRegressor
            model_cls = HistGradientBoostingRegressor
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            model_cls = HistGradientBoostingClassifier
        # X = self.preprocess(X, is_train=True)
        params = self._get_model_params()
        self.model = model_cls(**params)
        self.model.fit(X, y)


    def _set_default_params(self):
        """Default parameters for the model."""
        default_params = {
            "max_iter": 10,
            "random_state": 1,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        """Specifics allowed input data and that all other dtypes should be handled
        by the model-agnostic preprocessor.
        """
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = {
            "valid_raw_types": ["int", "float", "category"],
        }
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params


def get_configs_for_custom_model_fs(*, num_random_configs: int = 1):
    """Generate the hyperparameter configurations to run for our custom model."""
    from autogluon.common.space import Int

    from tabarena.utils.config_utils import ConfigGenerator

    manual_configs = [
        {},
    ]
    search_space = {
        "max_iter": Int(1, 20),
    }

    gen_custom_fs = ConfigGenerator(
        model_cls=RandomFeatureSelectionModel,
        manual_configs=manual_configs,
        search_space=search_space,
    )
    return gen_custom_fs.generate_all_bag_experiments(
        num_random_configs=num_random_configs, fold_fitting_strategy="sequential_local"
    )
