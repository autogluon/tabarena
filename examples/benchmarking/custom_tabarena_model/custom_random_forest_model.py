"""Example of a custom TabArena Model for a Random Forest model.

Note: due to the pickle protocol used in TabArena, the model class must be in a separate
file and not in the main script running the experiments!
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from autogluon.core.models import AbstractModel
from autogluon.features import LabelEncoderFeatureGenerator

if TYPE_CHECKING:
    import pandas as pd


class CustomRandomForestModel(AbstractModel):
    """Minimal implementation of a model compatible with the scikit-learn API.
    For more details on how to implement an abstract model, see https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-model.html
    and compare to implementations of models under tabarena.benchmark/models/ag/.
    """

    ag_key = "CRF"
    ag_name = "CustomRF"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        """Model-specific preprocessing of the input data."""
        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.fillna(0).to_numpy(dtype=np.float32)

    def _fit(
        self,
        X: pd.DataFrame,  # training data
        y: pd.Series,  # training labels
        # X_val=None,  # val data (unused in RF model)
        # y_val=None,  # val labels (unused in RF model)
        # time_limit=None,  # time limit in seconds (ignored in tutorial)
        num_cpus: int = 1,  # number of CPUs to use for training
        # num_gpus: int = 0,  # number of GPUs to use for training
        **kwargs,  # kwargs includes many other potential inputs, refer to AbstractModel documentation for details
    ):
        # Select model class
        if self.problem_type in ["regression"]:
            from sklearn.ensemble import RandomForestRegressor

            model_cls = RandomForestRegressor
        else:
            from sklearn.ensemble import RandomForestClassifier

            # case for 'binary' and 'multiclass',
            model_cls = RandomForestClassifier

        X = self.preprocess(X, y=y, is_train=True)
        params = self._get_model_params()
        self.model = model_cls(**params)
        self.model.fit(X, y)

    def _set_default_params(self):
        """Default parameters for the model."""
        default_params = {
            "n_estimators": 10,
            "n_jobs": -1,
            "random_state": 0,
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


def get_configs_for_custom_rf(*, num_random_configs: int = 1):
    """Generate the hyperparameter configurations to run for our custom model."""
    from autogluon.common.space import Int

    from tabarena.utils.config_utils import ConfigGenerator

    manual_configs = [
        {},
    ]
    search_space = {
        "n_estimators": Int(4, 50),
    }

    gen_custom_rf = ConfigGenerator(
        model_cls=CustomRandomForestModel,
        manual_configs=manual_configs,
        search_space=search_space,
    )
    return gen_custom_rf.generate_all_bag_experiments(
        num_random_configs=num_random_configs, fold_fitting_strategy="sequential_local"
    )
