from pandas import DataFrame, Series

from mafese import Data
from mafese import UnsupervisedSelector

from autogluon.common.features.types import R_INT, R_FLOAT, R_OBJECT
from autogluon.features.generators.abstract import AbstractFeatureSelector

import logging
import time

logger = logging.getLogger(__name__)


class MAFESE(AbstractFeatureSelector):
    """ Select features from the data using Boruta algorithm """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mafese = None
        self._y = None
        self._model = None
        self._n_max_features = None
        self._selected_features = None

    def _fit_transform(self, X: DataFrame, y: Series, model, n_max_features, **kwargs) -> tuple[DataFrame, dict]:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features

        problem_type = self._infer_problem_type(y=y)
        if problem_type == "binary":
            problem = "classification"
        elif problem_type == "multiclass":
            problem = "classification"
        else:
            problem = "regression"

        data = Data(X, y)
        data.split_train_test(test_size=0.1, inplace=True)

        data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
        data.X_test = scaler_X.transform(data.X_test)
        data.y_train, scaler_y = data.encode_label(data.y_train)  # This is for classification problem only
        data.y_test = scaler_y.transform(data.y_test)

        self._mafese_kwargs = {"problem": problem, "method": "DR", "n_features": n_max_features}
        self._mafese = UnsupervisedSelector(**self._mafese_kwargs)
        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
            time_start_fit = time.time()
            kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
            if kwargs["time_limit"] <= 0:
                logger.warning(
                    f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                if n_max_features is not None and len(X.columns) > n_max_features:
                    X_out = X.sample(n=n_max_features, axis=1)
                    return X_out
                else:
                    return X
        self._mafese.fit(data.X_train, data.y_train)
        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
            time_start_fit = time.time()
            kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
            if kwargs["time_limit"] <= 0:
                logger.warning(
                    f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                if n_max_features is not None and len(X.columns) > n_max_features:
                    X_out = X.sample(n=n_max_features, axis=1)
                    return X_out
                else:
                    return X
        selected_features_indexes = self._mafese.selected_feature_indexes
        X_out = X.iloc[:, selected_features_indexes]
        self._selected_features = list(X_out.columns)
        type_family_groups_special = {}
        return X_out, type_family_groups_special

    def _transform(self, X: DataFrame, *, is_train: bool = False) -> DataFrame:
        if is_train:
            X = self._mafese.fit_transform(X, self._y, self._model, self._n_max_features)
            self._selected_features = list(X.columns)
        else:
            X = X[self._selected_features]
        return X

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT, R_FLOAT, R_OBJECT])
