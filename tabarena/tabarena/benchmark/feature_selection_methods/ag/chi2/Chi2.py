from autogluon.common.features.types import R_INT, R_FLOAT, R_OBJECT
from pandas import DataFrame, Series

from autogluon.features.generators.abstract import AbstractFeatureSelector

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

import logging
import time
logger = logging.getLogger(__name__)


class Chi2(AbstractFeatureSelector):
    """ Select k best features from the data using Chi^2 score """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._chi2 = None
        self._y = None
        self._model = None
        self._n_max_features = None
        self._selected_features = None

    def _fit_transform(self, X: DataFrame, y: Series, model, n_max_features: int, **kwargs) -> tuple[DataFrame, dict]:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features

        self._select_best_kwargs = {"score_func": chi2, "k": n_max_features}
        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
            time_start_fit = time.time()
            kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
            kwargs["start_time"] = time_start_fit
            if kwargs["time_limit"] <= 0:
                logger.warning(
                    f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                if n_max_features is not None and len(X.columns) > n_max_features:
                    X_out = X.sample(n=n_max_features, axis=1)
                    return X_out
                else:
                    return X
        self._chi2 = SelectKBest(**self._select_best_kwargs).set_output(transform="pandas")
        X_out = self._transform(X, is_train=True)
        self._selected_features = list(X_out.columns)
        type_family_groups_special = {}
        return X_out, type_family_groups_special


    def _transform(self, X: DataFrame, *, is_train: bool = False) -> DataFrame:
        if is_train:
            X = self._chi2.fit_transform(X, self._y)
        else:
            X = self._chi2.transform(X)
        return X


    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT, R_FLOAT, R_OBJECT])
