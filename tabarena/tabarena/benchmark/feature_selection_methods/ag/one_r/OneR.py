from autogluon.common.features.types import R_INT, R_FLOAT, R_OBJECT
from pandas import DataFrame, Series
import numpy as np

from autogluon.features.generators.abstract import AbstractFeatureSelector

import logging
import time
logger = logging.getLogger(__name__)


class OneR(AbstractFeatureSelector):
    """ OneR Feature Selection """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._one_r = None
        self._y = None
        self._model = None
        self._n_max_features = None
        self._selected_features = None


    def _fit_transform(self, X: DataFrame, y: Series, model, n_max_features: int, **kwargs) -> tuple[DataFrame, dict]:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features
        from tabarena.benchmark.feature_selection_methods.ag.one_r.method.OneRFS import OneRFS
        self._one_r = OneRFS()
        # Time limit
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
        self._one_r.fit(X.to_numpy(), y.to_numpy())
        selected_feature = self._one_r.feature_idx_
        selected_feature_list = [selected_feature] if np.isscalar(selected_feature) else list(selected_feature)
        X_out = X.iloc[:, selected_feature_list]
        if n_max_features is not None and len(X_out.columns) > n_max_features:
            X_out = X_out.sample(n=n_max_features, axis=1)
        self._selected_features = list(X_out.columns)
        type_family_groups_special = {}
        return X_out, type_family_groups_special


    def _transform(self, X: DataFrame, *, is_train: bool = False) -> DataFrame:
        if is_train:
            self._one_r.fit(X, self.y)
            selected_feature = self._one_r.feature_idx_
            selected_feature_list = [selected_feature] if np.isscalar(selected_feature) else list(selected_feature)
            X_out = X.iloc[:, selected_feature_list]
            self._selected_features = list(X_out.columns)
        else:
            selected_feature = self._one_r.feature_idx_
            selected_feature_list = [selected_feature] if np.isscalar(selected_feature) else list(selected_feature)
            X_out = X.iloc[:, selected_feature_list]
            self._selected_features = list(X_out.columns)
        return X_out


    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT, R_FLOAT, R_OBJECT])
