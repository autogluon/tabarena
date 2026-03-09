from __future__ import annotations

import logging
import time
import warnings

import pandas as pd
import sklearn

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class LassoFS:
    """Lasso feature selector."""

    def __init__(self, model):
        self._y = None
        self._model = model
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model, n_max_features, **kwargs) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features
        X_selected = self.lasso(X, y, model, n_max_features, **kwargs)
        X_selected = pd.DataFrame(X, columns=X_selected.columns, index=X.index)
        self._selected_features = list(X_selected.columns)
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]

    def lasso(self, X_train, y_train, model, n_max_features, **kwargs):
        lasso = sklearn.linear_model.Lasso(random_state=1)
        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
            time_start_fit = time.time()
            kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
            kwargs["start_time"] = time_start_fit
            if kwargs["time_limit"] <= 0:
                logger.warning(
                    f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                )
                if n_max_features is not None and len(X_train.columns) > n_max_features:
                    return X_train.sample(n=n_max_features, axis=1)
                return X_train
        lasso.fit(X_train, y_train)
        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
            time_start_fit = time.time()
            kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
            kwargs["start_time"] = time_start_fit
            if kwargs["time_limit"] <= 0:
                logger.warning(
                    f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                )
                if n_max_features is not None and len(X_train.columns) > n_max_features:
                    return X_train.sample(n=n_max_features, axis=1)
                return X_train
        selector = sklearn.feature_selection.SelectFromModel(lasso, prefit=True)
        X_train_selected = selector.transform(X_train)
        selected_features = X_train.columns[selector.get_support()].tolist()
        return pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
