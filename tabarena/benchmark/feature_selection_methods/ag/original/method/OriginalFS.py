from __future__ import annotations

import pandas as pd

import warnings
import logging
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class OriginalFS:
    """Original feature selector (select all features) """

    def __init__(self):
        self._y = None
        self._model = None
        self._n_max_features = None
        self._selected_features = None


    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model, n_max_features, **kwargs) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features
        self._selected_features = list(X.columns)
        return X


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self._selected_features = list(X.columns)
        return X