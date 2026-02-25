from __future__ import annotations

import numpy as np
import pandas as pd

import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import copy
import logging
import time

from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class CARTFS:
    """CART feature selector"""

    def __init__(self):
        self._y = None
        self._model = None
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model, n_max_features, **kwargs) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features

        feature_ranking = self.cart(X, y)

        sorted_idx = np.argsort(-feature_ranking)
        selected_features_idx = sorted_idx[:n_max_features]
        selected_features = X.columns[selected_features_idx]
        X_selected = X[selected_features]
        self._selected_features = list(X_selected.columns)
        return X_selected


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]


    def cart(self, X_train, y_train):
        CART = DecisionTreeClassifier(random_state=0)
        CART.fit(X_train, y_train)
        importances = CART.feature_importances_
        # std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
        return importances
