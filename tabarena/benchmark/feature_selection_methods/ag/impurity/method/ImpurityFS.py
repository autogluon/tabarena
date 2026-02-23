from __future__ import annotations

import numpy as np
import pandas as pd

import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class ImpurityFS:
    """Impurity feature selector"""

    def __init__(self, model):
        self._y = None
        self._model = model
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model, n_max_features, **kwargs) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features
        X_np = X.to_numpy()
        y_np = y.to_numpy()

        feature_ranking = self.feature_ranking(self.t_score(X_np, y_np))

        selected_features_idx = feature_ranking[:n_max_features]
        selected_features = X.columns[selected_features_idx]

        X_selected = X[selected_features]
        self._selected_features = list(X_selected.columns)
        return X_selected


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]

    def t_score(self, X, y):
        """
        This function calculates t_score for each feature, where t_score is only used for binary problem
        t_score = |mean1-mean2|/sqrt(((std1^2)/n1)+((std2^2)/n2)))

        Input
        -----
        X: {numpy array}, shape (n_samples, n_features)
            input data
        y: {numpy array}, shape (n_samples,)
            input class labels

        Output
        ------
        F: {numpy array}, shape (n_features,)
            t-score for each feature
        """

        n_samples, n_features = X.shape
        F = np.zeros(n_features)
        c = np.unique(y)
        if len(c) == 2:
            for i in range(n_features):
                f = X[:, i]
                # class0 contains instances belonging to the first class
                # class1 contains instances belonging to the second class
                class0 = f[y == c[0]]
                class1 = f[y == c[1]]
                mean0 = np.mean(class0)
                mean1 = np.mean(class1)
                std0 = np.std(class0)
                std1 = np.std(class1)
                n0 = len(class0)
                n1 = len(class1)
                t = mean0 - mean1
                t0 = np.true_divide(std0 ** 2, n0)
                t1 = np.true_divide(std1 ** 2, n1)
                F[i] = np.true_divide(t, (t0 + t1) ** 0.5)
        else:
            print('y should be guaranteed to a binary class vector')
            exit(0)
        return np.abs(F)

    def feature_ranking(self, F):
        """
        Rank features in descending order according to t-score, the higher the t-score, the more important the feature is
        """
        idx = np.argsort(F)
        return idx[::-1]
