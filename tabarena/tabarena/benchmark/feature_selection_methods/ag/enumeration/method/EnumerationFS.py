from __future__ import annotations

import copy
import logging
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class EnumerationFS:
    """Enumeration feature selector."""

    def __init__(self, model):
        self._y = None
        self._model = model
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model, n_max_features, **kwargs) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features
        init_feature_choice = [0] * len(X.columns)
        X_selected = self.get_enumerated_indices(X, y, model, n_max_features, init_feature_choice, **kwargs)
        X_selected = pd.DataFrame(X, columns=X_selected.columns, index=X.index)
        self._selected_features = list(X_selected.columns)
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]

    from itertools import combinations

    def get_enumerated_indices(self, X_train, y_train, model, n_max_features, feature_indices, **kwargs):
        """Enumerate all possible feature combinations starting from n_max_features down to 1 as long as we have time."""
        best_score = -np.inf
        best_indices = None
        best_selected_features = None
        n_features_total = len(feature_indices)

        # Start from n_max_features and go down to 1
        for num_features in range(n_max_features, 0, -1):
            # Generate all combinations of feature indices for current num_features
            feature_indices_list = list(range(n_features_total))

            for feature_combo in combinations(feature_indices_list, num_features):
                # Time limit check
                if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                    time_cur = time.time()
                    kwargs["time_limit"] -= time_cur - kwargs["start_time"]
                    if kwargs["time_limit"] <= 0:
                        logger.warning(
                            f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                        )
                        if best_indices is not None:
                            self._selected_features = best_selected_features
                            return X_train[self._selected_features]
                        # Fallback: randomly select n_max_features (will not be triggered normally)
                        X_out = X_train.sample(n=n_max_features, axis=1)
                        self._selected_features = list(X_out.columns)
                        return X_out

                # Create feature indices array
                feature_indices_candidate = [0] * n_features_total
                for idx in feature_combo:
                    feature_indices_candidate[idx] = 1
                print(feature_indices_candidate)
                feature_mask = [bool(i) for i in feature_indices_candidate]
                X_train_selection = X_train.iloc[:, feature_mask]

                # Evaluate this combination
                score = self.evaluate_subset(self, X_train_selection, y_train, model)
                if score > best_score:
                    best_score = score
                    best_indices = feature_indices_candidate
                    best_selected_features = list(X_train_selection.columns)
        self._selected_features = best_selected_features
        return best_indices

    @staticmethod
    def evaluate_subset(self, X, y, model):
        X_train, _X_test, y_train, _y_test = train_test_split(X, y, test_size=0.2)
        model_copy = copy.deepcopy(model)
        model_copy.params["fold_fitting_strategy"] = "sequential_local"
        model_copy = model_copy.fit(X=X_train, y=y_train, k_fold=8)
        self._model = model_copy
        return model_copy.score_with_oof(y=y_train)
