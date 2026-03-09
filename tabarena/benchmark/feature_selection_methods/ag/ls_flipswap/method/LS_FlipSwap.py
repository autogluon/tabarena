from __future__ import annotations

import numpy as np
import pandas as pd

import warnings

from sklearn.model_selection import train_test_split

import copy
import logging
import time
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class LS_FlipSwap:
    """LS feature selector (only flip and swap)"""

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
        X_selected = self.get_ls_flipswap_indices(X, y, model, n_max_features, init_feature_choice, **kwargs)
        X_selected = pd.DataFrame(X, columns=X_selected.columns, index=X.index)
        self._selected_features = list(X_selected.columns)
        return X_selected


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]


    def get_ls_flipswap_indices(self, X_train, y_train, model, n_max_features, feature_indices, **kwargs):
        termination_condition = False
        best_score = -np.inf
        best_indices = feature_indices
        while not termination_condition:
            # Time limit
            if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                time_start_fit = time.time()
                kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                if kwargs["time_limit"] <= 0:
                    logger.warning(
                        f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                    if self._selected_features is None:
                        X_out = X_train.sample(n=n_max_features, axis=1)
                        self._selected_features = list(X_out.columns)
                        return X_out
                    else:
                        if n_max_features is not None and len(self._selected_features) > n_max_features:
                            X_out = X_train.sample(n=n_max_features, axis=1)
                            self._selected_features = list(X_out.columns)
                            return X_out
                        else:
                            return X_train[self._selected_features]
            list_of_feature_indices = self.get_flipswap_indices(feature_indices)
            print(list_of_feature_indices)
            improvement_found = False
            for feature_indices in list_of_feature_indices:
                feature_mask = [bool(i) for i in feature_indices]
                X_train_selection = X_train.iloc[:, feature_mask]
                if sum(feature_mask) <= n_max_features:
                    score = self.evaluate_subset(self, X_train_selection, y_train, model)
                    if score > best_score:
                        best_score = score
                        best_indices = feature_indices
                        improvement_found = True
            feature_indices = best_indices
            all_ones = all(i == 1 for i in feature_indices)
            if all_ones or not improvement_found:
                termination_condition = True
            self._selected_features = list(X_train_selection.columns)
        return best_indices


    @staticmethod
    def get_flipswap_indices(feature_indices):
        list_of_feature_indices = []
        print("Indices: " + str(feature_indices))
        # Flipping
        for idx, val in enumerate(feature_indices):
            new_feature_indices = feature_indices.copy()
            new_feature_indices[idx] = abs(1 - val)  # flip 0 to 1 or 1 to 0
            # Only add if at least one feature is selected (not all zeros)
            if sum(new_feature_indices) > 0:
                list_of_feature_indices.append(new_feature_indices)

        # Swapping
        for idx1 in range(len(feature_indices)):
            for idx2 in range(idx1 + 1, len(feature_indices)):
                new_feature_indices = feature_indices.copy()
                # Swap values at idx1 and idx2
                new_feature_indices[idx1], new_feature_indices[idx2] = new_feature_indices[idx2], new_feature_indices[
                    idx1]
                # Only add if at least one feature is selected (not all zeros)
                if sum(new_feature_indices) > 0:
                    list_of_feature_indices.append(new_feature_indices)
        list_of_feature_indices = [i for i in list_of_feature_indices if i != feature_indices]
        return list_of_feature_indices


    @staticmethod
    def evaluate_subset(self, X, y, model):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model_copy = copy.deepcopy(model)
        model_copy.params["fold_fitting_strategy"] = "sequential_local"
        model_copy = model_copy.fit(X=X_train, y=y_train, k_fold=8)
        self._model = model_copy
        return model_copy.score_with_oof(y=y_train)
