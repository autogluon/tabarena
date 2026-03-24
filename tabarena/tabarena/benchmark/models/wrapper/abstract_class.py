from __future__ import annotations

import numpy as np
import pandas as pd
from autogluon.core.data.label_cleaner import LabelCleaner, LabelCleanerDummy
from autogluon.core.metrics import Scorer
from autogluon.features import AutoMLPipelineFeatureGenerator

from tabarena.utils.time_utils import Timer


class AbstractExecModel:
    can_get_error_val = False
    can_get_oof = False
    can_get_per_child_oof = False
    can_get_per_child_test = False
    can_get_per_child_val_idx = False

    # TODO: Prateek: Find a way to put AutoGluon as default - in the case the user does not want their own class
    def __init__(
        self,
        problem_type: str,
        eval_metric: Scorer,
        preprocess_data: bool = True,
        preprocess_label: bool = True,
        shuffle_test: bool = True,
        shuffle_seed: int = 0,
        reset_index_test: bool = True,
        # TODO: default to True in the future.
        shuffle_features: bool = False,
        target_name: str | None = None,
        group_on: str | list[str] | None = None,
        stratify_on: str | None = None,
        time_on: str | None = None,
        group_time_on: str | None = None,
        default_n_splits: int = 8,
    ):
        """Abstract Model executor class.

        Parameters
        ----------
        target_name:
            The name of the target column. This might be needed for splitting.
        stratify_on:
            The name of the column used for stratification during splitting.
        group_on:
            Name of the column that identifies groups for group-wise splitting during
            validation. If not None, this column will be used to ensure that all rows
            with the same value in this column are kept in the same split.
        """
        self.problem_type = problem_type
        self.eval_metric = eval_metric
        self.preprocess_data = preprocess_data
        self.preprocess_label = preprocess_label
        self.shuffle_test = shuffle_test
        self.shuffle_seed = shuffle_seed
        self.reset_index_test = reset_index_test
        self.label_cleaner: LabelCleaner = None
        self._feature_generator = None
        self.failure_artifact = None
        self.shuffle_features = shuffle_features

        # New logic for group-wise splitting during validation
        self.target_name = target_name
        self.stratify_on = stratify_on
        self.group_on = group_on
        self.time_on = time_on
        self.group_time_on = group_time_on
        self.default_n_groups = default_n_splits
        self.groups_indicator_col_name = "__tabarena_group_split_indicator__"

    def transform_y(self, y: pd.Series) -> pd.Series:
        return self.label_cleaner.transform(y)

    def inverse_transform_y(self, y: pd.Series) -> pd.Series:
        return self.label_cleaner.inverse_transform(y)

    def transform_y_pred_proba(self, y_pred_proba: pd.DataFrame) -> pd.DataFrame:
        return self.label_cleaner.transform_proba(y_pred_proba, as_pandas=True)

    def inverse_transform_y_pred_proba(self, y_pred_proba: pd.DataFrame) -> pd.DataFrame:
        return self.label_cleaner.inverse_transform_proba(y_pred_proba, as_pandas=True)

    def transform_X(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.preprocess_data:
            return self._feature_generator.transform(X)
        return X

    def _preprocess_fit_transform(self, X: pd.DataFrame, y: pd.Series):
        if self.preprocess_label:
            self.label_cleaner = LabelCleaner.construct(problem_type=self.problem_type, y=y)
        else:
            self.label_cleaner = LabelCleanerDummy(problem_type=self.problem_type)
        if self.preprocess_data:
            self._feature_generator = AutoMLPipelineFeatureGenerator()
            X = self._feature_generator.fit_transform(X=X, y=y)
        y = self.transform_y(y)
        return X, y

    def post_fit(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
        pass

    def fit_custom(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, *, split_seed: int | None = None) -> dict:
        og_index = X_test.index
        inv_perm = None

        if self.shuffle_test:
            perm, inv_perm = _make_perm(len(X_test), seed=self.shuffle_seed)
            X_test = X_test.iloc[perm]
        if self.reset_index_test:
            X_test = X_test.reset_index(drop=True)
        if self.shuffle_features:
            assert split_seed is not None, "If shuffle_features is True, split_seed must not be None!"
            features = list(X.columns)
            rng = np.random.default_rng(seed=split_seed)
            rng.shuffle(features)
            X, X_test = X[features], X_test[features]

        out = self._fit_custom(X=X, y=y, X_test=X_test)

        if self.shuffle_test:
            # Inverse-permute outputs back to original X_test order
            out["predictions"] = _apply_inv_perm(out["predictions"], inv_perm, index=og_index)
            if out["probabilities"] is not None:
                out["probabilities"] = _apply_inv_perm(out["probabilities"], inv_perm, index=og_index)
        elif self.reset_index_test:
            out["predictions"].index = og_index
            if out["probabilities"] is not None:
                out["probabilities"].index = og_index

        return out

    def _resolve_custom_splits(self, *, X: pd.DataFrame, y: pd.Series) -> pd.Series | None:
        """Build a custom group split indicator if needed.

        Returns
        -------
        groups_indicator: np.ndarray or None
            None, if no splits indicator is needed.
            Otherwise, a Series of shape (n_samples,) where each value is an integer
            representing the split assignment for that row.
        """
        groups_indicator = None

        if self.group_on is not None:
            groups_indicator = self._resolve_group_splits(X=X, y=y)

        return groups_indicator

    def _resolve_group_splits(self, *, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Create a group-based split given the specified group_on column(s).
        Then transform this into split indices for AutoGluon's group splitter logic.

        Some comments about this logic:
            - If we are given a list of group_on columns, we create a combined group
                identifier by concatenating the values in those columns.
            - We use stratification if stratify_on is specified.
            - We dynamically adjust the number of splits to be the minimum of
                default_n_splits (=8) and the number of unique groups in the data.
        """
        from sklearn.model_selection import (
            StratifiedGroupKFold,
            GroupKFold,
        )

        # Get group label
        group_col = self.group_on
        if isinstance(group_col, list):
            # If multiple columns are specified, create a combined group identifier
            groups_data = X[group_col].astype(str).agg("_".join, axis=1)
        else:
            groups_data = X[group_col]

        n_groups_in_data = groups_data.nunique()
        assert n_groups_in_data > 1, f"Need at least 2 unique groups for group-wise splitting, but got {n_groups_in_data} unique groups from column(s) {group_col}!"
        n_splits = min(self.default_n_groups, n_groups_in_data)
        print(f"Found #groups in data: {n_groups_in_data}")

        if self.stratify_on is None:
            stratify_on_data = None
        else:
            assert self.target_name is not None
            if self.stratify_on == self.target_name:
                stratify_on_data = y
            else:
                assert self.stratify_on in X.columns, f"Stratification column '{self.stratify_on}' not found in features!"
                stratify_on_data = X[self.stratify_on]

        splitter = GroupKFold if stratify_on_data is None else StratifiedGroupKFold
        sklearn_splits = splitter(
            n_splits=n_splits, random_state=42, shuffle=True
        ).split(X=X, y=y, groups=groups_data)

        groups_indicator = np.full(shape=len(X), fill_value=-1, dtype=int)
        for fold_idx, (_, test_index) in enumerate(sklearn_splits):
            groups_indicator[test_index] = fold_idx

        return pd.Series(groups_indicator)

    # TODO: Prateek, Add a toggle here to see if user wants to fit or fit and predict, also add model saving functionality
    # TODO: Nick: Temporary name
    def _fit_custom(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame) -> dict:
        """
        Calls the fit function of the inheriting class and proceeds to perform predictions based on the problem type

        Returns
        -------
        dict
            Returns predictions, probabilities, fit time and inference time
        """
        from tabarena.utils.memory_utils import CpuMemoryTracker, GpuMemoryTracker

        with CpuMemoryTracker() as cpu_tracker, GpuMemoryTracker(device=0) as gpu_tracker, Timer() as timer_fit:
            self.fit(X, y)

        self.post_fit(X=X, y=y, X_test=X_test)

        if self.problem_type in ["binary", "multiclass"]:
            with Timer() as timer_predict:
                y_pred_proba = self.predict_proba(X_test)
            y_pred = self.predict_from_proba(y_pred_proba)
        else:
            with Timer() as timer_predict:
                y_pred = self.predict(X_test)
            y_pred_proba = None

        out = {
            "predictions": y_pred,
            "probabilities": y_pred_proba,
            "time_train_s": timer_fit.duration,
            "time_infer_s": timer_predict.duration,
        }

        out["memory_usage"] = dict(
            peak_mem_cpu=cpu_tracker.peak_rss,
            min_mem_cpu=cpu_tracker.min_rss,

            peak_mem_gpu=gpu_tracker.peak_allocated,
            peak_mem_gpu_reserved=gpu_tracker.peak_reserved,
            min_mem_gpu=gpu_tracker.min_allocated,
            min_mem_gpu_reserved=gpu_tracker.min_reserved,

            gpu_tracking_enabled=gpu_tracker.enabled,
        )

        return out

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None):
        X, y = self._preprocess_fit_transform(X=X, y=y)
        if X_val is not None:
            X_val = self.transform_X(X_val)
            y_val = self.transform_y(y_val)
            groups_indicator = None
        else:
            groups_indicator = self._resolve_custom_splits(X=X, y=y)

        return self._fit(X=X, y=y, X_val=X_val, y_val=y_val, groups_indicator=groups_indicator)

    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None, groups_indicator: pd.Series | None =None):
        raise NotImplementedError

    def predict_from_proba(self, y_pred_proba: pd.DataFrame) -> pd.Series:
        if isinstance(y_pred_proba, pd.DataFrame):
            return y_pred_proba.idxmax(axis=1)
        else:
            return np.argmax(y_pred_proba, axis=1)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = self.transform_X(X=X)
        y_pred = self._predict(X)
        return self.inverse_transform_y(y=y_pred)

    def _predict(self, X: pd.DataFrame):
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self.transform_X(X=X)
        y_pred_proba = self._predict_proba(X=X)
        return self.inverse_transform_y_pred_proba(y_pred_proba=y_pred_proba)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def cleanup(self):
        pass

    def get_metric_error_val(self) -> float:
        raise NotImplementedError


def _make_perm(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (perm, inv_perm) for length n, using a deterministic RNG seed."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(n)
    return perm, inv_perm


def _apply_inv_perm(obj, inv_perm: np.ndarray, index: pd.Index | None = None):
    """Inverse-permute predictions while preserving type (Series/DataFrame/ndarray)."""
    if isinstance(obj, pd.Series):
        vals = obj.to_numpy()[inv_perm]
        return pd.Series(vals, index=index, name=obj.name)
    if isinstance(obj, pd.DataFrame):
        vals = obj.to_numpy()[inv_perm, :]
        return pd.DataFrame(vals, index=index, columns=obj.columns)
    # Fallback: numpy array or array-like
    arr = np.asarray(obj)
    return arr[inv_perm]
