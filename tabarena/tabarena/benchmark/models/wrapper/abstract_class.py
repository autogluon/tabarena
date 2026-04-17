from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from autogluon.core.data.label_cleaner import LabelCleaner, LabelCleanerDummy
from autogluon.core.metrics import Scorer
from autogluon.features import AutoMLPipelineFeatureGenerator

from tabarena.utils.time_utils import Timer
from tabarena.benchmark.models.wrapper.validation_utils import TabArenaValidationProtocolExecMixin

class AbstractExecModel(TabArenaValidationProtocolExecMixin):
    can_get_error_val = False
    can_get_oof = False
    can_get_preprocessing = False
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
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        self._can_use_data_in_place = False
        self._split_seed = "NOTSET"

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

    # TODO: Prateek, Add a toggle here to see if user wants to fit or fit and predict, also add model saving functionality
    # TODO: Nick: Temporary name
    def fit_custom(
            self,
            X: pd.DataFrame | None,
            y: pd.Series | None,
            X_test: pd.DataFrame | None,
            *,
            split_seed: int | None = None,
            lazy_load_function: Callable | None = None
    ) -> dict:
        """
        Calls the fit function of the inheriting class and proceeds to perform predictions based on the problem type

        Arguments
        ---------
        split_seed:
            If not None, the seed that is different per split to use for shuffling features.
        lazy_load_function:
            If not None, a function that one can call to load X, y, X_test lazily (e.g. to save memory by not
            loading them until needed). If provided, X, y, and X_test arguments must be None.

        Returns
        -------
        dict
            Returns predictions, probabilities, fit time and inference time
        """
        from tabarena.utils.memory_utils import CpuMemoryTracker, GpuMemoryTracker
        self._split_seed = split_seed

        if lazy_load_function is not None:
            assert X is None and y is None and X_test is None, "If lazy_load_function is provided, X and y must be None"
            X, y, _ = lazy_load_function()
            self._can_use_data_in_place = True

        shuffled_features = None
        if self.shuffle_features:
            assert split_seed is not None, "If shuffle_features is True, split_seed must not be None!"
            shuffled_features = list(X.columns)
            rng = np.random.default_rng(seed=split_seed)
            rng.shuffle(shuffled_features)
            X = X[shuffled_features]

        with CpuMemoryTracker() as cpu_tracker, GpuMemoryTracker(device=0) as gpu_tracker, Timer() as timer_fit:
            self.fit(X, y)

        # Reload all, allows X,y to be used in-place
        if lazy_load_function is not None:
            del X, y, X_test  # Free memory from previous load
            X, y, X_test = lazy_load_function()

        og_index = X_test.index
        inv_perm = None
        if self.shuffle_test:
            perm, inv_perm = _make_perm(len(X_test), seed=self.shuffle_seed)
            X_test = X_test.iloc[perm]
        if self.reset_index_test:
            X_test = X_test.reset_index(drop=True)
        if shuffled_features is not None:
            X_test = X_test[shuffled_features]
            X = X[shuffled_features]

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

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None):
        X, y = self._preprocess_fit_transform(X=X, y=y)
        if X_val is not None:
            X_val = self.transform_X(X_val)
            y_val = self.transform_y(y_val)

        return self._fit(X=X, y=y, X_val=X_val, y_val=y_val)

    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None):
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
