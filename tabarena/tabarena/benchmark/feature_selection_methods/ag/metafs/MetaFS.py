import logging
import time

from pandas import DataFrame, Series

from autogluon.common.features.types import R_INT, R_FLOAT, R_OBJECT
from autogluon.features.generators.abstract import AbstractFeatureSelector
logger = logging.getLogger(__name__)


class MetaFS(AbstractFeatureSelector):
    """MetaFS feature selection using metalearning approach."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._metafs = None
        self._y = None
        self._model = None
        self._n_max_features = None
        self._selected_features = None


    def _fit_transform(self, X: DataFrame, y: Series, model, n_max_features: int, **kwargs) -> tuple[DataFrame, dict]:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features
        from tabarena.benchmark.feature_selection_methods.ag.metafs.method.MetaFS import MetaFS
        self._metafs = MetaFS(model)
        # Time limit
        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
            time_start_fit = time.time()
            kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
            if kwargs["time_limit"] <= 0:
                logger.warning(
                    f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                if n_max_features is not None and len(X.columns) > n_max_features:
                    X_out = X.sample(n=n_max_features, axis=1)
                    return X_out
                else:
                    return X

        X_train_fs = self._metafs.fit_transform(X, y, model, n_max_features, **kwargs)

        self._selected_features = list(X_train_fs.columns)
        X_out = X[self._selected_features].copy()
        type_family_groups_special = {}
        return X_out, type_family_groups_special


    def _transform(self, X: DataFrame, *, is_train: bool = False) -> DataFrame:
        if is_train:
            X = self._metafs.fit_transform(X, self._y, self._model, self._n_max_features)
            self._selected_features = list(X.columns)
        else:
            X = X[self._selected_features]
        return X


    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT, R_FLOAT, R_OBJECT])
