from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from autogluon.common.features.types import R_FLOAT, R_INT, R_OBJECT
from autogluon.features.generators.abstract import AbstractFeatureSelector

if TYPE_CHECKING:
    from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


class Original(AbstractFeatureSelector):
    """Original feature selector (select all features)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._original = None
        self._y = None
        self._model = None
        self._n_max_features = None
        self._selected_features = None

    def _fit_transform(self, X: DataFrame, y: Series, model, n_max_features: int, **kwargs) -> tuple[DataFrame, dict]:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features
        from tabarena.benchmark.feature_selection_methods.ag.original.method.OriginalFS import OriginalFS

        self._original = OriginalFS()
        X_out = self._original.fit_transform(X, y, model, n_max_features, **kwargs)
        self._selected_features = list(X_out.columns)
        type_family_groups_special = {}
        return X_out, type_family_groups_special

    def _transform(self, X: DataFrame, *, is_train: bool = False) -> DataFrame:
        if is_train:
            X = self._original.fit_transform(X, self._y, self._model, self._n_max_features)
            self._selected_features = list(X.columns)
        else:
            X = X[self._original._selected_features]
        return X

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT, R_FLOAT, R_OBJECT])
