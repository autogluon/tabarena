from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.common.features.types import (
    R_DATETIME,
    S_DATETIME_AS_INT,
    S_DATETIME_AS_OBJECT,
)
from autogluon.features import AbstractFeatureGenerator

if TYPE_CHECKING:
    import pandas as pd


class DateTimeFeatureGenerator(AbstractFeatureGenerator):
    """Generate features from datetime columns using skrub."""

    def _fit_transform(self, X: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
        from skrub import DatetimeEncoder, TableVectorizer

        self._vectorizer = TableVectorizer(
            low_cardinality="drop",
            high_cardinality="drop",
            numeric="drop",
            datetime=DatetimeEncoder(add_weekday=True, periodic_encoding="circular"),
        )

        X_out = self._transform(X, is_train=True)
        # Not int, but should be fine.
        type_family_groups_special = {S_DATETIME_AS_INT: list(X_out.columns)}
        return X_out, type_family_groups_special

    def _transform(self, X: pd.DataFrame, *, is_train: bool = False) -> pd.DataFrame:
        if is_train:
            X = self._vectorizer.fit_transform(X)
        else:
            X = self._vectorizer.transform(X)

        return X

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return {
            "required_raw_special_pairs": [
                (R_DATETIME, None),
                (None, [S_DATETIME_AS_OBJECT]),
            ]
        }
