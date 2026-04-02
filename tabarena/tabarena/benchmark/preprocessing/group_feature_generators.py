from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from autogluon.common.features.types import (
    R_BOOL,
    R_CATEGORY,
    R_FLOAT,
    R_INT,
    R_OBJECT,
    S_DATETIME_AS_OBJECT,
    S_IMAGE_BYTEARRAY,
    S_IMAGE_PATH,
    S_TEXT,
    S_TEXT_SPECIAL,
)
from autogluon.features import AbstractFeatureGenerator

GROUP_INDEX_FEATURES = "group_index_features"


class GroupAggregationFeatureGenerator(AbstractFeatureGenerator):
    """Pre-generator that handles group columns in the preprocessing pipeline.

    When ``generate_features=True`` (dataset has per-group labels):
      - Computes per-group aggregations (mean/std/min/max/last for numeric;
        count/last/nunique for non-numeric) joined back to all original rows.
      - Scores numeric agg features by absolute Pearson correlation with ``y``;
        scores categorical agg features by ``1 - normalized_entropy`` of their
        value distribution (unsupervised, no ``y`` needed).
      - Selects the top ``n_top_features`` columns across both score types.
      - At transform time only the aggregations needed for selected features
        are recomputed (no wasted work on discarded columns).
      - Drops the group column(s) from output.

    When ``generate_features=False`` (per-sample labels or unspecified):
      - Simply drops the group column(s) from output without adding any new
        features.

    Parameters
    ----------
    group_col:
        Name(s) of the group column(s).  When a list is passed the columns are
        concatenated with ``"_"`` to form a single composite key.
    generate_index_features:
        If ``True`` compute and join aggregation features; if ``False`` only
        drop the group column(s).
    n_top_features:
        Maximum number of aggregation columns to retain.
    """

    _NUM_AGGS = ("mean", "std", "min", "max", "last")
    _CAT_AGGS = ("count", "last", "nunique")

    def __init__(
        self,
        *,
        group_col: str | list[str],
        generate_index_features: bool = True,
        n_top_features: int = 50,
        group_time_on: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.group_col: list[str] = [group_col] if isinstance(group_col, str) else list(group_col)
        self.generate_features = generate_index_features
        self.n_top_features = n_top_features
        self.group_time_on = group_time_on
        self._selected_features: list[str] = []
        # Maps kept for efficient test-time aggregation:
        # {source_col: [agg_func, ...]} for each feature type.
        self._num_agg_map: dict[str, list[str]] = {}
        self._cat_agg_map: dict[str, list[str]] = {}

    def _fit_transform(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> tuple[pd.DataFrame, dict]:
        if not self.generate_features:
            self._log(
                20,
                f"GroupAggregationFeatureGenerator: dropping group columns "
                f"{self.group_col} without generating features "
                f"(group_labels is not PER_GROUP).",
            )
            return X.drop(columns=self.group_col, errors="ignore"), {}

        self._log(
            20,
            f"GroupAggregationFeatureGenerator: generating groupby aggregation "
            f"features for group columns {self.group_col}.",
        )

        X_agg, feature_source = self._compute_all_agg_features(X)
        group_key = self._build_group_key(X)
        self._log(20, f"GroupAggregationFeatureGenerator: dropping group columns {self.group_col}.")
        X = X.drop(columns=self.group_col)

        for col in X_agg.columns:
            X[col] = group_key.map(X_agg[col])

        # Score and select top-N features.
        scores = self._score_features(X, list(X_agg.columns), y)
        sorted_features = sorted(scores, key=scores.__getitem__, reverse=True)
        self._selected_features = sorted_features[: self.n_top_features]

        # Build efficient test-time agg maps: only compute what was selected.
        num_map: dict[str, list[str]] = defaultdict(list)
        cat_map: dict[str, list[str]] = defaultdict(list)
        for feat in self._selected_features:
            src_col, agg_func, is_num = feature_source[feat]
            if is_num:
                num_map[src_col].append(agg_func)
            else:
                cat_map[src_col].append(agg_func)
        self._num_agg_map = dict(num_map)
        self._cat_agg_map = dict(cat_map)

        drop_cols = [c for c in X_agg.columns if c not in self._selected_features]
        if drop_cols:
            X = X.drop(columns=drop_cols)

        self._log(
            20,
            f"GroupAggregationFeatureGenerator: selected {len(self._selected_features)} "
            f"groupby features out of {len(X_agg.columns)} generated.",
        )
        type_family_groups_special = {GROUP_INDEX_FEATURES: list(X_agg.columns)}
        return X, type_family_groups_special

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.generate_features:
            return X.drop(columns=self.group_col, errors="ignore")

        # Only compute the aggregations needed for selected features.
        X_agg = self._compute_selected_agg_features(X)
        group_key = self._build_group_key(X)
        X = X.drop(columns=self.group_col, errors="ignore")

        for col in self._selected_features:
            if col in X_agg.columns:
                X[col] = group_key.map(X_agg[col])
            else:
                X[col] = float("nan")
        return X

    def _score_features(self, X: pd.DataFrame, agg_cols: list[str], y: pd.Series) -> dict[str, float]:
        """Return a score in [0, 1] for every agg column.

        Numeric columns are scored by |Pearson corr with y|.
        Non-numeric columns are scored by ``1 - normalized_entropy`` of their
        value distribution (unsupervised).
        """
        scores: dict[str, float] = {}

        num_cols = [c for c in agg_cols if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in agg_cols if not pd.api.types.is_numeric_dtype(X[c])]

        if num_cols:
            y_numeric = pd.to_numeric(y, errors="coerce")
            corrs = X[num_cols].corrwith(y_numeric).abs().fillna(0.0)
            scores.update(corrs.to_dict())

        for col in cat_cols:
            scores[col] = self._concentration_score(X[col])

        return scores

    @staticmethod
    def _concentration_score(series: pd.Series) -> float:
        """Unsupervised score for a categorical series: ``1 - normalized_entropy``.

        A column where all groups share the same last category scores 1.0
        (maximally concentrated); a uniformly distributed column scores 0.0.
        """
        counts = series.dropna().value_counts()
        n_unique = len(counts)
        if n_unique <= 1:
            return 1.0
        probs = counts / counts.sum()
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        max_entropy = float(np.log(n_unique))
        return 1.0 - entropy / max_entropy if max_entropy > 0 else 1.0

    def _sort_by_time(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return X with rows within each group sorted by ``group_time_on``.

        Uses groupby.apply so only the within-group order changes; ``last``
        then returns the most recent observation for each group.
        """
        if self.group_time_on is not None and self.group_time_on in X.columns:
            sorted_index = (
                X.groupby(self.group_col, sort=False, group_keys=False)
                .apply(lambda g: g.sort_values(self.group_time_on), include_groups=False)
                .index
            )
            return X.loc[sorted_index]
        return X

    def _compute_all_agg_features(self, X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[str, str, bool]]]:
        """Compute all per-group aggregations.

        Returns:
        -------
        agg_df : pd.DataFrame
            One row per unique group key.
        feature_source : dict
            ``{feature_name: (source_col, agg_func, is_numeric)}``
        """
        X = self._sort_by_time(X)
        group_key = self._build_group_key(X)
        feature_cols = [c for c in X.columns if c not in self.group_col]
        num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(X[c])]

        feature_source: dict[str, tuple[str, str, bool]] = {}
        parts: list[pd.DataFrame] = []

        if num_cols:
            num_agg = X.groupby(group_key)[num_cols].agg(list(self._NUM_AGGS))
            num_agg.columns = ["_".join(x) for x in num_agg.columns]
            for col in num_cols:
                for agg in self._NUM_AGGS:
                    feature_source[f"{col}_{agg}"] = (col, agg, True)
            parts.append(num_agg)

        if cat_cols:
            cat_agg = X.groupby(group_key)[cat_cols].agg(list(self._CAT_AGGS))
            cat_agg.columns = ["_".join(x) for x in cat_agg.columns]
            for col in cat_cols:
                for agg in self._CAT_AGGS:
                    feature_source[f"{col}_{agg}"] = (col, agg, False)
            parts.append(cat_agg)

        if not parts:
            return pd.DataFrame(), {}

        return pd.concat(parts, axis=1), feature_source

    def _compute_selected_agg_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute only the aggregations needed for the selected features."""
        X = self._sort_by_time(X)
        group_key = self._build_group_key(X)
        parts: list[pd.DataFrame] = []

        if self._num_agg_map:
            for col, aggs in self._num_agg_map.items():
                if col not in X.columns:
                    continue
                agg_df = X.groupby(group_key)[[col]].agg(aggs)
                agg_df.columns = [f"{col}_{a}" for a in aggs]
                parts.append(agg_df)

        if self._cat_agg_map:
            for col, aggs in self._cat_agg_map.items():
                if col not in X.columns:
                    continue
                agg_df = X.groupby(group_key)[[col]].agg(aggs)
                agg_df.columns = [f"{col}_{a}" for a in aggs]
                parts.append(agg_df)

        if not parts:
            return pd.DataFrame()

        return pd.concat(parts, axis=1)

    def _build_group_key(self, X: pd.DataFrame) -> pd.Series:
        """Return a Series (same index as X) with the composite group key."""
        if len(self.group_col) == 1:
            return X[self.group_col[0]]
        return X[self.group_col].astype(str).agg("_".join, axis=1)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return {
            "valid_raw_types": [R_OBJECT, R_CATEGORY, R_BOOL, R_FLOAT, R_INT],
            "invalid_special_types": [
                S_DATETIME_AS_OBJECT,
                S_IMAGE_PATH,
                S_IMAGE_BYTEARRAY,
                S_TEXT,
                S_TEXT_SPECIAL,
            ],
        }
