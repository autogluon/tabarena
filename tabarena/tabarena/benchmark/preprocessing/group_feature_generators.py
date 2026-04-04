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
from tqdm import tqdm

GROUP_INDEX_FEATURES = "group_index_features"


class GroupAggregationFeatureGenerator(AbstractFeatureGenerator):
    """Pre-generator that handles group columns in the preprocessing pipeline.

    When ``generate_features=True`` (dataset has per-group labels):
      - Computes per-group aggregations (mean/std/min/max/last for numeric;
        count/last/nunique for non-numeric) joined back to all original rows.
      - Selects the top ``n_top_features`` aggregation columns by variance
        (unsupervised — no target information used during selection).
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
            f"features and then drop group columns {self.group_col}.",
        )

        group_key = self._build_group_key(X)
        feature_cols = [c for c in X.columns if c not in self.group_col]
        num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(X[c])]
        time_cols = [self.group_time_on] if self.group_time_on and self.group_time_on in X.columns else []

        # Compute all per-group aggregations and select the top n_top_features
        # by variance (unsupervised).  Categorical aggregations are encoded as
        # integer category codes before computing variance.
        variance: dict[str, float] = {}
        feature_source: dict[str, tuple[str, str, bool]] = {}

        col_iter = [(c, True, list(self._NUM_AGGS)) for c in num_cols] + [
            (c, False, list(self._CAT_AGGS)) for c in cat_cols
        ]
        for col, is_num, aggs in tqdm(col_iter, desc="Computing groupby aggregations", unit="col"):
            slice_cols = list(dict.fromkeys(self.group_col + time_cols + [col]))
            X_col = self._sort_by_time(X[slice_cols])
            agg_df = X_col.groupby(self._build_group_key(X_col), observed=True)[[col]].agg(aggs)
            agg_df.columns = [f"{col}_{a}" for a in aggs]
            for feat in agg_df.columns:
                s = group_key.map(agg_df[feat])
                if not pd.api.types.is_numeric_dtype(s):
                    s = s.astype("category").cat.codes.astype(float)
                    s[s < 0] = np.nan
                variance[feat] = float(s.var())
            for agg in aggs:
                feature_source[f"{col}_{agg}"] = (col, agg, is_num)

        # Select top n_top_features by variance (descending), tie-break by name.
        ranked = sorted(variance.keys(), key=lambda f: (-variance[f], f))
        self._selected_features = ranked[: self.n_top_features]

        # Build test-time agg maps.
        num_map: dict[str, list[str]] = defaultdict(list)
        cat_map: dict[str, list[str]] = defaultdict(list)
        for feat in self._selected_features:
            src_col, agg_func, is_num = feature_source[feat]
            (num_map if is_num else cat_map)[src_col].append(agg_func)
        self._num_agg_map = dict(num_map)
        self._cat_agg_map = dict(cat_map)

        self._log(
            20,
            f"GroupAggregationFeatureGenerator: selected {len(self._selected_features)} "
            f"groupby features out of {len(variance)} generated.",
        )
        return self._transform(X), {GROUP_INDEX_FEATURES: self._selected_features}

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.generate_features:
            return X.drop(columns=self.group_col)

        # Only compute the aggregations needed for selected features.
        X_agg = self._compute_selected_agg_features(X)
        group_key = self._build_group_key(X)
        X = X.drop(columns=self.group_col)

        mapped = pd.DataFrame(
            {col: group_key.map(X_agg[col]) for col in self._selected_features},
            index=X.index,
        )
        return pd.concat([X, mapped], axis=1)

    def _sort_by_time(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return X with rows within each group sorted by ``group_time_on``.

        Uses groupby.apply so only the within-group order changes; ``last``
        then returns the most recent observation for each group.
        """
        if self.group_time_on is not None and self.group_time_on in X.columns:
            sorted_index = (
                X.groupby(self.group_col, sort=False, observed=True, group_keys=False)
                .apply(lambda g: g.sort_values(self.group_time_on), include_groups=False)
                .index
            )
            return X.loc[sorted_index]
        return X

    def _compute_selected_agg_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute only the aggregations needed for the selected features."""
        X = self._sort_by_time(X)
        group_key = self._build_group_key(X)
        parts: list[pd.DataFrame] = []

        if self._num_agg_map:
            for col, aggs in self._num_agg_map.items():
                if col not in X.columns:
                    continue
                agg_df = X.groupby(group_key, observed=True)[[col]].agg(aggs)
                agg_df.columns = [f"{col}_{a}" for a in aggs]
                parts.append(agg_df)

        if self._cat_agg_map:
            for col, aggs in self._cat_agg_map.items():
                if col not in X.columns:
                    continue
                agg_df = X.groupby(group_key, observed=True)[[col]].agg(aggs)
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
