"""OneR feature selection."""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

logger = logging.getLogger(__name__)

MISSING_SENTINEL = "__missing__"

def create_bins(X_col, y, min_bucket_size=6):
    """"Creates bins with minimum number of samples per majority class per feature. Returns feature values that define the bins."""
    # Sort instances by feature value
    sorted_idx = X_col.argsort()
    sorted_vals = X_col.iloc[sorted_idx].values
    sorted_labels = y.iloc[sorted_idx].values

    breakpoints = []
    i = 0
    n = len(sorted_vals)

    while i < n:
        class_counts = {}
        j = i

        while j < n:
            label = sorted_labels[j]
            class_counts[label] = class_counts.get(label, 0) + 1
            j += 1

            majority_count = max(class_counts.values())
            at_boundary = (j == n) or (sorted_vals[j] != sorted_vals[j - 1]) # whether we're in the next value range/category

            if majority_count >= min_bucket_size and at_boundary:
                break

        if j < n:
            breakpoints.append(sorted_vals[j])
        i = j

    return breakpoints # interval breakpoints
    

def merge_bins(breakpoints, sorted_vals, sorted_labels):
    """Merges adjacent bins sharing the same majority class."""
    # Determine majority class per bucket
    all_points = [sorted_vals[0]] + breakpoints + [np.inf]
    predictions = []
    for i in range(len(all_points) - 1):
        mask = (sorted_vals >= all_points[i]) & (sorted_vals < all_points[i + 1])
        labels_in_bucket = sorted_labels[mask]
        unique, counts = np.unique(labels_in_bucket, return_counts=True)
        predictions.append(unique[np.argmax(counts)])

    merged_breakpoints = []
    for i, bp in enumerate(breakpoints):
        if predictions[i] != predictions[i + 1]:
            merged_breakpoints.append(bp)

    return merged_breakpoints


def bin_column(X_col: pd.Series, breakpoints: list[float]) -> pd.Series:
    """Map a numeric column to interval-label strings using the given breakpoints."""
    boundaries = [-np.inf] + breakpoints + [np.inf]
    labels = [f"interval_{i}" for i in range(len(boundaries) - 1)]

    result = X_col.copy().astype(object)
    for i, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        mask = (X_col >= lo) & (X_col < hi)
        result[mask] = labels[i]
    return result


def build_hypothesis(X_col, y, class_labels, default_class):
    """Create mappings from feature value to the majority class for a given feature."""
    hypothesis: dict = {}
    for value in X_col.astype(str).unique():
        mask = X_col.astype(str) == value
        if mask.sum() == 0:
            continue
        counts = y[mask].value_counts().reindex(class_labels, fill_value=0)
        hypothesis[value] = class_labels[int(np.argmax(counts))]
 
    hypothesis.setdefault(MISSING_SENTINEL, default_class) # fallback for values unseen at fit time (including missing if absent)
    return hypothesis


def hypotheses_error_rate(X: pd.DataFrame, y: pd.Series, hypotheses: dict[str, dict]) -> dict[str, float]:
    """Vectorised per-feature rule error rate on the training set."""
    error_rates: dict[str, float] = {}
    for feature_col, hypothesis in hypotheses.items():
        predicted = X[feature_col].astype(str).map(
            lambda v, h=hypothesis: h.get(v, h[MISSING_SENTINEL])
        )
        error_rates[feature_col] = (predicted != y).sum() / len(y)
    return error_rates


def _time_exceeded(start_time, time_limit):
    return time_limit is not None and (time.monotonic() - start_time) >= time_limit


class OneRFeatureSelector(AbstractFeatureSelector):
    """OneR Feature Selection.

    Reference: Holte, Robert C. "Very simple classification rules perform well on most commonly used datasets." 
    Machine learning 11.1 (1993): 63-90.

    Parameters
    ----------
    min_bucket_size:
        Minimum number of same-class instances needed to close a numeric bucket (default=6).
    """
 
    name = "OneRFeatureSelector"
    feature_scoring_method: bool = False
 
    def __init__(self, min_bucket_size: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.min_bucket_size = min_bucket_size

    
    def _fit_feature_selection(  # noqa: C901, PLR0912
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        
        start_time = time.monotonic()
        class_labels = sorted(y.unique())
        
        X = X.copy() # avoid ag problems during preprocessing

        # 1. Replace NaNs with a sentinel category.
        for col in X.columns:
            if hasattr(X[col], 'cat'):  # is Categorical
                X[col] = X[col].cat.add_categories(MISSING_SENTINEL)
            X[col] = X[col].fillna(MISSING_SENTINEL)

        # 2. Determine default class as a fallback.
        default_class = y.value_counts().idxmax()

        # 3. Discretize numerical features
        numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

        for col in numerical_cols:
            if _time_exceeded(start_time, time_limit):
                logger.warning("Time limit reached during binning.")
                break

            X[col] = X[col].astype(object)
            mask = X[col] != MISSING_SENTINEL  # consider non-missing rows only
            X_num = X.loc[mask, col].astype(float)
            y_num = y.loc[mask]
 
            breakpoints = create_bins(X_num, y_num, self.min_bucket_size)

            order = X_num.argsort()
            breakpoints = merge_bins(
                breakpoints,
                X_num.iloc[order].values,
                y_num.iloc[order].values,
            )

            # replace numeric values with interval labels; sentinel rows stay
            X[col] = X[col].astype(object)
            X.loc[mask, col] = bin_column(X_num, breakpoints)
            X[col] = pd.Categorical(X[col].astype(str))

        # 4. Create hypotheses for each feature and select the best one
        hypotheses: dict[str, dict] = {}
 
        for feature_col in X.columns:
            if _time_exceeded(start_time, time_limit):
                logger.warning("Time limit reached during hypothesis building.")
                break
 
            hypotheses[feature_col] = build_hypothesis(X[feature_col], y, class_labels, default_class)

        # 5. Choose the max_features features with the lowest error on the training set (1R)
        error_rates = hypotheses_error_rate(X, y, hypotheses)

        if not error_rates:
            logger.warning("No valid hypotheses generated. Returning empty selection.")
            return []

        sorted_features = sorted(error_rates, key=error_rates.__getitem__)
        return [str(f) for f in sorted_features[: self.max_features]]