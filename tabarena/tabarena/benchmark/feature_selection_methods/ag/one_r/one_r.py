"""OneR feature selection."""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class OneRFeatureSelector(AbstractFeatureSelector):
    """OneR Feature Selection.

    Reference: Holte, Robert C. "Very simple classification rules
    perform well on most commonly used datasets." Machine learning
    11.1 (1993): 63-90.
    Implementation by Bastian Schäfer
    Changes to the algorithm in the paper by Bastian Schäfer:
                           - Add time constraint
    """

    name = "OneRFeatureSelector"
    feature_scoring_method: bool = False

    def _fit_feature_selection(  # noqa: C901, PLR0912
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None,
    ) -> dict[str, float]:
        start_time = time.monotonic()
        hypotheses = {}
        class_labels = sorted(y.unique())
        n_class_labels = len(class_labels)

        X = X.fillna(X.mean(numeric_only=True))

        # 1. Create 3D array with class counts for each feature value and class label
        max_n_values = X.nunique().max()
        value_to_idx = {}
        count_cva = np.zeros((n_class_labels, max_n_values, len(X.columns)), dtype=int)
        for feature_index, feature_col in enumerate(X.columns):
            elapsed_time = time.monotonic() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            unique_values = sorted(X[feature_col].astype(str).unique())
            value_to_idx[feature_index] = {val: idx for idx, val in enumerate(unique_values)}
            for value_index, feature_val in enumerate(unique_values):
                elapsed_time = time.monotonic() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break
                mask = X[feature_col] == feature_val
                class_counts = y[mask].value_counts().reindex(class_labels, fill_value=0)
                for count_index, class_label in enumerate(class_labels):
                    elapsed_time = time.monotonic() - start_time
                    if (time_limit is not None) and (elapsed_time >= time_limit):
                        logger.warning(
                            f"Warning: FeatureSelection Method has no time left to train... "
                            f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                        )
                        break
                    count_cva[count_index, value_index, feature_index] = class_counts[class_label]

        # 2. Determine default class and its accuracy (default_class_accuracy is only used for 1R*)
        total_per_class = count_cva.sum(axis=(1, 2))
        default_class_idx = np.argmax(total_per_class)

        # 3. Transform numerical features into categorical features by binning them
        numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
        for col in numerical_cols:
            elapsed_time = time.monotonic() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            index = X.columns.get_loc(col)
            intervals = self.create_bins(X[col], count_cva, index, value_to_idx[index])
            bin_labels = [f"interval_{i}" for i in range(len(intervals))]
            X[col] = pd.Categorical(self.assign_interval_labels(X[col], intervals, bin_labels))

        # 4. Create hypotheses for each feature and select the best one
        class_to_label = dict(enumerate(class_labels))
        for feature_col in X.columns:
            elapsed_time = time.monotonic() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            values = sorted(X[feature_col].astype(str).dropna().unique())
            values = [str(v) for v in values]
            hypothesis = {}
            for value in values:
                elapsed_time = time.monotonic() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break
                mask = X[feature_col] == value
                if mask.sum() == 0:
                    continue
                class_counts = y[mask].value_counts().reindex(class_labels, fill_value=0)
                optimal_idx = np.argmax(class_counts)
                hypothesis[value] = class_to_label[optimal_idx]
            hypothesis["missing"] = class_to_label[default_class_idx]
            hypotheses[feature_col] = hypothesis

        # 5. Choose the rule with the highest accuracy on the training set (1R)
        accuracies = self.hypotheses_accuracy(X, y, hypotheses)
        if not accuracies:
            logger.warning("No valid hypotheses generated. Returning an empty selection.")
            return []
        best_feature = max(accuracies, key=accuracies.get)
        best_feature_idx = X.columns.get_loc(best_feature)

        selected_feature_list = [self._original_features[best_feature_idx]]
        return [str(feat) for feat in selected_feature_list]

    @staticmethod
    def class_value_counts(X, y, feature_col, class_labels):
        """Compute class value counts as a crosstab for a given feature column."""
        return pd.crosstab(X[feature_col], y).reindex(columns=class_labels, fill_value=0)

    @staticmethod
    def create_bins(X_col, count_cva, feature_idx, value_to_idx, small=3):
        """Create bins for a numerical feature column based on optimal class boundaries."""
        sorted_values = sorted(X_col.unique())

        def optimal_class(value):
            value_idx = value_to_idx[str(value)]
            return np.argmax(count_cva[:, value_idx, feature_idx])

        intervals = []
        current_interval = [sorted_values[0]]
        for _i, val in enumerate(sorted_values[1:], 1):
            # Constraint (b): if next value has different optimal class → close interval
            if optimal_class(val) != optimal_class(current_interval[0]):
                # Constraint (a): check dominant class covers > SMALL values
                dominant_class_count = sum(
                    1 for v in current_interval if optimal_class(v) == optimal_class(current_interval[0])
                )
                if dominant_class_count > small:
                    intervals.append((current_interval[0], current_interval[-1]))
                    current_interval = [val]
                else:
                    current_interval.append(val)  # merge
            else:
                current_interval.append(val)
        intervals.append((current_interval[0], current_interval[-1]))
        return intervals

    @staticmethod
    def assign_interval_labels(X_col, intervals, bin_labels):
        """Map values to interval labels directly."""
        result = X_col.astype("object").copy()
        n_intervals = len(intervals)
        for i, (low, high) in enumerate(intervals):
            mask = (X_col >= low) & (X_col < high) if i < n_intervals - 1 else X_col >= low
            result[mask] = bin_labels[i]
        return result

    @staticmethod
    def hypotheses_accuracy(X, y, hypotheses):
        """Vectorized accuracy for all hypotheses."""
        accuracies = {}
        for feature_col, hypothesis in hypotheses.items():
            # Vectorized prediction
            predicted = X[feature_col].astype(str).map(lambda v, h=hypothesis: h.get(v, h["missing"]))
            correct = (predicted == y).sum()
            accuracies[feature_col] = correct / len(X)
        return accuracies
