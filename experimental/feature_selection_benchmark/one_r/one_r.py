import logging
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class OneRFeatureSelector(AbstractFeatureSelector):
    """
    OneR Feature Selection.

    Reference: Holte, Robert C. "Very simple classification rules perform well on most commonly used datasets." Machine learning 11.1 (1993): 63-90.
    Implementation Source: https://github.com/rasbt/mlxtend/blob/366f717b87f2fcaeeb67b70432b6e1a801519eff/docs/sources/user_guide/classifier/OneRClassifier.ipynb#L4
                        The author of the code is Sebastian Raschka, AI Research Engineer and former Assistant Professor at the University of Wisconsin-Madison and main-author of 'MLxtend: Providing machine learning and data science utilities and extensions to Python’s scientific computing stack.' (2018).
    Changes to the implementation by Bastian Schäfer:
                           - Add time constraint
                           - Remove OneRClassifier class and adapt code to fit into the AbstractFeatureSelector class
                           -
    """

    name = "OneRFeatureSelector"
    feature_scoring_method: bool = False

    def _fit_feature_selection(self, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        X = X.to_numpy()
        self.resolve_ties = "chi-squared"
        for c in range(X.shape[1]):
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            if np.unique(X[:, c]).shape[0] == X.shape[0]:
                warnings.warn(
                    "Feature array likely contains at least one"
                    " non-categorical column."
                    " Column %d appears to have a unique value"
                    " in every row." % c
                )
            break

        n_class_labels = np.unique(y).shape[0]

        def compute_class_counts(X, y, feature_index, feature_value):
            mask = X[:, feature_index] == feature_value
            return np.bincount(y[mask], minlength=n_class_labels)

        prediction_dict = {}  # save feature_idx: feature_val, label, error

        # iterate over features
        for feature_index in np.arange(X.shape[1]):
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            # iterate over each possible value per feature
            for feature_value in np.unique(X[:, feature_index]):
                elapsed_time = time.time() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break
                class_counts = compute_class_counts(X, y, feature_index, feature_value)
                most_frequent_class = np.argmax(class_counts)
                self.class_labels_ = np.unique(y)

                # count all classes for that feature match
                # except the most frequent one
                inverse_index = np.ones(n_class_labels, dtype=bool)
                inverse_index[most_frequent_class] = False

                error = np.sum(class_counts[inverse_index])

                # compute the total error for each feature and
                #  save all the corresponding rules for a given feature
                if feature_index not in prediction_dict:
                    prediction_dict[feature_index] = {
                        "total error": 0,
                        "rules (value: class)": {},
                    }
                prediction_dict[feature_index]["rules (value: class)"][
                    feature_value
                ] = most_frequent_class
                prediction_dict[feature_index]["total error"] += error

            # get best feature (i.e., the feature with the lowest error)
            best_err = np.inf
            best_idx = [None]
            for i in prediction_dict:
                elapsed_time = time.time() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break
                if prediction_dict[i]["total error"] < best_err:
                    best_err = prediction_dict[i]["total error"]
                    best_idx[-1] = i

            if self.resolve_ties == "chi-squared":
                # collect duplicates
                for i in prediction_dict:
                    if i == best_idx[-1]:
                        continue
                    if prediction_dict[i]["total error"] == best_err:
                        best_idx.append(i)

                p_values = []
                for feature_idx in best_idx:
                    elapsed_time = time.time() - start_time
                    if (time_limit is not None) and (elapsed_time >= time_limit):
                        logger.warning(
                            f"Warning: FeatureSelection Method has no time left to train... "
                            f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                        )
                        break
                    rules = prediction_dict[feature_idx]["rules (value: class)"]

                    ary = np.zeros((n_class_labels, len(rules)))

                    for idx, r in enumerate(rules):
                        elapsed_time = time.time() - start_time
                        if (time_limit is not None) and (elapsed_time >= time_limit):
                            logger.warning(
                                f"Warning: FeatureSelection Method has no time left to train... "
                                f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                            )
                            break
                        ary[:, idx] = np.bincount(
                            y[X[:, feature_idx] == r], minlength=n_class_labels
                        )

                    # returns "stat, p, dof, expected"
                    _, p, _, _ = chi2_contingency(ary)
                p_values.append(p)
                best_p_idx = np.argmax(p_values)
                best_idx = best_idx[best_p_idx]
                self.p_value_ = p_values[best_p_idx]

            elif self.resolve_ties == "first":
                best_idx = best_idx[0]

        self.feature_idx_ = best_idx
        self.prediction_dict_ = prediction_dict[best_idx]

        selected_feature_list = [self._original_features[int(i)] for i in [self.feature_idx_]]
        return [str(feat) for feat in selected_feature_list]

