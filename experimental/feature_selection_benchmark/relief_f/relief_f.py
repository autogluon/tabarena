import logging
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class ReliefFFeatureSelector(AbstractFeatureSelector):
    """
    ReliefF Feature Selection.

    Reference: Kononenko, Igor, Edvard Šimec, and Marko Robnik-Šikonja. "Overcoming the myopia of inductive learning algorithms with RELIEFF." Applied Intelligence 7.1 (1997): 39-55.
    Implementation Source: https://github.com/jundongl/scikit-feature/blob/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/function/similarity_based/reliefF.py
                           The author of the code is Li, Jundong, Associate Professor at the University of Virginia and main-author of 'Feature selection: A data perspective' (2017).
    Changes to the implementation by Bastian Schäfer:
                           - Add time constraint
    """

    name = "ReliefFFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        k = 5
        columns = X.columns
        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        X = X_imputed.to_numpy()
        n_samples, n_features = X.shape

        distance = pairwise_distances(X, metric='manhattan')
        score = np.zeros(n_features)
        # the number of sampled instances is equal to the number of total instances
        for idx in range(n_samples):
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            near_hit = []
            near_miss = dict()

            self_fea = X[idx, :]
            c = np.unique(y).tolist()

            stop_dict = dict()
            for label in c:
                elapsed_time = time.time() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break
                stop_dict[label] = 0
            del c[c.index(y[idx])]

            p_dict = dict()
            p_label_idx = float(len(y[y == y[idx]])) / float(n_samples)

            for label in c:
                elapsed_time = time.time() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break
                p_label_c = float(len(y[y == label])) / float(n_samples)
                p_dict[label] = p_label_c / (1 - p_label_idx)
                near_miss[label] = []

            distance_sort = []
            distance[idx, idx] = np.max(distance[idx, :])

            for i in range(n_samples):
                elapsed_time = time.time() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break
                distance_sort.append([distance[idx, i], int(i), y[i]])
            distance_sort.sort(key=lambda x: x[0])

            for i in range(n_samples):
                elapsed_time = time.time() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break
                # find k nearest hit points
                if distance_sort[i][2] == y[idx]:
                    if len(near_hit) < k:
                        near_hit.append(distance_sort[i][1])
                    elif len(near_hit) == k:
                        stop_dict[y[idx]] = 1
                else:
                    # find k nearest miss points for each label
                    if len(near_miss[distance_sort[i][2]]) < k:
                        near_miss[distance_sort[i][2]].append(distance_sort[i][1])
                    else:
                        if len(near_miss[distance_sort[i][2]]) == k:
                            stop_dict[distance_sort[i][2]] = 1
                stop = True
                for (key, value) in stop_dict.items():
                    elapsed_time = time.time() - start_time
                    if (time_limit is not None) and (elapsed_time >= time_limit):
                        logger.warning(
                            f"Warning: FeatureSelection Method has no time left to train... "
                            f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                        )
                        break
                    if value != 1:
                        stop = False
                if stop:
                    break

                # update reliefF score
            near_hit_term = np.zeros(n_features)
            for ele in near_hit:
                elapsed_time = time.time() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break
                near_hit_term = np.array(abs(self_fea - X[ele, :])) + np.array(near_hit_term)

            near_miss_term = dict()
            for (label, miss_list) in near_miss.items():
                elapsed_time = time.time() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break
                near_miss_term[label] = np.zeros(n_features)
                for ele in miss_list:
                    elapsed_time = time.time() - start_time
                    if (time_limit is not None) and (elapsed_time >= time_limit):
                        logger.warning(
                            f"Warning: FeatureSelection Method has no time left to train... "
                            f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                        )
                        break
                    near_miss_term[label] = np.array(abs(self_fea - X[ele, :])) + np.array(near_miss_term[label])
                score += near_miss_term[label] / (k * p_dict[label])
            score -= near_hit_term / k

        feature_scores = dict(zip(columns, score))
        return feature_scores
