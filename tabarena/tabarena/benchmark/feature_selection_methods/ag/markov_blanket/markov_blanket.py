"""Markov blanket feature selection."""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class MarkovBlanketFeatureSelector(AbstractFeatureSelector):
    """MarkovBlanket Feature Selection.

    Reference: Koller, Daphne, and Mehran Sahami. "Toward optimal feature selection." ICML. Vol. 96. No. 28. 1996.
    Implementation Source: Algorithm in the paper
    Changes to the implementation by Bastian Schäfer:
                           - Add time constraint
    """

    name = "MarkovBlanketFeatureSelector"
    feature_scoring_method: bool = False

    def _fit_feature_selection(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[
        str, float]:
        start_time = time.monotonic()
        use_fi_in_delta = True
        cv_splits = 5
        random_state = 0
        k = 5

        columns = X.columns
        X = SimpleImputer(strategy="mean").fit_transform(X)
        X = pd.DataFrame(X, columns=columns)

        G = list(X.columns)

        # Pearson correlation matrix (p_ij)
        corr = X.corr(method="pearson").abs()

        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

        # Simple, stable probabilistic classifier for log-loss
        # (needs predict_proba -> LogisticRegression provides it)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

        for _ in range(min(self.max_features, len(G) - 1)):
            elapsed_time = time.monotonic() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(f"Warning: FeatureSelection Method has no time left to train... "
                               f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)")
                break
            best_feat = None
            best_delta = np.inf
            for fi in G:
                elapsed_time = time.monotonic() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(f"Warning: FeatureSelection Method has no time left to train... "
                                   f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)")
                    break
                others = [fj for fj in G if fj != fi]
                if not others:
                    continue

                Mi = list(corr.loc[fi, others].sort_values(ascending=False).head(k).index)

                feat_set = Mi + ([fi] if use_fi_in_delta else [])
                # expected cross-entropy = mean log loss
                # sklearn returns NEGATIVE log loss, so negate it back.
                scores = cross_val_score(clf, X[feat_set], y, cv=cv, scoring="neg_log_loss", error_score="raise")
                delta = float(-scores.mean())

                if delta < best_delta:
                    best_delta = delta
                    best_feat = fi

            G.remove(best_feat)
        return [str(feat) for feat in G]
