from __future__ import annotations

import time

import numpy as np
import pandas as pd

import warnings
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class MarkovBlanketFS:
    """MarkovBlanket feature selector"""

    def __init__(self, model):
        self._y = None
        self._model = model
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model, n_max_features, **kwargs) -> pd.DataFrame:
        n_eliminate = max(0, X.shape[1] - n_max_features)
        kept = self.mb_expected_cross_entropy(X, y, n_max_features, k=5, n_eliminate=n_eliminate, use_fi_in_delta=True, **kwargs)
        X_selected = X[kept]
        self._selected_features = list(X_selected.columns)
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]

    def mb_expected_cross_entropy(self, X: pd.DataFrame, y: pd.Series, n_max_features, k: int, n_eliminate: int,
                                   use_fi_in_delta: bool = True, cv_splits: int = 5, random_state: int = 0, **kwargs) -> list[
        str]:
        """
        Markov-blanket-style elimination.

        p_ij computed from Pearson correlation via X.corr() [web:131].
        For each Fi in current G:
          Mi = top-k abs-correlated features among G \ {Fi}
          delta(Fi | Mi) = expected cross-entropy (log loss) estimated by CV
                           on features Mi (+ Fi if use_fi_in_delta=True)

        Remove Fi with minimal delta, repeat n_eliminate times.
        Returns remaining feature names (G).
        """

        G = list(X.columns)

        # Pearson correlation matrix (p_ij) [web:131]
        corr = X.corr(method="pearson").abs()

        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

        # Simple, stable probabilistic classifier for log-loss
        # (needs predict_proba -> LogisticRegression provides it)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000)
        )

        for _ in range(min(n_eliminate, len(G) - 1)):
            if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                time_start_fit = time.time()
                kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                kwargs["start_time"] = time_start_fit
                if kwargs["time_limit"] <= 0:
                    logger.warning(
                        f"\tWarning: FeatureSelection Method has no time left to train... "
                        f"(Time Left = {kwargs['time_limit']:.1f}s)"
                    )
                    score = np.zeros(X.shape[1])
                    if n_max_features is not None and X.shape[1] > n_max_features:
                        selected_idx = np.random.choice(X.shape[1], size=n_max_features, replace=False)
                    else:
                        selected_idx = np.arange(X.shape[1])
                    score[selected_idx] = 1
                    return score
            best_feat = None
            best_delta = np.inf

            for fi in G:
                if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                    time_start_fit = time.time()
                    kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                    kwargs["start_time"] = time_start_fit
                    if kwargs["time_limit"] <= 0:
                        logger.warning(
                            f"\tWarning: FeatureSelection Method has no time left to train... "
                            f"(Time Left = {kwargs['time_limit']:.1f}s)"
                        )
                        score = np.zeros(X.shape[1])
                        if n_max_features is not None and X.shape[1] > n_max_features:
                            selected_idx = np.random.choice(X.shape[1], size=n_max_features, replace=False)
                        else:
                            selected_idx = np.arange(X.shape[1])
                        score[selected_idx] = 1
                        return score
                others = [fj for fj in G if fj != fi]
                if not others:
                    continue

                Mi = list(corr.loc[fi, others].sort_values(ascending=False).head(k).index)

                feat_set = Mi + ([fi] if use_fi_in_delta else [])
                # expected cross-entropy = mean log loss
                # sklearn returns NEGATIVE log loss, so negate it back.
                scores = cross_val_score(
                    clf,
                    X[feat_set],
                    y,
                    cv=cv,
                    scoring="neg_log_loss",
                    error_score="raise",
                )
                delta = float(-scores.mean())

                if delta < best_delta:
                    best_delta = delta
                    best_feat = fi

            G.remove(best_feat)

        return G

    def feature_ranking(self, F):
        """
        Rank features in descending order according to t-score, the higher the t-score, the more important the feature is
        """
        idx = np.argsort(F)
        return idx[::-1]
