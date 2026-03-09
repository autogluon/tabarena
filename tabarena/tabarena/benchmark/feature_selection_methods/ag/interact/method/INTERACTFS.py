from __future__ import annotations

import time

import numpy as np
import pandas as pd
import warnings
import logging

from scipy.stats import entropy

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class INTERACTFS:
    """INTERACT feature selector"""

    def __init__(self, model=None):
        self._y = None
        self._model = model
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model=None, n_max_features: int | None = None, **kwargs,) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features

        scores = self.interact(X, y, n_max_features, **kwargs)
        feature_ranking = self.feature_ranking(scores)

        if n_max_features is None or n_max_features >= X.shape[1]:
            selected_features = X.columns[feature_ranking]
        else:
            selected_features = X.columns[feature_ranking[:n_max_features]]

        X_selected = X.loc[:, selected_features]
        self._selected_features = list(X_selected.columns)
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            # fall back to stored params
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X.loc[:, self._selected_features]

    def interact(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            n_max_features: int | None,
            delta: float = 1e-4,
            **kwargs
    ) -> np.ndarray:
        """
        INTERACT (Yu & Liu): rank by SU, then backward eliminate using c-contribution.
        - SU(Xi, Y) = 2 * IG(Y|Xi) / (H(Xi)+H(Y))
        - CC(Fi, S) = ICR(S \\ {Fi}) - ICR(S)
        Remove Fi if CC(Fi, S) <= delta


        Returns
        -------
        selected_idx : np.ndarray of selected feature indices (ordered by SU desc)
        """

        # 1) Compute SU for each feature (your SU code, slightly factored)
        su_scores = self.symmetrical_uncertainty(X, y, n_max_features=None, **kwargs)

        # 2) Rank features by SU descending
        slist = list(np.argsort(su_scores)[::-1])

        # 3) Backward elimination using c-contribution (consistency contribution)
        counter = len(slist) - 1
        while counter >= 0 and len(slist) > 1:
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

            f_idx = slist[counter]
            cc = self._c_contribution(X, y, slist, f_idx)  # CC(F, Slist)
            if cc is None:
                score = np.zeros(X.shape[1])
                if n_max_features is not None and X.shape[1] > n_max_features:
                    selected_idx = np.random.choice(X.shape[1], size=n_max_features, replace=False)
                else:
                    selected_idx = np.arange(X.shape[1])
                score[selected_idx] = 1
                return score
            if cc <= delta:
                slist.pop(counter)  # remove feature
            counter -= 1

        # 4) Cut to n_max_features (INTERACT itself returns Sbest; you can cap it)
        if n_max_features is not None and len(slist) > n_max_features:
            slist = slist[:n_max_features]

        return np.array(slist, dtype=int)

    def symmetrical_uncertainty(self, X: pd.DataFrame, y: pd.Series, n_max_features, **kwargs
    ) -> np.ndarray:
        """
        Symmetrical Uncertainty for each feature:
          SU(X, Y) = 2 * IG(Y|X) / (H(X) + H(Y))
        where:
          IG(Y|X) = H(Y) - H(Y|X)  (information gain / mutual information).

        Returns
        -------
        np.ndarray of shape (n_features,)
        """
        n_samples, n_features = X.shape
        F = np.zeros(n_features, dtype=float)

        # H(Y)
        H_y = self._entropy_from_counts(y.value_counts(dropna=False))

        for i in range(n_features):
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

            f = X.iloc[:, i]
            # H(X)
            H_x = self._entropy_from_counts(f.value_counts(dropna=False))
            # H(Y|X) = sum_v p(v) * H(Y | X=v)
            p_v = f.value_counts(normalize=True, dropna=False)
            H_y_given_x = 0.0
            for v, p in p_v.items():
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

                y_sub = y[f.eq(v)]
                H_y_given_x += p * self._entropy_from_counts(y_sub.value_counts(dropna=False))

            IG = H_y - H_y_given_x  # IG(Y|X) = H(Y) - H(Y|X)

            denom = H_x + H_y
            F[i] = (2.0 * IG / denom) if denom > 0 else 0.0  # SU definition

        return np.abs(F)

    def _c_contribution(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            feature_set: list[int],
            f_idx: int,
    ) -> float:
        """
        CC(Fi, S) = ICR(S \\ {Fi}) - ICR(S)
        """
        icr_full = self._inconsistency_rate(X.iloc[:, feature_set], y)
        reduced = [j for j in feature_set if j != f_idx]
        icr_reduced = self._inconsistency_rate(X.iloc[:, reduced], y)
        if icr_reduced is None or icr_full is None:
            return None
        return icr_reduced - icr_full  # non-negative by monotonicity in INTERACT

    def _inconsistency_rate(self, X_sub: pd.DataFrame, y: pd.Series, **kwargs) -> float:
        """
        Inconsistency rate of a feature subset projection π_S(D) used by INTERACT.

        For each distinct feature pattern, count labels; inconsistency count for that pattern is:
            group_size - max_class_count_in_group
        ICR = total_inconsistency / n_samples
        """
        df = X_sub.copy()
        df["_y_"] = y.values

        # Group by feature pattern; within each pattern, count label frequencies
        incons = 0
        for _, grp in df.groupby(list(X_sub.columns), dropna=False):
            if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                time_start_fit = time.time()
                kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                kwargs["start_time"] = time_start_fit
                if kwargs["time_limit"] <= 0:
                    logger.warning(
                        f"\tWarning: FeatureSelection Method has no time left to train... "
                        f"(Time Left = {kwargs['time_limit']:.1f}s)"
                    )
                    return None
            counts = grp["_y_"].value_counts(dropna=False)
            incons += len(grp) - int(counts.max())

        return incons / len(df)

    @staticmethod
    def _entropy_from_counts(counts: pd.Series) -> float:
        """
        Shannon entropy (bits) from counts.
        scipy.stats.entropy accepts (possibly unnormalized) event counts.
        """
        return float(entropy(counts.to_numpy(), base=2))  # base=2 => bits
