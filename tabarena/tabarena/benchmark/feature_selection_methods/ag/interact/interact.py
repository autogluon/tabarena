"""INTERACT feature selection."""
from __future__ import annotations

import logging
import time
from math import log
from typing import TYPE_CHECKING

import numpy as np

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class INTERACTFeatureSelector(AbstractFeatureSelector):
    """INTERACT Feature Selection.

    Reference: Zhao, Zheng, and Huan Liu. "Searching for interacting
    features in subset selection." Intelligent Data Analysis 13.2
    (2009): 207-228.
    Implementation Inspiration: Information Gain code from
    https://github.com/Thijsvanede/info_gain/blob/
    master/info_gain/info_gain.py
    & Entropy code from
    https://github.com/jundongl/scikit-feature/blob/
    48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/
    function/information_theoretical_based.
    The author of the Information Gain code is Thijs van Ede, Associate
    Professor at the University of Twente and main-author of
    'FlowPrint: Semi-Supervised Mobile-App Fingerprinting on Encrypted
    Network Traffic' (2020), where they used information gain
    The author of the Entropy code is Li, Jundong, Associate Professor
    at the University of Virginia and main-author of
    'Feature selection: A data perspective' (2017).
    Changes to the implementation by Bastian Schäfer:
        - Adapt Information Gain and Symmetrical Uncertainty Code to
          INTERACT algorithm presented in the paper
        - Add time constraint
    """

    name = "INTERACTFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        delta = 1e-4
        # 1) Compute SU for each feature
        su_scores = self.symmetrical_uncertainty(X, y, time_limit, start_time)

        # 2) Rank features by SU descending
        slist = list(np.argsort(su_scores)[::-1])
        # Check for elapsed time in case the SU calculation took
        # too long, in that case return current ranking and
        # terminate FS method
        elapsed_time = time.monotonic() - start_time
        if (time_limit is not None) and (elapsed_time >= time_limit):
            logger.warning(
                f"Warning: FeatureSelection Method has no time left to train... "
                f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
            )
            return dict(zip(X.columns, np.array(slist, dtype=int)))

        # 3) Backward elimination using c-contribution (inconsistency rate)
        counter = len(slist) - 1
        while counter >= 0 and len(slist) > 1:
            elapsed_time = time.monotonic() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break

            f_idx = slist[counter]
            cc = self._c_contribution(
                X, y, slist, f_idx, time_limit, start_time,
            )  # CC(F, Slist)
            # self._c_contribution returns None if time limit is
            # reached, in that case return current ranking and
            # terminate FS method
            if cc is None:
                return dict(zip(X.columns, np.array(slist, dtype=int)))
            if cc <= delta:
                slist.pop(counter)  # remove feature
            counter -= 1

        return dict(zip(X.columns, np.array(slist, dtype=int)))

    def symmetrical_uncertainty(self, X: pd.DataFrame, y: pd.Series, time_limit, start_time) -> np.ndarray:
        """Symmetrical Uncertainty for each feature:
          SU(X, Y) = 2 * IG(Y|X) / (H(X) + H(Y))  (information gain / mutual information)
        where:
          IG(Y|X) = H(Y) - H(Y|X) .

        Returns:
        -------
        np.ndarray of shape (n_features,)
        """
        _n_samples, n_features = X.shape
        SU = np.zeros(n_features, dtype=float)

        H_y = self.entropyd(y.value_counts(dropna=False).to_numpy())

        for i in range(n_features):
            elapsed_time = time.monotonic() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            f = X.iloc[:, i]
            H_x = self.entropyd(f.value_counts(dropna=False).to_numpy())

            # H(Y|X) = sum_v p(v) * H(Y | X=v)
            p_v = f.value_counts(normalize=True, dropna=False)
            H_y_given_x = 0.0
            for v, p in p_v.items():
                elapsed_time = time.monotonic() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break

                y_sub = y[f.eq(v)]
                H_y_given_x += p * self.entropyd(y_sub.value_counts(dropna=False).to_numpy())

            IG = H_y - H_y_given_x  # IG(Y|X) = H(Y) - H(Y|X)

            MI = H_x + H_y  # MI = (H(X) + H(Y)
            SU[i] = (2.0 * IG / MI) if MI > 0 else 0.0

        return np.abs(SU)

    def _c_contribution(
        self, X: pd.DataFrame, y: pd.Series, feature_set: list[int], f_idx: int, time_limit, start_time
    ) -> float:
        r"""CC(Fi, S) = ICR(S \\ {Fi}) - ICR(S)."""
        icr_full = self._inconsistency_rate(X.iloc[:, feature_set], y, time_limit, start_time)
        reduced = [j for j in feature_set if j != f_idx]
        icr_reduced = self._inconsistency_rate(X.iloc[:, reduced], y, time_limit, start_time)
        if icr_reduced is None or icr_full is None:
            return None
        return icr_reduced - icr_full  # non-negative by monotonicity in INTERACT

    @staticmethod
    def _inconsistency_rate(X_sub: pd.DataFrame, y: pd.Series, time_limit, start_time) -> float:
        """Inconsistency rate of a feature subset projection π_S(D).

        For each distinct feature pattern, count labels; inconsistency count for that pattern is:
            group_size - max_class_count_in_group
        ICR = total_inconsistency / n_samples
        """
        df = X_sub.copy()
        df["_y_"] = y.to_numpy()

        # Group by feature pattern; within each pattern, count label frequencies
        incons = 0
        for _, grp in df.groupby(list(X_sub.columns), dropna=False, observed=False):
            elapsed_time = time.monotonic() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                return None
            counts = grp["_y_"].value_counts(dropna=False)
            incons += len(grp) - int(counts.max())

        return incons / len(df)

    def entropyd(self, sx, base=2):
        """Discrete entropy estimator given a list of samples which can be any hashable object."""
        return self.entropyfromprobs(self.hist(sx), base=base)

    @staticmethod
    def hist(sx):
        """Compute histogram (probability distribution) from a list of samples."""
        d = dict()
        for s in sx:
            d[s] = d.get(s, 0) + 1
        return (float(z) / len(sx) for z in d.values())

    def entropyfromprobs(self, probs, base=2):
        """Compute entropy from a probability distribution."""
        return -sum(map(self.elog, probs)) / log(base)

    @staticmethod
    def elog(x):
        """Compute x*log(x), returning 0 for x <= 0 or x >= 1."""
        if x <= 0.0 or x >= 1.0:
            return 0
        return x * log(x)
