"""Laplacian score feature selection."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csc_matrix, diags
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, normalize

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class LaplacianScoreFeatureSelector(AbstractFeatureSelector):
    """LaplacianScore Feature Selection.

    Reference: He, Xiaofei, Deng Cai, and Partha Niyogi. "Laplacian score for feature selection." Advances in neural
    information processing systems 18 (2005).
    Implementation Source:
    https://github.com/jundongl/scikit-feature/blob/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/function/similarity_based/lap_score.py#L6
    The author of the code is Li, Jundong, Associate Professor at the
    University of Virginia and main-author of
    'Feature selection: A data perspective' (2017).
    Changes to the implementation by Bastian Schäfer:
                           - Add time constraint
                           - Remove overhead code for the construction of the weight matrix
                           - A sklearn preprocessing normalization is used instead of the code of the author, which
                           returned matrices filled with 0s for the datasets we used, which caused the laplacian score
                           to be 1 for all features.
    """

    name = "LaplacianScoreFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None,  # noqa: ARG002
    ) -> dict[str, float]:
        """This function implements the laplacian score feature selection, steps are as follows:
        1. Construct the affinity matrix W if it is not specified
        2. For the r-th feature, we define fr = X(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
        3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
        4. Laplacian score for the r-th feature is score = (fr_hat'*L*fr_hat)/(fr_hat'*D*fr_hat).
        """
        columns = X.columns
        X = X.to_numpy()
        data_encoder = OrdinalEncoder()
        X = data_encoder.fit_transform(X)
        numeric_imputer = SimpleImputer(strategy="mean")
        X = numeric_imputer.fit_transform(X)
        W = self.construct_W(X)
        D = np.array(W.sum(axis=1))
        L = W
        tmp = np.dot(np.transpose(D), X.astype(int))
        D = diags(np.transpose(D), [0])
        Xt = np.transpose(X)
        t1 = np.transpose(np.dot(Xt, D.todense()))
        t2 = np.transpose(np.dot(Xt, L.todense()))
        D_prime = np.sum(np.multiply(t1, X), 0) - np.multiply(tmp, tmp) / D.sum()
        L_prime = np.sum(np.multiply(t2, X), 0) - np.multiply(tmp, tmp) / D.sum()
        D_prime[D_prime < 1e-12] = 10000
        score = 1 - np.array(np.multiply(L_prime, 1 / D_prime))[0, :]
        return dict(zip(columns, score))

    @staticmethod
    def construct_W(X):
        """Construct the affinity matrix W using cosine similarity and knn neighbor mode."""
        n_samples, _n_features = np.shape(X)
        # set k = 5 for knn neighbor
        k = 5
        X = normalize(X, norm="l2", axis=1, copy=False)
        # compute pairwise cosine distances
        D_cosine = np.dot(X, np.transpose(X))
        # sort the distance matrix D in descending order
        idx = np.argsort(-D_cosine, axis=1)
        idx_new = idx[:, 0 : k + 1]
        G = np.zeros((n_samples * (k + 1), 3))
        G[:, 0] = np.tile(np.arange(n_samples), (k + 1, 1)).reshape(-1)
        G[:, 1] = np.ravel(idx_new, order="F")
        G[:, 2] = 1
        # build the sparse affinity matrix W
        W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
        bigger = np.transpose(W) > W
        return W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
