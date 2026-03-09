from __future__ import annotations

import logging
import sys
import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import *
from sklearn.metrics.pairwise import pairwise_distances

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class LaplacianScoreFS:
    """LaplacianScore feature selector."""

    def __init__(self, model):
        self._y = None
        self._model = model
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model, n_max_features, **kwargs) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features
        X_np = X.to_numpy()

        feature_ranking = self.feature_ranking(self.lap_score(X_np, n_max_features))

        selected_features_idx = feature_ranking[:n_max_features]
        selected_features = X.columns[selected_features_idx]

        X_selected = X[selected_features]
        self._selected_features = list(X_selected.columns)
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]

    def lap_score(self, X, n_max_features, **kwargs):
        """This function implements the laplacian score feature selection, steps are as follows:
        1. Construct the affinity matrix W if it is not specified
        2. For the r-th feature, we define fr = X(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
        3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
        4. Laplacian score for the r-th feature is score = (fr_hat'*L*fr_hat)/(fr_hat'*D*fr_hat).

        Input
        -----
        X: {numpy array}, shape (n_samples, n_features)
            input data
        kwargs: {dictionary}
            W: {sparse matrix}, shape (n_samples, n_samples)
                input affinity matrix

        Output
        ------
        score: {numpy array}, shape (n_features,)
            laplacian score for each feature

        Reference
        ---------
        He, Xiaofei et al. "Laplacian Score for Feature Selection." NIPS 2005.
        """
        # if 'W' is not specified, use the default W
        W = self.construct_W(X)
        if W is None:
            score = np.zeros(X.shape[1])
            if n_max_features is not None and X.shape[1] > n_max_features:
                selected_idx = np.random.choice(X.shape[1], size=n_max_features, replace=False)
            else:
                selected_idx = np.arange(X.shape[1])
            score[selected_idx] = 1
            return score
        D = np.array(W.sum(axis=1))
        L = W
        tmp = np.dot(np.transpose(D), X)
        D = diags(np.transpose(D), [0])
        Xt = np.transpose(X)
        t1 = np.transpose(np.dot(Xt, D.todense()))
        t2 = np.transpose(np.dot(Xt, L.todense()))
        D_prime = np.sum(np.multiply(t1, X), 0) - np.multiply(tmp, tmp) / D.sum()
        L_prime = np.sum(np.multiply(t2, X), 0) - np.multiply(tmp, tmp) / D.sum()
        D_prime[D_prime < 1e-12] = 10000
        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
            time_start_fit = time.time()
            kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
            kwargs["start_time"] = time_start_fit
            if kwargs["time_limit"] <= 0:
                logger.warning(
                    f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                )
                score = np.zeros(X.shape[1])
                if n_max_features is not None and X.shape[1] > n_max_features:
                    selected_idx = np.random.choice(X.shape[1], size=n_max_features, replace=False)
                else:
                    selected_idx = np.arange(X.shape[1])
                score[selected_idx] = 1
                return score
        score = 1 - np.array(np.multiply(L_prime, 1 / D_prime))[0, :]
        return np.transpose(score)

    def feature_ranking(self, score):
        """Rank features in ascending order according to their laplacian scores, the smaller the laplacian score is, the more
        important the feature is.
        """
        return np.argsort(score, 0)

    def construct_W(self, X, **kwargs):
        r"""Construct the affinity matrix W through different ways.

        Notes:
        -----
        if kwargs is null, use the default parameter settings;
        if kwargs is not null, construct the affinity matrix according to parameters in kwargs

        Input
        -----
        X: {numpy array}, shape (n_samples, n_features)
            input data
        kwargs: {dictionary}
            parameters to construct different affinity matrix W:
            y: {numpy array}, shape (n_samples, 1)
                the true label information needed under the 'supervised' neighbor mode
            metric: {string}
                choices for different distance measures
                'euclidean' - use euclidean distance
                'cosine' - use cosine distance (default)
            neighbor_mode: {string}
                indicates how to construct the graph
                'knn' - put an edge between two nodes if and only if they are among the
                        k nearest neighbors of each other (default)
                'supervised' - put an edge between two nodes if they belong to same class
                        and they are among the k nearest neighbors of each other
            weight_mode: {string}
                indicates how to assign weights for each edge in the graph
                'binary' - 0-1 weighting, every edge receives weight of 1 (default)
                'heat_kernel' - if nodes i and j are connected, put weight W_ij = exp(-norm(x_i - x_j)/2t^2)
                                this weight mode can only be used under 'euclidean' metric and you are required
                                to provide the parameter t
                'cosine' - if nodes i and j are connected, put weight cosine(x_i,x_j).
                            this weight mode can only be used under 'cosine' metric
            k: {int}
                choices for the number of neighbors (default k = 5)
            t: {float}
                parameter for the 'heat_kernel' weight_mode
            fisher_score: {boolean}
                indicates whether to build the affinity matrix in a fisher score way, in which W_ij = 1/n_l if yi = yj = l;
                otherwise W_ij = 0 (default fisher_score = false)
            reliefF: {boolean}
                indicates whether to build the affinity matrix in a reliefF way, NH(x) and NM(x,y) denotes a set of
                k nearest points to x with the same class as x, and a different class (the class y), respectively.
                W_ij = 1 if i = j; W_ij = 1/k if x_j \in NH(x_i); W_ij = -1/(c-1)k if x_j \in NM(x_i, y) (default reliefF = false)

        Output
        ------
        W: {sparse matrix}, shape (n_samples, n_samples)
            output affinity matrix W
        """
        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
            time_start_fit = time.time()
            kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
            kwargs["start_time"] = time_start_fit
            if kwargs["time_limit"] <= 0:
                logger.warning(
                    f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                )
                return None
        # default metric is 'cosine'
        if "metric" not in kwargs:
            kwargs["metric"] = "cosine"

        # default neighbor mode is 'knn' and default neighbor size is 5
        if "neighbor_mode" not in kwargs:
            kwargs["neighbor_mode"] = "knn"
        if kwargs["neighbor_mode"] == "knn" and "k" not in kwargs:
            kwargs["k"] = 5
        if kwargs["neighbor_mode"] == "supervised" and "k" not in kwargs:
            kwargs["k"] = 5
        if kwargs["neighbor_mode"] == "supervised" and "y" not in kwargs:
            print("Warning: label is required in the supervised neighborMode!!!")
            sys.exit(0)

        # default weight mode is 'binary', default t in heat kernel mode is 1
        if "weight_mode" not in kwargs:
            kwargs["weight_mode"] = "binary"
        if kwargs["weight_mode"] == "heat_kernel":
            if kwargs["metric"] != "euclidean":
                kwargs["metric"] = "euclidean"
            if "t" not in kwargs:
                kwargs["t"] = 1
        elif kwargs["weight_mode"] == "cosine":
            if kwargs["metric"] != "cosine":
                kwargs["metric"] = "cosine"

        # default fisher_score and reliefF mode are 'false'
        if "fisher_score" not in kwargs:
            kwargs["fisher_score"] = False
        if "reliefF" not in kwargs:
            kwargs["reliefF"] = False

        n_samples, _n_features = np.shape(X)

        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
            time_start_fit = time.time()
            kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
            kwargs["start_time"] = time_start_fit
            if kwargs["time_limit"] <= 0:
                logger.warning(
                    f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                )
                return None

        # choose 'knn' neighbor mode
        if kwargs["neighbor_mode"] == "knn":
            k = kwargs["k"]
            if kwargs["weight_mode"] == "binary":
                if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                    time_start_fit = time.time()
                    kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                    kwargs["start_time"] = time_start_fit
                    if kwargs["time_limit"] <= 0:
                        logger.warning(
                            f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                        )
                        return None
                if kwargs["metric"] == "euclidean":
                    # compute pairwise euclidean distances
                    D = pairwise_distances(X)
                    D **= 2
                    # sort the distance matrix D in ascending order
                    dump = np.sort(D, axis=1)
                    idx = np.argsort(D, axis=1)
                    # choose the k-nearest neighbors for each instance
                    idx_new = idx[:, 0 : k + 1]
                    G = np.zeros((n_samples * (k + 1), 3))
                    G[:, 0] = np.tile(np.arange(n_samples), (k + 1, 1)).reshape(-1)
                    G[:, 1] = np.ravel(idx_new, order="F")
                    G[:, 2] = 1
                    # build the sparse affinity matrix W
                    W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                    bigger = np.transpose(W) > W
                    return W - W.multiply(bigger) + np.transpose(W).multiply(bigger)

                # THIS IS THE ONLY CODE WE USE USUALLY

                if kwargs["metric"] == "cosine":
                    # normalize the data first
                    X_normalized = np.power(np.sum(X * X, axis=1), 0.5)
                    for i in range(n_samples):
                        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                            time_start_fit = time.time()
                            kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                            kwargs["start_time"] = time_start_fit
                            if kwargs["time_limit"] <= 0:
                                logger.warning(
                                    f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                                )
                                return None
                        X[i, :] = X[i, :] / max(1e-12, X_normalized[i])
                    # compute pairwise cosine distances
                    D_cosine = np.dot(X, np.transpose(X))
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

            elif kwargs["weight_mode"] == "heat_kernel":
                t = kwargs["t"]
                # compute pairwise euclidean distances
                D = pairwise_distances(X)
                D **= 2
                # sort the distance matrix D in ascending order
                dump = np.sort(D, axis=1)
                idx = np.argsort(D, axis=1)
                idx_new = idx[:, 0 : k + 1]
                dump_new = dump[:, 0 : k + 1]
                # compute the pairwise heat kernel distances
                dump_heat_kernel = np.exp(-dump_new / (2 * t * t))
                G = np.zeros((n_samples * (k + 1), 3))
                G[:, 0] = np.tile(np.arange(n_samples), (k + 1, 1)).reshape(-1)
                G[:, 1] = np.ravel(idx_new, order="F")
                G[:, 2] = np.ravel(dump_heat_kernel, order="F")
                # build the sparse affinity matrix W
                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                bigger = np.transpose(W) > W
                return W - W.multiply(bigger) + np.transpose(W).multiply(bigger)

            elif kwargs["weight_mode"] == "cosine":
                # normalize the data first
                X_normalized = np.power(np.sum(X * X, axis=1), 0.5)
                for i in range(n_samples):
                    if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                        time_start_fit = time.time()
                        kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                        kwargs["start_time"] = time_start_fit
                        if kwargs["time_limit"] <= 0:
                            logger.warning(
                                f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                            )
                            return None
                    X[i, :] = X[i, :] / max(1e-12, X_normalized[i])
                # compute pairwise cosine distances
                D_cosine = np.dot(X, np.transpose(X))
                # sort the distance matrix D in ascending order
                dump = np.sort(-D_cosine, axis=1)
                idx = np.argsort(-D_cosine, axis=1)
                idx_new = idx[:, 0 : k + 1]
                dump_new = -dump[:, 0 : k + 1]
                G = np.zeros((n_samples * (k + 1), 3))
                G[:, 0] = np.tile(np.arange(n_samples), (k + 1, 1)).reshape(-1)
                G[:, 1] = np.ravel(idx_new, order="F")
                G[:, 2] = np.ravel(dump_new, order="F")
                # build the sparse affinity matrix W
                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                bigger = np.transpose(W) > W
                return W - W.multiply(bigger) + np.transpose(W).multiply(bigger)

        # choose supervised neighborMode
        elif kwargs["neighbor_mode"] == "supervised":
            k = kwargs["k"]
            # get true labels and the number of classes
            y = kwargs["y"]
            label = np.unique(y)
            n_classes = np.unique(y).size
            # construct the weight matrix W in a fisherScore way, W_ij = 1/n_l if yi = yj = l, otherwise W_ij = 0
            if kwargs["fisher_score"] is True:
                W = lil_matrix((n_samples, n_samples))
                for i in range(n_classes):
                    if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                        time_start_fit = time.time()
                        kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                        kwargs["start_time"] = time_start_fit
                        if kwargs["time_limit"] <= 0:
                            logger.warning(
                                f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                            )
                            return None
                    class_idx = y == label[i]
                    class_idx_all = class_idx[:, np.newaxis] & class_idx[np.newaxis, :]
                    W[class_idx_all] = 1.0 / np.sum(np.sum(class_idx))
                return W

            # construct the weight matrix W in a reliefF way, NH(x) and NM(x,y) denotes a set of k nearest
            # points to x with the same class as x, a different class (the class y), respectively. W_ij = 1 if i = j;
            # W_ij = 1/k if x_j \in NH(x_i); W_ij = -1/(c-1)k if x_j \in NM(x_i, y)
            if kwargs["reliefF"] is True:
                # when xj in NH(xi)
                G = np.zeros((n_samples * (k + 1), 3))
                id_now = 0
                for i in range(n_classes):
                    if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                        time_start_fit = time.time()
                        kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                        kwargs["start_time"] = time_start_fit
                        if kwargs["time_limit"] <= 0:
                            logger.warning(
                                f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                            )
                            return None
                    class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                    D = pairwise_distances(X[class_idx, :])
                    D **= 2
                    idx = np.argsort(D, axis=1)
                    idx_new = idx[:, 0 : k + 1]
                    n_smp_class = (class_idx[idx_new[:]]).size
                    if len(class_idx) <= k:
                        k = len(class_idx) - 1
                    G[id_now : n_smp_class + id_now, 0] = np.tile(class_idx, (k + 1, 1)).reshape(-1)
                    G[id_now : n_smp_class + id_now, 1] = np.ravel(class_idx[idx_new[:]], order="F")
                    G[id_now : n_smp_class + id_now, 2] = 1.0 / k
                    id_now += n_smp_class
                W1 = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                # when i = j, W_ij = 1
                for i in range(n_samples):
                    if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                        time_start_fit = time.time()
                        kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                        kwargs["start_time"] = time_start_fit
                        if kwargs["time_limit"] <= 0:
                            logger.warning(
                                f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                            )
                            return None
                    W1[i, i] = 1
                # when x_j in NM(x_i, y)
                G = np.zeros((n_samples * k * (n_classes - 1), 3))
                id_now = 0
                for i in range(n_classes):
                    if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                        time_start_fit = time.time()
                        kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                        kwargs["start_time"] = time_start_fit
                        if kwargs["time_limit"] <= 0:
                            logger.warning(
                                f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                            )
                            return None
                    class_idx1 = np.column_stack(np.where(y == label[i]))[:, 0]
                    X1 = X[class_idx1, :]
                    for j in range(n_classes):
                        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                            time_start_fit = time.time()
                            kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                            kwargs["start_time"] = time_start_fit
                            if kwargs["time_limit"] <= 0:
                                logger.warning(
                                    f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                                )
                                return None
                        if label[j] != label[i]:
                            class_idx2 = np.column_stack(np.where(y == label[j]))[:, 0]
                            X2 = X[class_idx2, :]
                            D = pairwise_distances(X1, X2)
                            idx = np.argsort(D, axis=1)
                            idx_new = idx[:, 0:k]
                            n_smp_class = len(class_idx1) * k
                            G[id_now : n_smp_class + id_now, 0] = np.tile(class_idx1, (k, 1)).reshape(-1)
                            G[id_now : n_smp_class + id_now, 1] = np.ravel(class_idx2[idx_new[:]], order="F")
                            G[id_now : n_smp_class + id_now, 2] = -1.0 / ((n_classes - 1) * k)
                            id_now += n_smp_class
                W2 = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                bigger = np.transpose(W2) > W2
                W2 = W2 - W2.multiply(bigger) + np.transpose(W2).multiply(bigger)
                return W1 + W2

            if kwargs["weight_mode"] == "binary":
                if kwargs["metric"] == "euclidean":
                    G = np.zeros((n_samples * (k + 1), 3))
                    id_now = 0
                    for i in range(n_classes):
                        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                            time_start_fit = time.time()
                            kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                            kwargs["start_time"] = time_start_fit
                            if kwargs["time_limit"] <= 0:
                                logger.warning(
                                    f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                                )
                                return None
                        class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                        # compute pairwise euclidean distances for instances in class i
                        D = pairwise_distances(X[class_idx, :])
                        D **= 2
                        # sort the distance matrix D in ascending order for instances in class i
                        idx = np.argsort(D, axis=1)
                        idx_new = idx[:, 0 : k + 1]
                        n_smp_class = len(class_idx) * (k + 1)
                        G[id_now : n_smp_class + id_now, 0] = np.tile(class_idx, (k + 1, 1)).reshape(-1)
                        G[id_now : n_smp_class + id_now, 1] = np.ravel(class_idx[idx_new[:]], order="F")
                        G[id_now : n_smp_class + id_now, 2] = 1
                        id_now += n_smp_class
                    # build the sparse affinity matrix W
                    W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                    bigger = np.transpose(W) > W
                    return W - W.multiply(bigger) + np.transpose(W).multiply(bigger)

                if kwargs["metric"] == "cosine":
                    # normalize the data first
                    X_normalized = np.power(np.sum(X * X, axis=1), 0.5)
                    for i in range(n_samples):
                        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                            time_start_fit = time.time()
                            kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                            kwargs["start_time"] = time_start_fit
                            if kwargs["time_limit"] <= 0:
                                logger.warning(
                                    f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                                )
                                return None
                        X[i, :] = X[i, :] / max(1e-12, X_normalized[i])
                    G = np.zeros((n_samples * (k + 1), 3))
                    id_now = 0
                    for i in range(n_classes):
                        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                            time_start_fit = time.time()
                            kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                            kwargs["start_time"] = time_start_fit
                            if kwargs["time_limit"] <= 0:
                                logger.warning(
                                    f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                                )
                                return None
                        class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                        # compute pairwise cosine distances for instances in class i
                        D_cosine = np.dot(X[class_idx, :], np.transpose(X[class_idx, :]))
                        # sort the distance matrix D in descending order for instances in class i
                        idx = np.argsort(-D_cosine, axis=1)
                        idx_new = idx[:, 0 : k + 1]
                        n_smp_class = len(class_idx) * (k + 1)
                        G[id_now : n_smp_class + id_now, 0] = np.tile(class_idx, (k + 1, 1)).reshape(-1)
                        G[id_now : n_smp_class + id_now, 1] = np.ravel(class_idx[idx_new[:]], order="F")
                        G[id_now : n_smp_class + id_now, 2] = 1
                        id_now += n_smp_class
                    # build the sparse affinity matrix W
                    W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                    bigger = np.transpose(W) > W
                    return W - W.multiply(bigger) + np.transpose(W).multiply(bigger)

            elif kwargs["weight_mode"] == "heat_kernel":
                G = np.zeros((n_samples * (k + 1), 3))
                id_now = 0
                for i in range(n_classes):
                    if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                        time_start_fit = time.time()
                        kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                        kwargs["start_time"] = time_start_fit
                        if kwargs["time_limit"] <= 0:
                            logger.warning(
                                f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                            )
                            return None
                    class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                    # compute pairwise cosine distances for instances in class i
                    D = pairwise_distances(X[class_idx, :])
                    D **= 2
                    # sort the distance matrix D in ascending order for instances in class i
                    dump = np.sort(D, axis=1)
                    idx = np.argsort(D, axis=1)
                    idx_new = idx[:, 0 : k + 1]
                    dump_new = dump[:, 0 : k + 1]
                    t = kwargs["t"]
                    # compute pairwise heat kernel distances for instances in class i
                    dump_heat_kernel = np.exp(-dump_new / (2 * t * t))
                    n_smp_class = len(class_idx) * (k + 1)
                    G[id_now : n_smp_class + id_now, 0] = np.tile(class_idx, (k + 1, 1)).reshape(-1)
                    G[id_now : n_smp_class + id_now, 1] = np.ravel(class_idx[idx_new[:]], order="F")
                    G[id_now : n_smp_class + id_now, 2] = np.ravel(dump_heat_kernel, order="F")
                    id_now += n_smp_class
                # build the sparse affinity matrix W
                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                bigger = np.transpose(W) > W
                return W - W.multiply(bigger) + np.transpose(W).multiply(bigger)

            elif kwargs["weight_mode"] == "cosine":
                # normalize the data first
                X_normalized = np.power(np.sum(X * X, axis=1), 0.5)
                for i in range(n_samples):
                    if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                        time_start_fit = time.time()
                        kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                        kwargs["start_time"] = time_start_fit
                        if kwargs["time_limit"] <= 0:
                            logger.warning(
                                f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                            )
                            return None
                    X[i, :] = X[i, :] / max(1e-12, X_normalized[i])
                G = np.zeros((n_samples * (k + 1), 3))
                id_now = 0
                for i in range(n_classes):
                    if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                        time_start_fit = time.time()
                        kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                        kwargs["start_time"] = time_start_fit
                        if kwargs["time_limit"] <= 0:
                            logger.warning(
                                f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                            )
                            return None
                    class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                    # compute pairwise cosine distances for instances in class i
                    D_cosine = np.dot(X[class_idx, :], np.transpose(X[class_idx, :]))
                    # sort the distance matrix D in descending order for instances in class i
                    dump = np.sort(-D_cosine, axis=1)
                    idx = np.argsort(-D_cosine, axis=1)
                    idx_new = idx[:, 0 : k + 1]
                    dump_new = -dump[:, 0 : k + 1]
                    n_smp_class = len(class_idx) * (k + 1)
                    G[id_now : n_smp_class + id_now, 0] = np.tile(class_idx, (k + 1, 1)).reshape(-1)
                    G[id_now : n_smp_class + id_now, 1] = np.ravel(class_idx[idx_new[:]], order="F")
                    G[id_now : n_smp_class + id_now, 2] = np.ravel(dump_new, order="F")
                    id_now += n_smp_class
                # build the sparse affinity matrix W
                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                bigger = np.transpose(W) > W
                return W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
        return None
