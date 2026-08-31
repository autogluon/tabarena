# Copyright (C) 2026 Xiaomi Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import sys
import random
import itertools
from collections import OrderedDict
from copy import deepcopy
from typing import List, Optional

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OrdinalEncoder,
    StandardScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
)
from sklearn.utils.validation import check_is_fitted

from .sklearn_utils import validate_data


class RecursionLimitManager:
    """Context manager to temporarily set the recursion limit.

    Parameters
    ----------
    limit : int
        The recursion limit to set temporarily.

    Examples
    --------
    >>> with RecursionLimitManager(4000):
    ...     # Perform operations that require a higher recursion limit
    ...     pass
    """

    def __init__(self, limit):
        self.limit = limit
        self.original_limit = None

    def __enter__(self):
        self.original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.limit)
        return self

    def __exit__(self, type, value, traceback):
        sys.setrecursionlimit(self.original_limit)
        return False  # Return False to propagate exceptions


class TransformToNumerical(TransformerMixin, BaseEstimator):
    """Transform non-numerical data in a DataFrame to numerical representations.

    This transformer automatically detects and converts categorical variables, text features,
    and boolean data types into numerical representations suitable for machine learning models.

    Parameters
    ----------
    verbose : bool, default=False
        Whether to print information about column classifications.

    Attributes
    ----------
    tfm_ : ColumnTransformer or FunctionTransformer
        The fitted transformer that handles the conversion of different column types.

        - If input is a DataFrame: a ``ColumnTransformer`` with ``OrdinalEncoder``
          for categorical columns and ``SimpleImputer`` for numeric columns.
        - If input is not a DataFrame: a ``FunctionTransformer`` that passes data
          through unchanged.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def fit(self, X, y=None):
        """Configure transformers for different column types in the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data. If a DataFrame, column types are used to determine
            appropriate transformations.

        y : None
            Ignored.

        Returns
        -------
        self : TransformToNumerical
            Returns self.
        """

        cat_tfm = OrdinalEncoder(
            dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
        )
        num_tfm = SimpleImputer()

        if not hasattr(X, "columns"):  # proxy way to check whether X is a dataframe without importing pandas
            # no dataframe, so we can't do column-wise transformations. Instead, we check if it's already numeric and if not, raise an error.
            
            # For compatibility with sklearn's tests
            if issparse(X):
                raise TypeError(
                    "Sparse input is not supported by TabLDM. "
                    "Convert X to a dense array, e.g. with X.toarray()."
                )
            X_arr = np.asarray(X)
            try:
                X_arr.astype(np.float64)
            except (ValueError, TypeError) as e:
                # Preserve the original exception type so that, e.g., object arrays
                # holding non-string/non-number elements still raise a TypeError.
                raise type(e)(
                    "NumPy arrays passed to TabLDM must be castable to a numeric dtype, "
                    f"but casting to float64 failed with: {e}. "
                    "If your data contains categorical or string columns, pass it as a pandas "
                    "DataFrame instead, so each column can be typed and preprocessed accordingly."
                ) from None
            self.tfm_ = num_tfm

        else:

            cat_cols = make_column_selector(dtype_include=["string", "object", "category", "boolean"])(X)
            cat_pos = [X.columns.get_loc(col) for col in cat_cols]

            high_cardinality_cols = [col for col in cat_cols if X[col].nunique() > 40]
            if high_cardinality_cols:
                import warnings

                warnings.warn(
                    f"The following categorical columns have a cardinality above 40: {high_cardinality_cols}. "
                    "High-cardinality columns might benefit from a better encoding than ordinal encoding, "
                    "e.g. Skrub's TableVectorizer for strings."
                )

            numeric_cols = make_column_selector(dtype_include="number")(X)
            numeric_pos = [X.columns.get_loc(col) for col in numeric_cols]

            self.tfm_ = ColumnTransformer(
                transformers=[("continuous", num_tfm, numeric_pos), ("categorical", cat_tfm, cat_pos)]
            )

        self.tfm_.fit(X)

        if self.verbose and hasattr(self.tfm_, "transformers_"):
            selected_cols = []
            for name, tfm, pos in self.tfm_.transformers_:
                if tfm != "drop":
                    cols = list(X.columns[pos])
                    selected_cols.extend(cols)
                    print(f"Columns classified as {name}: {cols}")

            dropped_cols = set(X.columns).difference(set(selected_cols))
            if len(dropped_cols) >= 1:
                print(f"The following columns are not used due to their data type: {list(dropped_cols)}")

        return self

    def transform(self, X):
        """Transform features using the fitted transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed array with numerical representations.
        """
        return self.tfm_.transform(X)


class UniqueFeatureFilter(TransformerMixin, BaseEstimator):
    """Filter that removes features with only one unique value in the training set.

    Parameters
    ----------
    threshold : int, default=1
        Features with unique values less than or equal to this threshold will be removed.

    Attributes
    ----------
    n_features_in_ : int
        Number of features in the training data.

    n_features_out_ : int
        Number of features after filtering.

    features_to_keep_ : ndarray
        Boolean mask for features to keep.

    Notes
    -----
    1. Features with unique values <= ``threshold`` are removed.
    2. When the input dataset has very few samples
       (:math:`n_{\\text{samples}} \\le \\text{threshold}`), all features are preserved
       regardless of their unique value counts. This is a safety mechanism because:

       - With few samples, it's difficult to reliably assess feature variability.
       - A feature might appear constant in few samples but vary in the complete dataset.
    """

    def __init__(self, threshold: int = 1):
        self.threshold = threshold

    def fit(self, X, y=None):
        """Learn which features to keep based on unique value counts.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        X = validate_data(self, X)

        # If there are very few samples, keep all features
        if X.shape[0] <= self.threshold:
            self.features_to_keep_ = np.ones(self.n_features_in_, dtype=bool)
        else:
            # For each feature, check if it has more than threshold unique values
            self.features_to_keep_ = np.array(
                [len(np.unique(X[:, i])) > self.threshold for i in range(self.n_features_in_)]
            )

        self.n_features_out_ = np.sum(self.features_to_keep_)

        return self

    def transform(self, X):
        """Filter features according to unique value counts.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features_out_)
            Transformed array with selected features.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        return X[:, self.features_to_keep_]


class OutlierRemover(TransformerMixin, BaseEstimator):
    """Transformer that clips extreme values based on training data distribution.

    This implementation uses a two-stage Z-score based approach to identify and
    clip outliers:

    1. First stage: Identify values with :math:`|z| > \text{threshold}` standard
       deviations and mark as missing.
    2. Second stage: Recompute statistics without outliers for more robust bounds.
    3. Final stage: Apply log-based clipping to maintain data distribution.

    Parameters
    ----------
    threshold : float, default=4.0
        Values beyond this number of standard deviations are considered outliers,
        i.e., values with :math:`|z| > \text{threshold}`.

    Attributes
    ----------
    n_features_in_ : int
        Number of features in the training data.

    means_ : ndarray of shape (n_features_in_,)
        Mean values per feature after removing outliers.

    stds_ : ndarray of shape (n_features_in_,)
        Standard deviation values per feature after removing outliers.

    lower_bounds_ : ndarray of shape (n_features_in_,)
        Lower bounds for clipping,
        :math:`\\mu - \\text{threshold} \\cdot \\sigma`.

    upper_bounds_ : ndarray of shape (n_features_in_,)
        Upper bounds for clipping,
        :math:`\\mu + \\text{threshold} \\cdot \\sigma`.
    """

    def __init__(self, threshold: float = 4.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        """Learn clipping bounds from training data using two-stage Z-score method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.

        y : None
            Ignored.

        Returns
        -------
        self : OutlierRemover
            Returns self.
        """
        X = validate_data(self, X)

        # First stage: Identify outliers using initial statistics
        self.means_ = np.nanmean(X, axis=0)
        self.stds_ = np.nanstd(X, axis=0, ddof=1 if X.shape[0] > 1 else 0)

        # Ensure standard deviations are not zero
        self.stds_ = np.maximum(self.stds_, 1e-6)

        # Create a clean copy with outliers replaced by NaN
        X_clean = X.copy()
        lower_bounds = self.means_ - self.threshold * self.stds_
        upper_bounds = self.means_ + self.threshold * self.stds_

        # Create masks for values outside bounds
        lower_mask = X < lower_bounds[np.newaxis, :]
        upper_mask = X > upper_bounds[np.newaxis, :]
        outlier_mask = np.logical_or(lower_mask, upper_mask)

        # Set outliers to NaN
        X_clean[outlier_mask] = np.nan

        # Second stage: Recompute statistics without outliers
        self.means_ = np.nanmean(X_clean, axis=0)
        self.stds_ = np.nanstd(X_clean, axis=0, ddof=1 if X.shape[0] > 1 else 0)

        # Ensure standard deviations are not zero
        self.stds_ = np.maximum(self.stds_, 1e-6)

        # Compute final bounds
        self.lower_bounds_ = self.means_ - self.threshold * self.stds_
        self.upper_bounds_ = self.means_ + self.threshold * self.stds_

        return self

    def transform(self, X):
        """Clip values based on learned bounds with log-based adjustments.

        Values are clipped using soft bounds:

        .. math::

            x_{\\text{clipped}} = \\max\\bigl(-\\log(1+|x|) + L,\\; x\\bigr)

            x_{\\text{clipped}} = \\min\\bigl(\\log(1+|x|) + U,\\; x\\bigr)

        where :math:`L` and :math:`U` are the lower and upper bounds.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed array with clipped values.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = np.maximum(-np.log1p(np.abs(X)) + self.lower_bounds_, X)
        X = np.minimum(np.log1p(np.abs(X)) + self.upper_bounds_, X)

        return X


class CustomStandardScaler(TransformerMixin, BaseEstimator):
    """Custom implementation of standard scaling with clipping.

    Computes the z-score :math:`z = (x - \\mu) / (\\sigma + \\epsilon)` and clips
    the result to ``[clip_min, clip_max]``.

    Parameters
    ----------
    clip_min : float, default=-100
        Lower bound for clipping transformed values.

    clip_max : float, default=100
        Upper bound for clipping transformed values.

    epsilon : float, default=1e-6
        Small constant :math:`\\epsilon` added to the standard deviation to avoid
        division by zero.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        The mean value for each feature in the training set.

    scale_ : ndarray of shape (n_features,)
        The standard deviation for each feature in the training set with
        :math:`\\epsilon` added.
    """

    def __init__(self, clip_min: float = -100, clip_max: float = 100, epsilon: float = 1e-6):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon

    def fit(self, X, y=None):
        """Compute the mean and std to be used for scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.

        y : None
            Ignored.

        Returns
        -------
        self : CustomStandardScaler
            Returns self.
        """

        if len(X.shape) == 1:
            # If X is a 1D array, reshape it to 2D
            X = X.reshape(-1, 1)

        X = validate_data(self, X)

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0) + self.epsilon

        return self

    def transform(self, X):
        """Standardize features by removing the mean and scaling to unit variance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed array after scaling and clipping.
        """
        if len(X.shape) == 1:
            is_vector = True
            X = X.reshape(-1, 1)
        else:
            is_vector = False

        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        X_scaled = (X - self.mean_) / self.scale_
        X_clipped = np.clip(X_scaled, self.clip_min, self.clip_max)

        return X_clipped.reshape(-1) if is_vector else X_clipped

    def inverse_transform(self, X):
        """Scale back the data to the original representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to inverse transform.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed array in original scale.
        """
        if len(X.shape) == 1:
            is_vector = True
            X = X.reshape(-1, 1)
        else:
            is_vector = False

        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X_out = X * self.scale_ + self.mean_

        return X_out.reshape(-1) if is_vector else X_out


class RTDLQuantileTransformer(BaseEstimator, TransformerMixin):
    """Quantile transformer adapted for tabular deep learning models.

    This implementation is based on research from the RTDL group and adds noise to training
    data before applying quantile transformation, improving robustness and generalization.
    It also dynamically adjusts the number of quantiles based on data size as
    :math:`\\min(n_{\\text{samples}} / 30,\\; \\text{n\\_quantiles})` with a minimum of 10.

    Parameters
    ----------
    noise : float, default=1e-3
        Magnitude of Gaussian noise to add relative to feature standard deviations.
        Set to 0 to disable noise addition.

    n_quantiles : int, default=1000
        Maximum number of quantiles to use. The actual number used is dynamically
        determined as :math:`\\min(\\lfloor n / 30 \\rfloor, \\text{n\\_quantiles})`
        with a minimum of 10.

    subsample : int, default=1_000_000_000
        Maximum number of samples used to estimate the quantiles for computational
        efficiency.

    output_distribution : {'uniform', 'normal'}, default='normal'
        Marginal distribution for the transformed data.

    random_state : int or None, default=None
        Seed for random number generation for reproducible noise and quantile sampling.

    Attributes
    ----------
    normalizer_ : QuantileTransformer
        Fitted transformer used to transform the data.

    Notes
    -----
    Adapted from https://github.com/yandex-research/tabular-dl-tabr/blob/75105013189c76bc4f247633c2fb856bc948e579/lib/data.py#L262
    following https://github.com/dholzmueller/pytabkit/blob/949bf81e3964f65a33dd2c252c3713c239c17b2d/pytabkit/models/utils.py#L431
    """

    def __init__(
        self,
        noise: float = 1e-3,
        n_quantiles: int = 1000,
        subsample: int = 1_000_000_000,
        output_distribution: str = "normal",
        random_state: Optional[int] = None,
    ):
        self.noise = noise
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.output_distribution = output_distribution
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the quantile transformer to training data with optional noise addition.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data to fit the transformer.

        y : None
            Ignored.

        Returns
        -------
        self : RTDLQuantileTransformer
            Returns self.
        """
        # Calculate the number of quantiles based on data size
        n_quantiles = max(min(X.shape[0] // 30, self.n_quantiles), 10)

        # Initialize QuantileTransformer
        normalizer = QuantileTransformer(
            output_distribution=self.output_distribution,
            n_quantiles=n_quantiles,
            subsample=self.subsample,
            random_state=self.random_state,
        )

        # Add noise if required
        X_modified = self._add_noise(X) if self.noise > 0 else X

        # Fit the normalizer
        normalizer.fit(X_modified)

        # Show that it's fitted
        self.normalizer_ = normalizer

        return self

    def transform(self, X, y=None):
        """Transform data using the fitted quantile transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to be transformed.

        y : None
            Ignored.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            The transformed data with distribution specified by
            ``output_distribution``.
        """
        check_is_fitted(self)
        return self.normalizer_.transform(X)

    def _add_noise(self, X):
        """Add noise to the input data proportional to feature standard deviations.

        The noise magnitude is controlled by the 'noise' parameter and is scaled
        inversely to the standard deviation of each feature to ensure
        consistent noise levels across features of different scales.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to add noise to.

        Returns
        -------
        X_noisy : ndarray of shape (n_samples, n_features)
            The input data with added Gaussian noise.
        """
        stds = np.std(X, axis=0, keepdims=True)
        noise_std = self.noise / np.maximum(stds, self.noise)
        rng = np.random.default_rng(self.random_state)
        X_noisy = X + noise_std * rng.standard_normal(X.shape)
        return X_noisy


class PreprocessingPipeline(TransformerMixin, BaseEstimator):
    """Preprocessing pipeline for tabular data.

    This pipeline combines scaling, normalization, and outlier handling.

    Parameters
    ----------
    normalization_method : str, default='power'
        Method for normalization: ``'power'``, ``'quantile'``,
        ``'quantile_rtdl'``, ``'robust'``, ``'none'``.

    outlier_threshold : float, default=4.0
        Z-score threshold for outlier detection. Values with
        :math:`|z| > \text{threshold}` are considered outliers.

    random_state : int or None, default=None
        Random seed for reproducible normalization.

    Attributes
    ----------
    n_features_in_ : int
        Number of features in the training data.

    standard_scaler_ : CustomStandardScaler
        The fitted standard scaler.

    normalizer_ : sklearn transformer or None
        The fitted normalization transformer (``PowerTransformer``,
        ``QuantileTransformer``, ``RTDLQuantileTransformer``, or
        ``RobustScaler``). ``None`` when ``normalization_method='none'``.

    outlier_remover_ : OutlierRemover
        The fitted outlier remover.

    X_transformed_ : ndarray of shape (n_samples, n_features)
        The transformed training input data. Saved for later use to avoid
        recomputation.
    """

    def __init__(
        self, normalization_method: str = "power", outlier_threshold: float = 4.0, random_state: Optional[int] = None
    ):
        self.normalization_method = normalization_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the preprocessing pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : None
            Ignored.

        Returns
        -------
        self : PreprocessingPipeline
            Returns self.
        """
        X = validate_data(self, X)

        # 1. Apply standard scaling
        self.standard_scaler_ = CustomStandardScaler()
        X_scaled = self.standard_scaler_.fit_transform(X)

        # 2. Apply normalization
        if self.normalization_method != "none":
            if self.normalization_method == "power":
                self.normalizer_ = PowerTransformer(method="yeo-johnson", standardize=True)
            elif self.normalization_method == "quantile":
                self.normalizer_ = QuantileTransformer(output_distribution="normal", random_state=self.random_state)
            elif self.normalization_method == "quantile_rtdl":
                self.normalizer_ = Pipeline(
                    [
                        (
                            "quantile_rtdl",
                            RTDLQuantileTransformer(output_distribution="normal", random_state=self.random_state),
                        ),
                        ("std", StandardScaler()),
                    ]
                )
            elif self.normalization_method == "robust":
                self.normalizer_ = RobustScaler(unit_variance=True)
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")

            self.X_min_ = np.min(X_scaled, axis=0, keepdims=True)
            self.X_max_ = np.max(X_scaled, axis=0, keepdims=True)
            X_normalized = self.normalizer_.fit_transform(X_scaled)
        else:
            self.normalizer_ = None
            X_normalized = X_scaled

        # 3. Handle outliers
        self.outlier_remover_ = OutlierRemover(threshold=self.outlier_threshold)
        self.X_transformed_ = self.outlier_remover_.fit_transform(X_normalized)

        return self

    def transform(self, X):
        """Apply the preprocessing pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_out : ndarray
            Preprocessed data.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, copy=True)
        # Standard scaling
        X = self.standard_scaler_.transform(X)
        # Normalization
        if self.normalizer_ is not None:
            try:
                # this can fail in rare cases if there is an outlier in X that was not present in fit()
                X = self.normalizer_.transform(X)
            except ValueError:
                # clip values to train min/max
                X = np.clip(X, self.X_min_, self.X_max_)
                X = self.normalizer_.transform(X)
        # Outlier removal
        X = self.outlier_remover_.transform(X)

        return X


class Shuffler:
    """Utility that generates permutations for ensemble creation.

    This class provides methods to create different types of permutations
    that can be used when creating ensemble variants of datasets.

    Parameters
    ----------
    n_elements : int
        Number of elements to shuffle.

    method : str, default='latin'
        Method used for shuffling:
        - ``'none'``: No shuffling.
        - ``'random'``: Random permutation.
        - ``'latin'``: Latin square permutation.
        - ``'shift'``: Circular shift of elements.

    max_elements_for_latin : int, default=4000
        Maximum number of elements for which Latin square permutations are
        generated. If the number of elements exceeds this limit, random
        permutations are used instead.

    random_state : int or None, default=None
        Random seed for reproducible shuffling.
    """

    def __init__(
        self,
        n_elements: int,
        method: str = "latin",
        max_elements_for_latin: int = 4000,
        random_state: Optional[int] = None,
    ):
        self.n_elements = n_elements
        self.method = method
        self.max_elements_for_latin = max_elements_for_latin
        self.random_state = random_state

    def shuffle(self, n_estimators: int) -> List[np.ndarray]:
        """Generate shuffling patterns for ensemble diversity.

        Creates permutations of indices according to the specified method,
        which can be used to reorder elements when creating ensemble variants.

        Parameters
        ----------
        n_estimators : int
            Number of permutations to generate.

            - For ``'none'`` method: Always returns a single pattern with no shuffling.
            - For ``'shift'`` method: Generates all possible circular shifts.
            - For ``'latin'`` method: Generates Latin square permutations.
            - For ``'random'`` method: For small element sets
              (:math:`n_{\\text{elements}} \\le 5`), samples from all possible
              permutations; otherwise generates random permutations.

        Returns
        -------
        list of ndarray
            List of permutation arrays, where each array contains
            indices that can be used to shuffle elements.
        """

        self.rng_ = random.Random(self.random_state)
        indices = list(range(self.n_elements))

        # Use the random method if n_elements exceeds the limit for Latin square
        if self.n_elements > self.max_elements_for_latin and self.method == "latin":
            method = "random"
        else:
            method = self.method

        # No shuffling
        if method == "none" or n_estimators == 1:
            shuffle_patterns = [indices]
            return shuffle_patterns

        # Generate permutations based on method
        if method == "shift":
            # All possible circular shifts
            shuffle_patterns = [indices[-i:] + indices[:-i] for i in range(self.n_elements)]
        elif method == "random":
            # Random permutations
            if self.n_elements <= 5:
                all_perms = [list(perm) for perm in itertools.permutations(indices)]
                shuffle_patterns = self.rng_.sample(all_perms, min(n_estimators, len(all_perms)))
            else:
                shuffle_patterns = [self.rng_.sample(indices, self.n_elements) for _ in range(n_estimators)]
        elif method == "latin":
            # Latin square permutations
            with RecursionLimitManager(100000):  # Set a higher recursion limit to avoid recursion error
                shuffle_patterns = self._latin_squares()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'shift', 'random', 'latin', or 'none'.")

        return shuffle_patterns

    def _latin_squares(self):
        """Generate Latin squares for shuffling.

        Returns
        -------
        list
            List of permutations forming a Latin square.
        """

        def _shuffle_transpose_shuffle(matrix):
            square = deepcopy(matrix)
            self.rng_.shuffle(square)
            trans = list(zip(*square))
            self.rng_.shuffle(trans)
            return trans

        def _rls(symbols):
            n = len(symbols)
            if n == 1:
                return [symbols]
            else:
                sym = self.rng_.choice(symbols)
                symbols.remove(sym)
                square = _rls(symbols)
                square.append(square[0].copy())
                for i in range(n):
                    square[i].insert(i, sym)
                return square

        symbols = list(range(self.n_elements))
        square = _rls(symbols)
        shuffles = _shuffle_transpose_shuffle(square)

        return [list(shuffle) for shuffle in shuffles]


class EnsembleGenerator(TransformerMixin, BaseEstimator):
    """Generate ensemble variants for robust tabular prediction with TabLDM.

    This class creates diverse data variants through:

    1. Applying different normalization techniques.
    2. Permuting feature orders to exploit position-invariance in transformer
       architectures.
    3. For classification: Shuffling class labels to prevent overfitting to
       specific class index patterns.

    Parameters
    ----------
    classification : bool
        Whether to generate ensembles for classification tasks.

    n_estimators : int
        Number of ensemble variants to generate.

    norm_methods : str or list[str] or None, default=None
        Normalization methods to apply:
        - ``'none'``: No normalization.
        - ``'power'``: Yeo-Johnson power transform.
        - ``'quantile'``: Transform feature distribution to approximately
          Gaussian, using the empirical quantiles.
        - ``'quantile_rtdl'``: Version of the quantile transform used
          typically in papers by the RTDL group.
        - ``'robust'``: Scale using median and quantiles.
        If set to None, ``['none', 'power']`` will be applied.

    feat_shuffle_method : str, default='latin'
        Feature permutation strategy:
        - ``'none'``: No shuffling and preserve original feature order.
        - ``'shift'``: Circular shifting.
        - ``'random'``: Random permutation.
        - ``'latin'``: Latin square patterns.

    class_shuffle_method : str, default='shift'
        Class label permutation strategy for classification tasks
        (``classification=True``):
        - ``'none'``: No shuffling and preserve original class labels.
        - ``'shift'``: Circular shifting.
        - ``'random'``: Random permutation.
        - ``'latin'``: Latin square patterns.

    cat_random_encode : bool, default=False
        Randomly permute the integer codes of categorical feature columns for
        each ensemble member. When enabled for classification, class labels
        also use random code permutations regardless of ``class_shuffle_method``.

    categorical_indices : array-like of int or None, default=None
        Feature columns in the input matrix that should be treated as
        categorical, before unique-feature filtering. When ``None``, columns
        with fewer than ten observed values are treated as categorical.

    outlier_threshold : float, default=4.0
        Z-score threshold for outlier detection and clipping. Values with
        :math:`|z| > \text{threshold}` are considered outliers.

    random_state : int or None, default=None
        Seed for reproducible ensemble generation.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features after filtering.

    n_classes_ : int
        Number of unique target classes for classification.

    unique_filter_ : UniqueFeatureFilter
        Filter that removes features with only one unique value.

    preprocessors_ : dict
        Maps normalization methods to fitted preprocessing pipelines.

    ensemble_configs_ : OrderedDict
        Generated ensemble configurations, organized by normalization method.
        Keys are normalization methods and values are lists of
        ``(X_shuffle, y_pattern)`` tuples, where ``y_pattern`` is a class
        shuffle for classification or ``None`` for regression.

    feature_shuffles_ : OrderedDict
        Maps normalization methods to lists of feature index permutations.

    class_shuffles_ : OrderedDict
        Maps normalization methods to lists of class index permutations for
        classification.

    X_ : ndarray
        Training feature data after filtering.

    y_ : ndarray
        Training target values.
    """

    def __init__(
        self,
        classification: bool,
        n_estimators: int,
        norm_methods: str | List[str] | None = None,
        feat_shuffle_method: str = "latin",
        class_shuffle_method: str = "shift",
        outlier_threshold: float = 4.0,
        random_state: Optional[int] = None,
        cat_random_encode: bool = False,
        categorical_indices: Optional[List[int]] = None,
        use_svd: bool = False,
        svd_n_components: Optional[int] = None,
        svd_ratio: float = 0.5,
    ):
        self.classification = classification
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method

        assert class_shuffle_method in ["none", "shift", "random", "latin"], "Invalid class shuffle method."
        self.class_shuffle_method = class_shuffle_method

        self.outlier_threshold = outlier_threshold
        self.random_state = random_state
        if not isinstance(cat_random_encode, bool):
            raise TypeError("cat_random_encode must be a bool")
        self.cat_random_encode = cat_random_encode
        self.categorical_indices = categorical_indices
        self.use_svd = use_svd
        self.svd_n_components = svd_n_components
        self.svd_ratio = svd_ratio

    def fit(self, X, y, frozen_unique_filter=None):
        """Create ensemble configurations and fit preprocessing pipelines.

        This method:

        1. Removes features with only one unique value.
        2. Generates diverse ensemble configurations.
        3. Fits preprocessing pipelines for each normalization method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature data.

        y : array-like of shape (n_samples,)
            Training target values.

        frozen_unique_filter : UniqueFeatureFilter or None, optional (default=None)
            If provided, reuse this pre-fitted filter instead of fitting a new one.
            This ensures fold-local generators produce the same n_features_in_ as
            the full-train generators, which is required for consistent ensemble
            view counts in enhanced mode.

        Returns
        -------
        self : EnsembleGenerator
            Fitted generator.
        """
        validate_data(self, X, y)

        if self.norm_methods is None:
            self.norm_methods_ = ["none", "power"]
        else:
            if isinstance(self.norm_methods, str):
                self.norm_methods_ = [self.norm_methods]
            else:
                self.norm_methods_ = self.norm_methods

        # Filter unique features
        if frozen_unique_filter is not None:
            # Reuse the pre-fitted filter from the full-train generator so that
            # fold-local generators produce identical n_features_in_.
            self.unique_filter_ = frozen_unique_filter
            X = self.unique_filter_.transform(X)
        else:
            self.unique_filter_ = UniqueFeatureFilter()
            X = self.unique_filter_.fit_transform(X)

        if self.categorical_indices is None:
            self.category_columns_ = [
                column
                for column in range(X.shape[1])
                if np.unique(X[:, column][~np.isnan(X[:, column])]).size < 10
            ]
        else:
            original_indices = set(self.categorical_indices)
            if any(
                isinstance(index, bool)
                or not isinstance(index, (int, np.integer))
                or index < 0
                or index >= self.unique_filter_.n_features_in_
                for index in self.categorical_indices
            ):
                raise ValueError("categorical_indices contains an invalid column")
            self.category_columns_ = [
                filtered_index
                for filtered_index, original_index in enumerate(
                    np.flatnonzero(self.unique_filter_.features_to_keep_)
                )
                if original_index in original_indices
            ]

        # SVD feature augmentation
        self.svd_augmentor_ = None
        if self.use_svd:
            n_comp = self.svd_n_components
            if n_comp is None:
                n_comp = max(1, int(X.shape[1] * self.svd_ratio))
                n_comp = min(n_comp, min(X.shape[0], X.shape[1]))
            self.svd_augmentor_ = SVDFeatureAugmentor(n_components=n_comp)
            self.svd_augmentor_.fit(X)
            X = self.svd_augmentor_.transform(X)
            if not np.isfinite(X).all():
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        self.X_ = X
        self.y_ = y

        # override n_features_in_ to account for unique feature filtering + SVD
        self.n_features_in_ = X.shape[1]
        if self.classification:
            self.n_classes_ = len(np.unique(y))

        self.rng_ = random.Random(self.random_state)
        self.ensemble_configs_, self.feature_shuffles_, y_patterns = self._generate_ensemble()

        self.category_code_mappings_ = OrderedDict()
        if self.cat_random_encode:
            for norm_method, shuffle_configs in self.ensemble_configs_.items():
                self.category_code_mappings_[norm_method] = [
                    self._new_category_code_mapping(X)
                    for _ in shuffle_configs
                ]

        if self.classification:
            self.class_shuffles_ = y_patterns

        # Fit preprocessing pipelines
        self.preprocessors_ = {}
        self.member_preprocessors_ = OrderedDict()
        for norm_method in self.ensemble_configs_:
            preprocessor = PreprocessingPipeline(
                normalization_method=norm_method,
                outlier_threshold=self.outlier_threshold,
                random_state=self.random_state,
            )
            preprocessor.fit(X)
            self.preprocessors_[norm_method] = preprocessor

            if self.cat_random_encode:
                member_preprocessors = []
                for mapping in self.category_code_mappings_[norm_method]:
                    member_preprocessor = PreprocessingPipeline(
                        normalization_method=norm_method,
                        outlier_threshold=self.outlier_threshold,
                        random_state=self.random_state,
                    )
                    member_preprocessor.fit(self._apply_category_code_mapping(X, mapping))
                    member_preprocessors.append(member_preprocessor)
                self.member_preprocessors_[norm_method] = member_preprocessors

        return self

    def _new_category_code_mapping(self, X):
        mapping = {}
        for column in self.category_columns_:
            observed = X[:, column][~np.isnan(X[:, column])]
            categories = np.unique(observed)
            codes = list(range(categories.size))
            self.rng_.shuffle(codes)
            mapping[column] = (categories, np.asarray(codes, dtype=np.int64))
        return mapping

    def _random_class_patterns(self):
        indices = list(range(self.n_classes_))
        return [
            self.rng_.sample(indices, len(indices))
            for _ in range(self.n_estimators)
        ]

    @staticmethod
    def _apply_category_code_mapping(X, mapping):
        result = np.array(X, copy=True)
        for column, (categories, codes) in mapping.items():
            values = result[:, column]
            observed = ~np.isnan(values)
            if not np.any(observed) or categories.size == 0:
                continue
            positions = np.searchsorted(categories, values[observed])
            bounded = np.minimum(positions, categories.size - 1)
            known = (positions < categories.size) & (
                categories[bounded] == values[observed]
            )
            mapped = np.full(
                positions.shape,
                codes[0],
                dtype=result.dtype,
            )
            mapped[known] = codes[bounded[known]]
            result[observed, column] = mapped
        return result

    def _generate_ensemble(self):
        """Create diverse ensemble configurations grouped by normalization method.

        Returns
        -------
        ensemble_configs : OrderedDict
            Maps normalization methods to shuffle configs.

        X_shuffle_dict : OrderedDict
            Maps normalization methods to lists of feature shuffle patterns.

        y_pattern_dict : OrderedDict
            Maps normalization methods to lists of class shuffles for
            classification or ``None`` patterns for regression.
        """

        # Generate feature shuffle patterns
        feat_shuffler = Shuffler(
            n_elements=self.n_features_in_, method=self.feat_shuffle_method, random_state=self.random_state
        )
        X_shuffles = feat_shuffler.shuffle(self.n_estimators)

        if self.classification:
            # For classification, generate class shuffle patterns
            if self.cat_random_encode:
                y_patterns = self._random_class_patterns()
            else:
                class_shuffler = Shuffler(
                    n_elements=self.n_classes_,
                    method=self.class_shuffle_method,
                    random_state=self.random_state,
                )
                y_patterns = class_shuffler.shuffle(self.n_estimators)
        else:
            y_patterns = [None]

        # Create configurations combining feature and target patterns
        shuffle_configs = list(itertools.product(X_shuffles, y_patterns))
        self.rng_.shuffle(shuffle_configs)

        shuffle_norm_configs = list(itertools.product(shuffle_configs, self.norm_methods_))
        shuffle_norm_configs = shuffle_norm_configs[: self.n_estimators]

        # Reorganize configs so that those with the same normalization method are grouped together
        used_methods = list(set([config[1] for config in shuffle_norm_configs]))

        ensemble_configs = OrderedDict()
        X_shuffle_dict = OrderedDict()
        y_pattern_dict = OrderedDict()

        for method in used_methods:
            shuffle_configs = [config[0] for config in shuffle_norm_configs if config[1] == method]
            X_shuffle_dict[method] = [config[0] for config in shuffle_configs]
            y_pattern_dict[method] = [config[1] for config in shuffle_configs]
            ensemble_configs[method] = shuffle_configs

        return ensemble_configs, X_shuffle_dict, y_pattern_dict

    def transform(self, X=None, mode="both", feature_mask=None):
        """Create ensemble data variants for in-context learning.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or None
            Test input data. Required when mode is ``'both'`` or ``'test'``.
            Can be ``None`` when mode is ``'train'``.

        mode : str, default='both'
            Controls what data is returned:

            - ``'both'``: Combines training and test data. Returns
              ``OrderedDict`` mapping normalization methods to
              ``(X_ensemble[n_variants, n_train+n_test, n_features], y_ensemble[n_variants, n_train])``.
            - ``'train'``: Returns only preprocessed and shuffled training
              data. Returns ``OrderedDict`` mapping normalization methods to
              ``(X_train_ensemble[n_variants, n_train, n_features], y_ensemble[n_variants, n_train])``.
            - ``'test'``: Returns only preprocessed and shuffled test data.
              Returns ``OrderedDict`` mapping normalization methods to
              ``(X_test_ensemble[n_variants, n_test, n_features],)``.

        feature_mask : ndarray of shape (n_original_features,) or None, default=None
            Boolean mask where ``True`` indicates masked (all-NaN) columns in
            the *original* feature space (before ``UniqueFeatureFilter``).  When
            provided, masked columns are dropped from both preprocessed training
            and test data, and feature shuffles are remapped to the reduced
            ``[0, K)`` space.  A transient ``masked_feature_shuffles_``
            attribute is stored for the caller to retrieve the remapped shuffles.

        Returns
        -------
        OrderedDict
            Dictionary mapping normalization methods to data tuples.
        """

        check_is_fitted(self, ["ensemble_configs_"])
        assert mode in ("both", "train", "test"), f"Invalid mode: {mode}"

        # Remap feature shuffles if a feature mask is provided to drop masked columns
        if feature_mask is not None:
            # Map mask from original feature space to filtered space
            filtered_mask = feature_mask[self.unique_filter_.features_to_keep_]
            kept_cols = ~filtered_mask
            # Build old-index -> new-index mapping for shuffle remapping
            idx_map = {}
            new_idx = 0
            for old_idx in range(len(filtered_mask)):
                if kept_cols[old_idx]:
                    idx_map[old_idx] = new_idx
                    new_idx += 1

            # Pre-compute remapped feature shuffles per norm method
            self.masked_feature_shuffles_ = OrderedDict()
            for norm_method, shuffle_configs in self.ensemble_configs_.items():
                remapped = []
                for feat_shuffle, _ in shuffle_configs:
                    remapped.append([idx_map[i] for i in feat_shuffle if i in idx_map])
                self.masked_feature_shuffles_[norm_method] = remapped

        if mode == "train":
            y = self.y_
            data = OrderedDict()
            for norm_method, shuffle_configs in self.ensemble_configs_.items():
                if feature_mask is not None:
                    kept = kept_cols
                else:
                    kept = None
                X_ensemble = []
                y_ensemble = []
                for i, (feat_shuffle, y_pattern) in enumerate(shuffle_configs):
                    if self.cat_random_encode:
                        X_preprocessed = self.member_preprocessors_[norm_method][i].X_transformed_
                    else:
                        X_preprocessed = self.preprocessors_[norm_method].X_transformed_
                    if kept is not None:
                        X_preprocessed = X_preprocessed[:, kept]
                    if feature_mask is not None:
                        feat_shuffle = self.masked_feature_shuffles_[norm_method][i]
                    X_ensemble.append(X_preprocessed[:, feat_shuffle])
                    if self.classification:
                        y_ensemble.append(np.array(y_pattern)[y.astype(int)])
                    else:
                        y_ensemble.append(y)
                data[norm_method] = (np.stack(X_ensemble, axis=0), np.stack(y_ensemble, axis=0))
            return data

        # mode == "test" or "both" requires X
        assert X is not None, "X is required when mode is 'test' or 'both'"
        X = self.unique_filter_.transform(X)

        # SVD feature augmentation (test data)
        if self.svd_augmentor_ is not None:
            X = self.svd_augmentor_.transform(X)
            if not np.isfinite(X).all():
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Fill masked columns with 0.0 so sklearn transformers don't choke on NaN
        if feature_mask is not None:
            X = np.array(X, dtype=np.float64)
            X[:, filtered_mask] = 0.0

        if mode == "test":
            data = OrderedDict()
            for norm_method, shuffle_configs in self.ensemble_configs_.items():
                X_ensemble = []
                for i, (feat_shuffle, _) in enumerate(shuffle_configs):
                    if self.cat_random_encode:
                        mapped_X = self._apply_category_code_mapping(
                            X, self.category_code_mappings_[norm_method][i]
                        )
                        X_test_preprocessed = self.member_preprocessors_[norm_method][i].transform(mapped_X)
                    else:
                        X_test_preprocessed = self.preprocessors_[norm_method].transform(X)
                    if feature_mask is not None:
                        X_test_preprocessed = X_test_preprocessed[:, kept_cols]
                    if feature_mask is not None:
                        feat_shuffle = self.masked_feature_shuffles_[norm_method][i]
                    X_ensemble.append(X_test_preprocessed[:, feat_shuffle])
                data[norm_method] = (np.stack(X_ensemble, axis=0),)
            return data

        # mode == "both"
        y = self.y_
        data = OrderedDict()
        for norm_method, shuffle_configs in self.ensemble_configs_.items():
            preprocessor = self.preprocessors_[norm_method]
            X_ensemble = []
            y_ensemble = []
            for i, (feat_shuffle, y_pattern) in enumerate(shuffle_configs):
                if self.cat_random_encode:
                    member_preprocessor = self.member_preprocessors_[norm_method][i]
                    X_train_pp = member_preprocessor.X_transformed_
                    mapped_X = self._apply_category_code_mapping(
                        X, self.category_code_mappings_[norm_method][i]
                    )
                    X_test_pp = member_preprocessor.transform(mapped_X)
                else:
                    X_train_pp = preprocessor.X_transformed_
                    X_test_pp = preprocessor.transform(X)
                if feature_mask is not None:
                    X_train_pp = X_train_pp[:, kept_cols]
                    X_test_pp = X_test_pp[:, kept_cols]
                X_variant = np.concatenate([X_train_pp, X_test_pp], axis=0)
                if feature_mask is not None:
                    feat_shuffle = self.masked_feature_shuffles_[norm_method][i]
                X_ensemble.append(X_variant[:, feat_shuffle])

                if self.classification:
                    # Apply class shuffle for classification
                    y_ensemble.append(np.array(y_pattern)[y.astype(int)])
                else:
                    y_ensemble.append(y)

            data[norm_method] = (np.stack(X_ensemble, axis=0), np.stack(y_ensemble, axis=0))

        return data


# Copyright (C) 2026 Xiaomi Corporation
# SPDX-License-Identifier: Apache-2.0

# ============================================================
# SVDFeatureAugmentor
# Appends SVD principal components as additional features.
# ============================================================

class SVDFeatureAugmentor(TransformerMixin, BaseEstimator):
    """Append SVD principal components as additional features to the original feature matrix.

    Parameters
    ----------
    n_components : int
        Number of SVD components to keep.
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        col_means = np.nanmean(X, axis=0)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            X = X.copy()
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ < 1e-10] = 1.0
        X_scaled = (X - self.mean_) / self.std_
        X_scaled = np.clip(X_scaled, -10, 10)

        n_comp = min(self.n_components, min(X.shape[0], X.shape[1]))
        try:
            from sklearn.utils.extmath import randomized_svd
            _, _, Vt = randomized_svd(X_scaled, n_components=n_comp, random_state=42)
        except Exception:
            _, _, Vt = np.linalg.svd(X_scaled, full_matrices=False)
            Vt = Vt[:n_comp]

        self.components_ = Vt
        self.n_features_in_ = X.shape[1]
        self.n_components_ = n_comp
        self.n_features_out_ = X.shape[1] + n_comp
        return self

    def transform(self, X):
        check_is_fitted(self, ["components_", "mean_", "std_"])
        X = np.asarray(X, dtype=np.float64)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            X = X.copy()
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                col_means = np.nanmean(X, axis=0)
            col_means = np.where(np.isnan(col_means), self.mean_, col_means)
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        X_scaled = (X - self.mean_) / self.std_
        X_scaled = np.clip(X_scaled, -10, 10)
        X_svd = X_scaled @ self.components_.T

        if not np.isfinite(X_svd).all():
            X_svd = np.nan_to_num(X_svd, nan=0.0, posinf=0.0, neginf=0.0)

        return np.concatenate([X, X_svd], axis=1)


# ============================================================
# GaussianRankNormalizer
# Maps each numeric column to a Gaussian distribution via rank-based
# probability estimation and the inverse error function.
# Ported from MiTabEnhancedClassifier.
# ============================================================

class GaussianRankNormalizer(TransformerMixin, BaseEstimator):
    """Gaussian rank normalization for numerical features.

    Maps each numeric column to a Gaussian distribution via rank-based
    probability estimation and the inverse error function. Small Gaussian
    noise is added to break ties before ranking.

    Parameters
    ----------
    random_state : int or None
        Seed for the noise generator.
    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        self.n_features_ = n_features
        self.references_ = []
        self.noise_scales_ = []
        self.noise_per_col_ = []

        for col in range(n_features):
            observed = X[:, col]
            mask = ~np.isnan(observed)
            vals = observed[mask]

            if vals.size <= 1:
                self.references_.append(np.array([], dtype=np.float64))
                self.noise_scales_.append(0.0)
                self.noise_per_col_.append(np.zeros(n_samples, dtype=np.float64))
                continue

            q1, q3 = np.percentile(vals, [25, 75])
            iqr = max(float(q3 - q1), 1e-6)
            noise_scale = 1e-4 * iqr

            noise = rng.standard_normal(n_samples) * noise_scale
            perturbed = observed[mask] + noise[mask]
            reference = np.sort(perturbed)

            self.references_.append(reference)
            self.noise_scales_.append(noise_scale)
            self.noise_per_col_.append(noise)

        return self

    def transform(self, X):
        from scipy.special import erfinv as _erfinv

        X = np.asarray(X, dtype=np.float64)
        X_out = X.copy()

        for col in range(self.n_features_):
            ref = self.references_[col]
            if ref.size == 0:
                continue

            noise_scale = self.noise_scales_[col]
            noise = self.noise_per_col_[col]
            n_ref = ref.size

            mask = ~np.isnan(X[:, col])
            perturbed = X[mask, col] + noise[:mask.sum()] * noise_scale
            ranks = np.searchsorted(ref, perturbed, side='left')
            probs = (ranks.astype(np.float64) + 0.5) / (n_ref + 1)
            probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
            X_out[mask, col] = _erfinv(2.0 * probs - 1.0) * np.sqrt(2.0)

        if not np.all(np.isfinite(X_out)):
            X_out = np.nan_to_num(X_out, nan=0.0, posinf=0.0, neginf=0.0)
        return X_out


# ============================================================
# GaussianRankGenerator
# Wrapper that produces Gaussian-rank-transformed ensemble data.
# ============================================================

class GaussianRankGenerator:
    """Wrapper that produces Gaussian-rank-transformed ensemble data.

    Implements the ``transform(X, mode)`` interface expected by
    ``_collect_candidate_probs`` so that the Gaussian rank candidate group
    can be collected alongside the other groups without special-casing.

    Parameters
    ----------
    n_estimators : int
        Number of ensemble members.
    feat_shuffle_method : str
        Feature permutation method (``"random"`` recommended).
    random_state : int or None
        Base seed for reproducibility.
    """

    def __init__(self, n_estimators, feat_shuffle_method="random", random_state=None):
        self.n_estimators = n_estimators
        self.feat_shuffle_method = feat_shuffle_method
        self.random_state = random_state

    def fit(self, X, y, frozen_unique_filter=None):
        if frozen_unique_filter is not None:
            self.unique_filter_ = frozen_unique_filter
            X = frozen_unique_filter.transform(X)
        else:
            self.unique_filter_ = UniqueFeatureFilter()
            X = self.unique_filter_.fit_transform(X)

        self.X_ = X.copy()
        self.y_ = y.copy()
        n_train, n_features = X.shape
        n_estimators = self.n_estimators

        # Detect numeric columns (unique values > 10) vs categorical.
        is_numeric = np.array(
            [len(np.unique(X[~np.isnan(X[:, c]), c])) > 10 for c in range(n_features)],
            dtype=bool,
        )
        self.is_numeric_ = is_numeric

        # Fit one GaussianRankNormalizer per estimator (different noise).
        self.normalizers_ = []
        for i in range(n_estimators):
            norm = GaussianRankNormalizer(random_state=self.random_state + i)
            norm.fit(X)
            self.normalizers_.append(norm)

        # Generate random feature permutations (one per estimator).
        self.feature_permutations_ = []
        rng = np.random.default_rng(self.random_state)
        for i in range(n_estimators):
            self.feature_permutations_.append(rng.permutation(n_features))

        self.ensemble_configs_ = {"gaussian_rank": list(range(n_estimators))}
        return self

    def transform(self, X_test=None, mode="test"):
        X_train = self.X_
        y_train = self.y_
        n_train = X_train.shape[0]

        views = []
        for i in range(self.n_estimators):
            X_tr_i = self.normalizers_[i].transform(X_train)
            X_tr_i = X_tr_i[:, self.feature_permutations_[i]]

            if X_test is not None:
                X_te_i = self.normalizers_[i].transform(X_test)
                X_te_i = X_te_i[:, self.feature_permutations_[i]]

            if mode == "train":
                views.append(X_tr_i)
            elif mode == "test":
                views.append(X_te_i)
            else:  # "both"
                views.append(np.concatenate([X_tr_i, X_te_i], axis=0))

        Xs = np.stack(views, axis=0)
        ys = np.stack([y_train] * self.n_estimators, axis=0)

        if mode == "test":
            return OrderedDict({"gaussian_rank": (Xs,)})
        return OrderedDict({"gaussian_rank": (Xs, ys)})


# ============================================================
# PCADecorrelator (simplified, for enhanced classifier)
# When mean_abs_corr > threshold, replaces features with PCA rotations.
# ============================================================

class PCADecorrelator(TransformerMixin, BaseEstimator):
    """Replaces features with PCA rotations when mean_abs_corr > threshold.

    Parameters
    ----------
    corr_threshold : float, default=0.4
        Threshold for mean absolute correlation to trigger PCA.
    """

    def __init__(self, corr_threshold: float = 0.4):
        self.corr_threshold = corr_threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0)
        stds = X.std(axis=0)
        valid = stds > 1e-10
        self.active_ = False
        self.valid_mask_ = valid
        if valid.sum() > 2:
            corr = np.abs(np.corrcoef(X[:, valid].T))
            np.fill_diagonal(corr, 0)
            self.mean_corr_ = float(corr.mean())
            if self.mean_corr_ > self.corr_threshold:
                self.active_ = True
                from sklearn.decomposition import PCA
                n_comp = int(valid.sum())
                self.pca_ = PCA(n_components=n_comp, random_state=42)
                self.pca_.fit(X[:, valid])
        return self

    def transform(self, X):
        if not self.active_:
            return X
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0)
        X_out = X.copy()
        X_out[:, self.valid_mask_] = self.pca_.transform(X[:, self.valid_mask_])
        if not np.isfinite(X_out).all():
            X_out = np.nan_to_num(X_out, nan=0.0, posinf=0.0, neginf=0.0)
        return X_out


# ============================================================
# InteractionAugmentor (simplified, for enhanced classifier)
# Appends pairwise product features when n_features <= max_features.
# ============================================================

class InteractionAugmentor(TransformerMixin, BaseEstimator):
    """Appends pairwise product features when n_features <= max_features.

    Parameters
    ----------
    max_features : int, default=5
        Maximum number of features to trigger augmentation.
    """

    def __init__(self, max_features: int = 5):
        self.max_features = max_features

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        self.active_ = (X.shape[1] <= self.max_features) and (X.shape[1] >= 2)
        if self.active_:
            n = X.shape[1]
            self.pairs_ = [(i, j) for i in range(n) for j in range(i + 1, n)]
            self.n_features_out_ = n + len(self.pairs_)
        else:
            self.n_features_out_ = X.shape[1]
        return self

    def transform(self, X):
        if not self.active_:
            return X
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0)
        interactions = [
            (X[:, i] * X[:, j]).reshape(-1, 1)
            for i, j in self.pairs_
        ]
        X_aug = np.hstack([X] + interactions)
        if not np.isfinite(X_aug).all():
            X_aug = np.nan_to_num(X_aug, nan=0.0, posinf=0.0, neginf=0.0)
        return X_aug
