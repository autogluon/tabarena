"""Feature selection validity benchmark: measures ability to distinguish real from noise features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_dataset_for_validity(
    *,
    X: pd.DataFrame,
    noise: float,
    noise_type: str,
    rng: np.random.Generator,
    **kwargs, # noqa: ARG001
) -> pd.DataFrame:
    """Create the new version of the dataset for validity assessment.

    Args:
        X: original feature dataset
        noise: Proportion of noise features (relative to original number of features) to add
        noise_type: Type of noise to add
        rng: NumPy random Generator for reproducibility
        kwargs: common interface support, ignores other input arguments.
    """
    n_noise = int(len(X.columns) * noise)

    return add_noise(X=X, n_noise=n_noise, noise_type=noise_type, rng=rng)


def add_noise(*, X: pd.DataFrame, n_noise: int, rng: np.random.Generator, noise_type: str = "gaussian") -> pd.DataFrame:
    """Add noisy synthetic features to a dataset and shuffle all features.

    Args:
        X: The input feature matrix.
        n_noise: The number of noisy features to add.
        rng: NumPy random Generator for reproducibility.
        noise_type: Type of noise for numeric features ("gaussian" or "uniform").

    Returns:
        Tuple of (augmented_shuffled_DataFrame, mask_dict) where mask maps
        feature names to True if they are original features.
    """
    noise_cols = {}
    n_samples, n_features = X.shape

    for i in range(n_noise):
        col_idx = rng.integers(0, n_features)
        sample_col = X.iloc[:, col_idx]

        if sample_col.dtype.kind in "biufc":
            if noise_type == "gaussian":
                noise = rng.normal(sample_col.mean(), sample_col.std(), n_samples)
            else:
                noise = rng.uniform(sample_col.min(), sample_col.max(), n_samples)
        elif sample_col.dtype.kind == "O":
            unique_vals = sample_col.dropna().unique()
            if len(unique_vals) > 1:
                probs = sample_col.value_counts(normalize=True).to_numpy()
                noise = rng.choice(unique_vals, n_samples, p=probs)
            else:
                noise = np.full(n_samples, unique_vals[0])
        else:
            noise = rng.normal(0, 1, n_samples)

        noise_cols[f"__noise_feature_{i}__"] = noise

    noise_df = pd.DataFrame(noise_cols)
    return pd.concat([X, noise_df], axis=1)