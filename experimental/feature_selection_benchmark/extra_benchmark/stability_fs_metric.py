"""Feature selection stability benchmark: measures consistency across bootstrap repeats."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def get_dataset_for_stability(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    rng: np.random.Generator,
    **kwargs,
) -> tuple[pd.DataFrame, pd.Series]:
    """Args:
    X: original feature dataset
    y: original target variable
    rng: NumPy random Generator for reproducibility
    kwargs: common interface support, ignores other input arguments.
    """
    sample_idx = rng.choice(len(X), size=len(X), replace=True)

    return X.iloc[sample_idx].reset_index(drop=True), y.iloc[sample_idx].reset_index(drop=True)
