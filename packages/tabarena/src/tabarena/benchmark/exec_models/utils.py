from __future__ import annotations

import numpy as np
import pandas as pd


def _make_perm(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (perm, inv_perm) for length n, using a deterministic RNG seed."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(n)
    return perm, inv_perm


def _apply_inv_perm(obj, inv_perm: np.ndarray, index: pd.Index | None = None):
    """Inverse-permute predictions while preserving type (Series/DataFrame/ndarray)."""
    if isinstance(obj, pd.Series):
        vals = obj.to_numpy()[inv_perm]
        return pd.Series(vals, index=index, name=obj.name)
    if isinstance(obj, pd.DataFrame):
        vals = obj.to_numpy()[inv_perm, :]
        return pd.DataFrame(vals, index=index, columns=obj.columns)
    # Fallback: numpy array or array-like
    arr = np.asarray(obj)
    return arr[inv_perm]
