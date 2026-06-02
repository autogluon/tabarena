from __future__ import annotations

import math


def get_split_idx(
    fold: int = 0,
    repeat: int = 0,
    sample: int = 0,
    n_folds: int = 1,
    n_repeats: int = 1,
    n_samples: int = 1,
) -> int:
    assert fold < n_folds
    assert repeat < n_repeats
    assert sample < n_samples
    split_idx = n_folds * n_samples * repeat + n_samples * fold + sample
    return split_idx


def get_split_vals_from_split_idx(
    split_idx: int,
    n_folds: int = 1,
    n_repeats: int = 1,
    n_samples: int = 1,
) -> tuple[int, int, int]:
    repeat = math.floor(split_idx / (n_folds * n_samples))
    remainder = split_idx % (n_folds * n_samples)
    fold = math.floor(remainder / n_samples)
    sample = remainder % n_samples

    assert fold < n_folds
    assert repeat < n_repeats
    assert sample < n_samples
    return repeat, fold, sample
