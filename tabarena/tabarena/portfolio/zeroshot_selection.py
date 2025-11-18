from __future__ import annotations

import pandas as pd
import numpy as np


def zeroshot_configs(
    val_scores: np.array,
    output_size: int,
    weights: list[int] | None = None,
) -> list[int]:
    """
    :param val_scores: a tensor with shape (n_task, n_configs) that contains evaluations to consider
    :param output_size: number of configurations to return, in some case where no configuration helps anymore,
    the portfolio can have smaller length.
    :return: a list of index configuration for the portfolio where all indices are in [0, `n_configs` - 1]
    """
    assert val_scores.ndim == 2

    df_val_scores = pd.DataFrame(val_scores)
    ranks = pd.DataFrame(df_val_scores).rank(axis=1)

    if weights is not None:
        ranks = ranks.multiply(weights, axis=0)

    res = []
    best_mean = None
    for _ in range(output_size):
        # Select greedy-best configuration considering all others
        if ranks.empty:
            # Nothing more to add
            break

        cur_ranks_mean = ranks.mean(axis=0)
        best_idx = cur_ranks_mean.idxmin()
        cur_best_mean = cur_ranks_mean[best_idx]

        if best_mean is not None and cur_best_mean == best_mean:
            # No improvement
            break
        best_mean = cur_best_mean

        # Update ranks for choosing each configuration considering the previously chosen ones
        ranks.clip(upper=ranks[best_idx], axis=0, inplace=True)
        # Drop the chosen configuration as a future candidate
        df_val_scores.drop(columns=best_idx, inplace=True)
        res.append(best_idx)
    return res
