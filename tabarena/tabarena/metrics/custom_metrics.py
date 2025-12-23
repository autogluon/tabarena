from __future__ import annotations

import numpy as np
import pandas as pd
from autogluon.core.constants import BINARY
from autogluon.core.metrics import METRICS, make_scorer


def amex_metric(
    target: np.ndarray | pd.DataFrame, preds: np.ndarray | pd.DataFrame
) -> float:
    """Optimized AMEX Competition Metric.

    Ref: https://www.kaggle.com/competitions/amex-default-prediction/discussion/328020
    """
    if isinstance(target, pd.DataFrame):
        target = target.to_numpy()
    if isinstance(preds, pd.DataFrame):
        preds = preds.to_numpy()

    indices = np.argsort(preds)[::-1]
    preds, target = preds[indices], target[indices]

    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_mask = cum_norm_weight <= 0.04
    d = np.sum(target[four_pct_mask]) / np.sum(target)

    weighted_target = target * weight
    lorentz = (weighted_target / weighted_target.sum()).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    n_pos = np.sum(target)
    n_neg = target.shape[0] - n_pos
    gini_max = 10 * n_neg * (n_pos + 20 * n_neg - 19) / (n_pos + 20 * n_neg)

    g = gini / gini_max
    return 0.5 * (g + d)


ag_amex_metric = make_scorer(
    name="amex_metric",
    score_func=amex_metric,
    greater_is_better=True,
    optimum=1,
    needs_proba=True,
)
METRICS[BINARY][ag_amex_metric.name] = ag_amex_metric
