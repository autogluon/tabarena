"""How balanced a task's target is, and the thresholds that split tasks into balanced and imbalanced.

Classification tasks are judged on the imbalance ratio, the size of the largest class over the
smallest one. Regression tasks have no classes, so they are judged on the skewness of the target
instead: a long-tailed target puts the same pressure on a model as a rare class does. The
thresholds are module constants so the ``"balanced"`` / ``"imbalanced"`` subset predicates of every
arena context read the same line.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

#: A classification target is imbalanced when its largest class holds at least this many times
#: the samples of its smallest one. For a binary target this is a minority share of about 9% or
#: less, so only the clearly skewed datasets (fraud, failures, rare diseases) count as imbalanced
#: and the moderately skewed 20/80 ones stay in the balanced half.
IMBALANCE_RATIO_THRESHOLD = 10.0
#: A regression target is imbalanced when the absolute Fisher skewness of its values reaches this,
#: the usual bar for a "highly skewed" distribution.
TARGET_SKEWNESS_THRESHOLD = 1.0


def target_distribution_stats(
    target: Sequence | pd.Series | np.ndarray,
    *,
    is_classification: bool,
) -> tuple[float | None, float | None, float | None]:
    """``(minority_fraction, imbalance_ratio, skewness)`` of one task's target column.

    The first two are set for classification and ``None`` for regression; skewness is the other
    way round. Missing target values are ignored. A degenerate target (a single class, a constant
    regression target) yields ``None`` for the statistics it cannot define.
    """
    values = pd.Series(target).dropna()
    if values.empty:
        return None, None, None
    if is_classification:
        counts = values.value_counts()
        if len(counts) < 2:
            return None, None, None
        minority = int(counts.min())
        return float(minority / int(counts.sum())), float(int(counts.max()) / minority), None
    numbers = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(numbers) < 3 or float(numbers.std()) == 0.0:
        return None, None, None
    centered = numbers - numbers.mean()
    skewness = float(np.mean(centered**3) / np.mean(centered**2) ** 1.5)
    return None, None, skewness if math.isfinite(skewness) else None


def is_target_imbalanced(
    *,
    problem_type: str,
    imbalance_ratio: float | None,
    skewness: float | None,
) -> bool | None:
    """Whether a task counts as imbalanced under the module thresholds, or ``None`` when unknown.

    Classification (``binary`` / ``multiclass``) reads ``imbalance_ratio``, regression reads
    ``skewness``; a task whose statistic was never computed (metadata predating these fields)
    belongs to neither subset rather than being guessed into one.
    """
    if problem_type == "regression":
        if skewness is None or pd.isna(skewness):
            return None
        return bool(abs(float(skewness)) >= TARGET_SKEWNESS_THRESHOLD)
    if imbalance_ratio is None or pd.isna(imbalance_ratio):
        return None
    return bool(float(imbalance_ratio) >= IMBALANCE_RATIO_THRESHOLD)
