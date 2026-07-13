from __future__ import annotations

import numpy as np
from autogluon.core.metrics import _ThresholdScorer

from ._cpp_metrics import CppMetrics


class _FastThresholdScorer(_ThresholdScorer):
    """`_ThresholdScorer` that skips sklearn's `type_of_target` validation when
    `y_true` is a bool array.

    `type_of_target` fully sorts `y_true` on every call, which dominates the cost of
    scoring in ensemble simulation, where the same labels are scored thousands of
    times. A bool `y_true` (which `preprocess_bulk` guarantees) is binary by
    construction, so the validation is redundant there; any other dtype falls back
    to the standard validation.
    """

    def _preprocess(self, y_true, y_pred, **kwargs):
        if isinstance(y_true, np.ndarray) and y_true.dtype == np.bool_:
            if isinstance(y_pred, list):
                y_pred = np.array(y_pred)
            return y_true, y_pred, kwargs
        return super()._preprocess(y_true, y_pred, **kwargs)


# TODO: Consider having `setup.py` automatically compile the C++ code to avoid having to manually do so.
# Score functions that need decision values
# Requires compiled C++ code, refer to `_cpp_metrics/README.md` for details
fast_roc_auc_cpp = _FastThresholdScorer(name="roc_auc", score_func=CppMetrics().roc_auc_score, optimum=1, sign=1)


def _preprocess_bulk(y_true: np.ndarray, y_pred_bulk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return y_true.astype(np.bool_), y_pred_bulk.astype(np.float32)


fast_roc_auc_cpp.preprocess_bulk = _preprocess_bulk
