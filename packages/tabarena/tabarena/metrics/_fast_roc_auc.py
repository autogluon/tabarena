from typing import Tuple

import numpy as np

from autogluon.core.metrics import make_scorer

from ._roc_auc_cpp import CppAuc


# TODO: Consider having `setup.py` automatically compile the C++ code to avoid having to manually do so.
# Score functions that need decision values
# Requires compiled C++ code, refer to `_roc_auc_cpp/README.md` for details
fast_roc_auc_cpp = make_scorer('roc_auc',
                               CppAuc().roc_auc_score,
                               greater_is_better=True,
                               needs_threshold=True)


def _preprocess_bulk(y_true: np.ndarray, y_pred_bulk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return y_true.astype(np.bool_), y_pred_bulk.astype(np.float32)


fast_roc_auc_cpp.preprocess_bulk = _preprocess_bulk
