"""Regression tests for TabM's static memory estimate.

The estimate is the number AutoGluon budgets parallel bagging folds with; GPU
benchmark runs compare it against a VRAM-sized memory limit
(``fake_memory_for_estimates``). The calibration case is a real OOM: TabM config
r117 (pwl embeddings with 127 bins, d_block=496, k=32, auto batch size 512) on
APSFailure (50666 rows, 169 numerical + 1 categorical feature) peaked at
~12.2 GiB VRAM per bagging fold, while the estimate reported 7.07 GB — so 8
folds were co-scheduled on a 95 GiB GPU and every fit crashed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabarena.models.tabm.model import TabMModel

# Hyperparameters of TabM config r117 (the OOM case described above).
R117 = {
    "amp": False,
    "arch_type": "tabm-mini",
    "batch_size": "auto",
    "d_block": 496,
    "d_embedding": 28,
    "dropout": 0.0,
    "gradient_clipping_norm": 1.0,
    "lr": 0.0004037824650511316,
    "n_blocks": 3,
    "num_emb_n_bins": 127,
    "num_emb_type": "pwl",
    "patience": 16,
    "share_training_batches": False,
    "tabm_k": 32,
    "weight_decay": 0.0,
}

OBSERVED_PEAK_BYTES = 12.2 * 1024**3  # per-fold VRAM peak observed in the crashed run


def _aps_failure_like() -> pd.DataFrame:
    """Training data with APSFailure's shape and dtypes (values are irrelevant)."""
    X = pd.DataFrame(np.zeros((50_666, 169), dtype=np.float32))
    X.columns = [f"num_{i}" for i in range(169)]
    X["cat"] = pd.Categorical(["a"] * len(X))
    return X


def _estimate(hyperparameters: dict) -> int:
    return TabMModel.estimate_memory_usage_static(
        X=_aps_failure_like(),
        problem_type="binary",
        num_classes=2,
        hyperparameters=hyperparameters,
    )


def test_estimate_covers_observed_vram_peak():
    est = _estimate(R117)
    assert est >= OBSERVED_PEAK_BYTES
    # ... but not so pessimistic that a single fold would be rejected on a 96 GB budget.
    assert est <= 2 * OBSERVED_PEAK_BYTES


def test_auto_eval_batch_resolves_to_train_batch():
    """An "auto" eval batch must resolve like TabMImplementation: eval batch = train batch.

    APSFailure's shape resolves the auto train batch size to 512.
    """
    assert _estimate(R117) == _estimate({**R117, "eval_batch_size": 512})
