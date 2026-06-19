from __future__ import annotations

import numpy as np
import pytest
import sklearn
from autogluon.core.metrics import log_loss, roc_auc

from tabarena.metrics import _fast_log_loss, _fast_roc_auc
from tabarena.metrics.bench_utils import (
    generate_y_true_and_y_pred_binary,
    generate_y_true_and_y_pred_proba,
    generate_y_true_and_y_pred_proba_bulk,
)


@pytest.mark.parametrize(
    ("y_true", "y_pred"), [([0, 2, 1, 1], [[0.1, 0.2, 0.7], [0.2, 0.1, 0.7], [0.3, 0.4, 0.3], [0.01, 0.9, 0.09]])]
)
def test_fast_log_loss(y_true, y_pred):
    """Ensure fast_log_loss produces equivalent scores to AutoGluon and Scikit-Learn log_loss implementations."""
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.float32)

    assert_fast_log_loss_equivalence(y_true=y_true, y_pred=y_pred, num_classes=y_pred.shape[1], rtol=1e-6)


def assert_fast_log_loss_equivalence(y_true, y_pred, num_classes, rtol=1e-7):
    ag_loss = log_loss(y_true, y_pred)
    sk_loss = -sklearn.metrics.log_loss(y_true, y_pred, labels=list(range(num_classes)))
    np.testing.assert_allclose(ag_loss, sk_loss, rtol=rtol)

    y_pred_opt = _fast_log_loss.extract_true_class_prob(y_true, y_pred)
    fast_loss = _fast_log_loss.fast_log_loss(y_true, y_pred_opt)
    fast_loss_end_to_end = _fast_log_loss.fast_log_loss_end_to_end(y_true, y_pred)

    np.testing.assert_allclose(ag_loss, fast_loss, rtol=rtol)
    np.testing.assert_allclose(ag_loss, fast_loss_end_to_end, rtol=rtol)


def assert_fast_roc_auc_equivalence(y_true, y_pred, rtol=1e-7):
    ag_score = roc_auc(y_true, y_pred)
    sk_score = sklearn.metrics.roc_auc_score(y_true, y_pred)
    np.testing.assert_allclose(ag_score, sk_score, rtol=rtol)

    fast_score = _fast_roc_auc.fast_roc_auc_cpp(y_true, y_pred)
    np.testing.assert_allclose(ag_score, fast_score, rtol=rtol)


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([0, 1, 0, 0], [0, 0, 0, 0]),
        ([0, 1, 0, 0], [1, 1, 1, 1]),
        ([0, 1, 0, 0], [0, 1, 0, 1]),
        ([0, 1, 0, 0], [0, 1, 0, 0]),
        ([0, 1, 0, 0], [1, 0, 1, 1]),
        ([0, 1, 0, 0], [0.6, 0.5, 0.4, 0.3]),
        ([0, 1, 0, 0], [0.5, 0.5, 0.6, 0.6]),
        ([0, 1, 0, 0], [0.6, 0.6, 0.6, 0.6]),
        ([1, 0, 1, 1], [0.6, 0.6, 0.6, 0.6]),
    ],
)
def test_fast_roc_auc_ties(y_true, y_pred):
    """Ensure fast_roc_auc produces equivalent scores to AutoGluon and Scikit-Learn roc_auc implementations
    when ties exist.
    """
    y_true = np.array(y_true, dtype=np.bool_)
    y_pred = np.array(y_pred, dtype=np.float32)
    assert_fast_roc_auc_equivalence(y_true=y_true, y_pred=y_pred)


@pytest.mark.parametrize(
    "num_samples",
    [
        100,
        1000,
        10000,
        100000,
    ],
)
def test_fast_roc_auc_large(num_samples):
    y_true, y_pred = generate_y_true_and_y_pred_binary(num_samples=num_samples)
    assert_fast_roc_auc_equivalence(y_true=y_true, y_pred=y_pred)


@pytest.mark.parametrize(
    ("num_samples", "num_classes"),
    [
        (1, 2),
        (1, 10),
        (1000, 2),
        (1000, 10),
        (10000, 2),
        (10000, 100),
    ],
)
def test_fast_log_loss_large(num_samples, num_classes):
    """Ensure fast_log_loss produces equivalent scores to AutoGluon and Scikit-Learn log_loss implementations
    across various data dimensions.
    """
    y_true, y_pred = generate_y_true_and_y_pred_proba(num_samples=num_samples, num_classes=num_classes)
    assert_fast_log_loss_equivalence(y_true=y_true, y_pred=y_pred, num_classes=num_classes, rtol=1e-4)


@pytest.mark.parametrize(
    ("num_configs", "num_samples", "num_classes"),
    [
        (1, 1, 2),
        (2, 1, 2),
        (10, 1, 2),
        (10, 1, 10),
        (1, 1000, 2),
        (10, 1000, 10),
        (100, 1000, 10),
        (10, 10000, 2),
        (10, 10000, 100),
    ],
)
def test_fast_log_loss_bulk(num_configs, num_samples, num_classes):
    """Ensure fast_log_loss produces equivalent scores to AutoGluon and Scikit-Learn log_loss implementations
    in the scenario where we have multiple configs and potentially weight them.
    This verifies we can perform operations such as config weighting
    to the optimized predictions and still obtain equivalent log_loss results.
    """
    y_true, y_pred_bulk = generate_y_true_and_y_pred_proba_bulk(
        num_configs=num_configs,
        num_samples=num_samples,
        num_classes=num_classes,
    )
    rtol = 1e-6

    for i in range(num_configs):
        y_pred = y_pred_bulk[i]
        assert_fast_log_loss_equivalence(y_true=y_true, y_pred=y_pred, num_classes=num_classes, rtol=rtol)

    config_weights = np.random.rand(num_configs)
    config_weights /= np.sum(config_weights)
    np.testing.assert_allclose(np.sum(config_weights), 1)

    y_pred_bulk_weighted = [pred * weight for pred, weight in zip(y_pred_bulk, config_weights, strict=False)]
    y_pred_ensemble = np.sum(y_pred_bulk_weighted, axis=0)
    assert_fast_log_loss_equivalence(y_true=y_true, y_pred=y_pred_ensemble, num_classes=num_classes, rtol=rtol)

    y_pred_bulk_opt = _fast_log_loss.extract_true_class_prob_bulk(y_true, y_pred_bulk)
    y_pred_bulk_opt_weighted = [pred * weight for pred, weight in zip(y_pred_bulk_opt, config_weights, strict=False)]

    y_pred_opt_ensemble = np.sum(y_pred_bulk_opt_weighted, axis=0)
    fast_loss_ensemble = _fast_log_loss.fast_log_loss(y_true, y_pred_opt_ensemble)
    ag_loss_ensemble = log_loss(y_true, y_pred_ensemble)

    np.testing.assert_allclose(ag_loss_ensemble, fast_loss_ensemble, rtol=rtol)
