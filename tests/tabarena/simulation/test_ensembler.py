from __future__ import annotations

import numpy as np
import pytest
from autogluon.core.metrics import get_metric
from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection

from tabarena.simulation.ensemble import (
    GreedyEnsembler,
    LegacyEnsemblerAdapter,
)


def _make_binary_task(n_models=8, n_samples=500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.random(n_samples) < 0.4
    preds = np.stack(
        [np.clip(y * rng.uniform(0.3, 0.8) + rng.normal(0, 0.4, n_samples), 0, 1) for _ in range(n_models)]
    ).astype(np.float32)
    return y.astype(np.bool_), preds


def _make_regression_task(n_models=8, n_samples=500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 1, n_samples)
    preds = np.stack([y + rng.normal(0, s, n_samples) for s in rng.uniform(0.3, 1.0, n_models)]).astype(np.float32)
    return y, preds


@pytest.mark.parametrize("problem_type", ["binary", "regression"])
def test_greedy_ensembler_matches_ensemble_selection(problem_type):
    """GreedyEnsembler must produce identical weights and predictions to using AutoGluon's
    EnsembleSelection directly (the historical code path).
    """
    if problem_type == "binary":
        y, preds = _make_binary_task()
        metric = get_metric(metric="roc_auc", problem_type=problem_type)
    else:
        y, preds = _make_regression_task()
        metric = get_metric(metric="rmse", problem_type=problem_type)

    reference = EnsembleSelection(
        ensemble_size=20, problem_type=problem_type, metric=metric, random_state=np.random.RandomState(0)
    )
    reference.fit(predictions=list(preds), labels=y)

    ensembler = GreedyEnsembler(
        problem_type=problem_type, metric=metric, ensemble_size=20, random_state=np.random.RandomState(0)
    )
    ensembler.fit(predictions=preds, labels=y)

    np.testing.assert_array_equal(ensembler.model_weights(), reference.weights_)
    np.testing.assert_array_equal(ensembler.predict_proba(preds), reference.predict_proba(preds))
    if problem_type == "binary":
        np.testing.assert_array_equal(ensembler.predict(preds, problem_type=problem_type), reference.predict(preds))
    np.testing.assert_array_equal(ensembler.models_used(), reference.weights_ != 0)


def test_legacy_adapter_matches_ensemble_selection():
    """LegacyEnsemblerAdapter wraps a pre-interface class without changing behavior."""
    y, preds = _make_binary_task(seed=1)
    metric = get_metric(metric="roc_auc", problem_type="binary")

    reference = EnsembleSelection(
        ensemble_size=20, problem_type="binary", metric=metric, random_state=np.random.RandomState(0)
    )
    reference.fit(predictions=list(preds), labels=y)

    adapted = LegacyEnsemblerAdapter(
        problem_type="binary",
        metric=metric,
        ensemble_cls=EnsembleSelection,
        ensemble_kwargs={"ensemble_size": 20, "random_state": np.random.RandomState(0)},
    )
    adapted.fit(predictions=preds, labels=y)

    np.testing.assert_array_equal(adapted.model_weights(), reference.weights_)
    np.testing.assert_array_equal(adapted.predict_proba(preds), reference.predict_proba(preds))


def test_legacy_adapter_predict_problem_type_override_restores():
    """The problem_type override in predict is applied for the call and restored after."""
    y, preds = _make_binary_task(seed=2)
    metric = get_metric(metric="roc_auc", problem_type="binary")
    adapted = LegacyEnsemblerAdapter(
        problem_type="binary",
        metric=metric,
        ensemble_cls=EnsembleSelection,
        ensemble_kwargs={"ensemble_size": 5},
    )
    adapted.fit(predictions=preds, labels=y)
    adapted.predict(preds, problem_type="binary")
    assert adapted._ensemble.problem_type == "binary"


def test_fit_does_not_mutate_caller_predictions():
    """subsample_size triggers in-place mutation inside AutoGluon's EnsembleSelection;
    the adapters must shield the caller's arrays/list from it.
    """
    y, preds = _make_binary_task(n_samples=500, seed=3)
    metric = get_metric(metric="roc_auc", problem_type="binary")
    preds_list = list(preds)
    preds_list_copy = [p.copy() for p in preds_list]

    ensembler = GreedyEnsembler(problem_type="binary", metric=metric, ensemble_size=5, subsample_size=100)
    ensembler.fit(predictions=preds_list, labels=y)

    assert all(np.array_equal(a, b) for a, b in zip(preds_list, preds_list_copy, strict=False)), (
        "GreedyEnsembler.fit mutated the caller's predictions"
    )
