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


# -------------------------
# TaskEvaluator integration (stage 2)
# -------------------------
def _run_task_evaluator(ensembler_cls, ensembler_kwargs, *, problem_type, eval_metric, fit_eval_metric, y, preds):
    from tabarena.simulation.ensemble_selection_config_scorer import TaskEvaluator

    evaluator = TaskEvaluator(
        ensembler_cls=ensembler_cls,
        ensembler_kwargs=ensembler_kwargs,
        eval_metric=eval_metric,
        fit_eval_metric=fit_eval_metric,
        problem_type=problem_type,
    )
    results, ensemble = evaluator.run(
        pred_train=preds,
        y_train=y,
        pred_test=preds,
        y_test=y,
        return_metric_error_val=True,
        pred_val=preds,
        y_val=y,
    )
    return results, ensemble


def _task_for_metric(metric_name):
    if metric_name == "roc_auc":
        from tabarena.metrics._fast_roc_auc import fast_roc_auc_cpp

        y, preds = _make_binary_task()
        return "binary", fast_roc_auc_cpp, y, preds
    if metric_name == "log_loss":
        from tabarena.metrics._fast_log_loss import fast_log_loss

        rng = np.random.default_rng(0)
        n_samples, n_classes, n_models = 400, 3, 6
        y = rng.integers(0, n_classes, n_samples)
        preds = rng.random((n_models, n_samples, n_classes))
        preds /= preds.sum(axis=2, keepdims=True)
        return "multiclass", fast_log_loss, y, preds.astype(np.float32)
    if metric_name == "rmse":
        y, preds = _make_regression_task()
        return "regression", get_metric(metric="rmse", problem_type="regression"), y, preds
    raise ValueError(metric_name)


@pytest.mark.parametrize("metric_name", ["roc_auc", "log_loss", "rmse"])
def test_task_evaluator_new_interface_matches_legacy(metric_name):
    """TaskEvaluator with the default GreedyEnsembler must produce identical results to
    the historical ensemble_method=EnsembleSelection path (resolved via EnsembleScorer),
    across all three metric regimes (incl. the metric-preprocessed log_loss space and
    the needs_pred rmse path).
    """
    from tabarena.simulation.ensemble_selection_config_scorer import EnsembleScorer

    problem_type, metric, y, preds = _task_for_metric(metric_name)

    legacy_cls, legacy_kwargs = EnsembleScorer._resolve_ensembler(
        ensembler_cls=None,
        ensembler_kwargs=None,
        ensemble_method=EnsembleSelection,
        ensemble_method_kwargs={"ensemble_size": 20},
    )
    new_cls, new_kwargs = EnsembleScorer._resolve_ensembler(
        ensembler_cls=None, ensembler_kwargs={"ensemble_size": 20}, ensemble_method=None, ensemble_method_kwargs=None
    )
    assert new_cls is GreedyEnsembler

    results_legacy, _ = _run_task_evaluator(
        legacy_cls,
        legacy_kwargs,
        problem_type=problem_type,
        eval_metric=metric,
        fit_eval_metric=metric,
        y=y,
        preds=preds,
    )
    results_new, _ = _run_task_evaluator(
        new_cls, new_kwargs, problem_type=problem_type, eval_metric=metric, fit_eval_metric=metric, y=y, preds=preds
    )

    assert results_legacy["metric_error"] == results_new["metric_error"]
    assert results_legacy["metric_error_val"] == results_new["metric_error_val"]
    np.testing.assert_array_equal(results_legacy["ensemble_weights"], results_new["ensemble_weights"])
    np.testing.assert_array_equal(results_legacy["ensemble_models_used"], results_new["ensemble_models_used"])
    np.testing.assert_array_equal(results_new["ensemble_models_used"], results_new["ensemble_weights"] != 0)


def test_resolve_ensembler_drops_ensemble_size_for_unsupported_cls():
    """EnsembleSelectionConfigScorer always plumbs ensemble_size in; ensemblers that
    don't take one must not be forced to accept it.
    """
    from tabarena.simulation.ensemble import AbstractEnsembler
    from tabarena.simulation.ensemble_selection_config_scorer import EnsembleScorer

    class NoSizeEnsembler(AbstractEnsembler):
        def _fit(self, *, predictions, labels, time_limit=None):
            pass

        def predict_proba(self, predictions):
            return predictions[0]

    cls, kwargs = EnsembleScorer._resolve_ensembler(
        ensembler_cls=NoSizeEnsembler,
        ensembler_kwargs={"ensemble_size": 100},
        ensemble_method=None,
        ensemble_method_kwargs=None,
    )
    assert cls is NoSizeEnsembler
    assert "ensemble_size" not in kwargs

    cls, kwargs = EnsembleScorer._resolve_ensembler(
        ensembler_cls=GreedyEnsembler,
        ensembler_kwargs={"ensemble_size": 40},
        ensemble_method=None,
        ensemble_method_kwargs=None,
    )
    assert cls is GreedyEnsembler
    assert kwargs == {"ensemble_size": 40}


# -------------------------
# Reference ensemblers + guardrails (stage 3)
# -------------------------
def test_single_best_ensembler():
    from tabarena.simulation.ensemble import SingleBestEnsembler

    y, preds = _make_regression_task(n_models=5, seed=4)
    metric = get_metric(metric="rmse", problem_type="regression")
    per_model_errors = [metric.error(y, p) for p in preds]
    best = int(np.argmin(per_model_errors))

    ensembler = SingleBestEnsembler(problem_type="regression", metric=metric)
    ensembler.fit(predictions=preds, labels=y)

    expected = np.zeros(5)
    expected[best] = 1.0
    np.testing.assert_array_equal(ensembler.model_weights(), expected)
    np.testing.assert_array_equal(ensembler.predict_proba(preds), preds[best])
    assert ensembler.info() == {"best_index": best}
    assert ensembler.models_used().sum() == 1


def test_top_k_average_ensembler():
    from tabarena.simulation.ensemble import TopKAverageEnsembler

    y, preds = _make_binary_task(n_models=6, seed=5)
    metric = get_metric(metric="roc_auc", problem_type="binary")
    errors = np.array([metric.error(y, p) for p in preds])
    top3 = set(np.argsort(errors, kind="stable")[:3])

    ensembler = TopKAverageEnsembler(problem_type="binary", metric=metric, k=3)
    ensembler.fit(predictions=preds, labels=y)

    weights = ensembler.model_weights()
    assert set(np.flatnonzero(weights)) == top3
    np.testing.assert_allclose(weights[weights != 0], 1 / 3)
    # k larger than the model count is capped
    ensembler_all = TopKAverageEnsembler(problem_type="binary", metric=metric, k=100)
    ensembler_all.fit(predictions=preds, labels=y)
    np.testing.assert_allclose(ensembler_all.model_weights(), 1 / 6)


def test_fixed_weights_ensembler():
    from tabarena.simulation.ensemble import FixedWeightsEnsembler

    y, preds = _make_binary_task(n_models=4, seed=6)
    metric = get_metric(metric="roc_auc", problem_type="binary")
    weights = [0.5, 0.5, 0.0, 0.0]

    ensembler = FixedWeightsEnsembler(problem_type="binary", metric=metric, weights=weights)
    ensembler.fit(predictions=preds, labels=y)
    # use the fitted (float64) weights so dtype promotion matches predict_proba exactly
    w = ensembler.model_weights()
    np.testing.assert_array_equal(ensembler.predict_proba(preds), w[0] * preds[0] + w[1] * preds[1])
    np.testing.assert_array_equal(ensembler.models_used(), [True, True, False, False])

    bad = FixedWeightsEnsembler(problem_type="binary", metric=metric, weights=[1.0])
    with pytest.raises(ValueError, match="1 weights for 4 models"):
        bad.fit(predictions=preds, labels=y)


def test_ensembler_swap_through_task_evaluator():
    """Swapping the ensembling method is a single constructor argument."""
    from tabarena.simulation.ensemble import SingleBestEnsembler, TopKAverageEnsembler

    problem_type, metric, y, preds = _task_for_metric("roc_auc")
    for ensembler_cls, ensembler_kwargs in [
        (SingleBestEnsembler, {}),
        (TopKAverageEnsembler, {"k": 2}),
    ]:
        results, ensemble = _run_task_evaluator(
            ensembler_cls,
            ensembler_kwargs,
            problem_type=problem_type,
            eval_metric=metric,
            fit_eval_metric=metric,
            y=y,
            preds=preds,
        )
        assert np.isfinite(results["metric_error"])
        assert len(results["ensemble_weights"]) == len(preds)
        assert results["ensemble_models_used"].dtype == bool


def test_nonlinear_ensembler_rejected_on_preprocessed_metric_space():
    """A non-linear ensembler must be rejected when the metric feeds a transformed
    (linear-only) prediction space, and accepted on untransformed spaces.
    """
    from tabarena.metrics._fast_log_loss import fast_log_loss
    from tabarena.simulation.ensemble import AbstractEnsembler
    from tabarena.simulation.ensemble_selection_config_scorer import TaskEvaluator

    class NonLinearEnsembler(AbstractEnsembler):
        linear = False

        def _fit(self, *, predictions, labels, time_limit=None):
            pass

        def predict_proba(self, predictions):
            return np.maximum.reduce(list(predictions))

    problem_type, _, y, preds = _task_for_metric("log_loss")
    evaluator = TaskEvaluator(
        ensembler_cls=NonLinearEnsembler,
        ensembler_kwargs={},
        eval_metric=fast_log_loss,
        fit_eval_metric=fast_log_loss,
        problem_type=problem_type,
    )
    with pytest.raises(ValueError, match="non-linear"):
        evaluator.init_ens()

    # Untransformed metric space: allowed
    rmse = get_metric(metric="rmse", problem_type="regression")
    y_r, preds_r = _make_regression_task()
    results, _ = _run_task_evaluator(
        NonLinearEnsembler,
        {},
        problem_type="regression",
        eval_metric=rmse,
        fit_eval_metric=rmse,
        y=y_r,
        preds=preds_r,
    )
    assert np.isfinite(results["metric_error"])
    assert "ensemble_weights" not in results  # not weight-based
    assert results["ensemble_models_used"].all()  # conservative default: all models used
