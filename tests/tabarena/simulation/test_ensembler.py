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


def _run_no_preprocessing_task_evaluator(
    ensembler_cls, ensembler_kwargs, *, problem_type, eval_metric, fit_eval_metric, y, preds
):
    from tabarena.simulation.ensemble_selection_config_scorer import NoPreprocessingTaskEvaluator

    evaluator = NoPreprocessingTaskEvaluator(
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


@pytest.mark.parametrize("metric_name", ["roc_auc", "log_loss", "rmse"])
def test_no_preprocessing_task_evaluator_matches_task_evaluator(metric_name):
    """NoPreprocessingTaskEvaluator with a non EnsembleSelection method must match TaskEvaluator across all three metric regimes (incl. the metric-preprocessed log_loss space and the needs_pred rmse path)."""
    from tabarena.simulation.ensemble import TopKAverageEnsembler

    problem_type, metric, y, preds = _task_for_metric(metric_name)

    ensembler_cls = TopKAverageEnsembler
    ensembler_kwargs = {}

    results, _ = _run_task_evaluator(
        ensembler_cls,
        ensembler_kwargs,
        problem_type=problem_type,
        eval_metric=metric,
        fit_eval_metric=metric,
        y=y,
        preds=preds,
    )
    results_no_preprocessing, _ = _run_no_preprocessing_task_evaluator(
        ensembler_cls,
        ensembler_kwargs,
        problem_type=problem_type,
        eval_metric=metric,
        fit_eval_metric=metric,
        y=y,
        preds=preds,
    )

    if metric_name == "log_loss":
        assert pytest.approx(results["metric_error"], 1e-6) == results_no_preprocessing["metric_error"]
        assert pytest.approx(results["metric_error_val"], 1e-6) == results_no_preprocessing["metric_error_val"]
    else:
        assert results["metric_error"] == results_no_preprocessing["metric_error"]
        assert results["metric_error_val"] == results_no_preprocessing["metric_error_val"]
    np.testing.assert_array_equal(results["ensemble_weights"], results_no_preprocessing["ensemble_weights"])
    np.testing.assert_array_equal(results["ensemble_models_used"], results_no_preprocessing["ensemble_models_used"])
    np.testing.assert_array_equal(
        results_no_preprocessing["ensemble_models_used"], results_no_preprocessing["ensemble_weights"] != 0
    )


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


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_median_ensembler_basic(problem_type):
    from tabarena.simulation.ensemble import MedianEnsembler

    if problem_type == "binary":
        y, preds = _make_binary_task(seed=7)
        metric = get_metric(metric="roc_auc", problem_type=problem_type)
    elif problem_type == "multiclass":
        y, preds = _make_multiclass_task(seed=7)
        metric = get_metric(metric="log_loss", problem_type=problem_type)
    else:
        y, preds = _make_regression_task(seed=7)
        metric = get_metric(metric="rmse", problem_type=problem_type)

    ensembler = MedianEnsembler(problem_type=problem_type, metric=metric)
    ensembler.fit(predictions=preds, labels=y)

    # new data (a copy) uses the full refit model
    out = ensembler.predict_proba(preds.copy())
    if problem_type == "multiclass":
        assert out.shape == (len(y), preds.shape[2])
        np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-6)
    else:
        assert out.shape == (len(y),)
    if problem_type in ["binary", "multiclass"]:
        assert np.all(out <= 1.0)
        assert np.all(out >= 0.0)
    assert ensembler.model_weights() is None


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_median_predictions(problem_type):
    from tabarena.simulation.ensemble import MedianEnsembler

    if problem_type == "binary":
        preds = np.array(
            [
                [0.2, 0.7, 0.8, 0.7],
                [0.4, 0.6, 0.5, 0.4],
                [0.3, 0.6, 0.6, 0.4],
            ],
            dtype=np.float32,
        )
        expected_proba = np.array(
            [0.3, 0.6, 0.6, 0.4],
            dtype=np.float32,
        )
        expected_classes = np.array([0, 1, 1, 0])
        metric = get_metric(metric="roc_auc", problem_type=problem_type)
    elif problem_type == "multiclass":
        preds = np.array(
            [
                [
                    [0.0, 0.1, 0.9],
                    [0.2, 0.5, 0.3],
                    [0.6, 0.1, 0.3],
                ],
                [
                    [0.2, 0.4, 0.4],
                    [0.2, 0.5, 0.3],
                    [0.2, 0.7, 0.1],
                ],
                [
                    [0.1, 0.1, 0.8],
                    [0.2, 0.6, 0.2],
                    [0.5, 0.4, 0.1],
                ],
            ],
            dtype=np.float32,
        )
        expected_proba = np.array(
            [
                [0.1, 0.1, 0.8],
                [0.2, 0.5, 0.3],
                [0.5, 0.4, 0.1],
            ],
            dtype=np.float32,
        )
        expected_classes = np.array([2, 1, 0])
        metric = get_metric(metric="log_loss", problem_type=problem_type)
    elif problem_type == "regression":
        preds = np.array(
            [
                [10, 30, 10],
                [20, 20, 30],
                [30, 20, 30],
            ],
            dtype=np.float32,
        )
        expected_proba = np.array([20, 20, 30], dtype=np.float32)
        metric = get_metric("rmse", problem_type=problem_type)

    ensembler = MedianEnsembler(problem_type=problem_type, metric=metric)

    out_proba = ensembler.predict_proba(preds.copy())
    out_classes = ensembler.predict(preds.copy())
    np.testing.assert_allclose(out_proba, expected_proba, rtol=1e-6)
    if problem_type in ["binary", "multiclass"]:
        assert np.all(out_classes == expected_classes)


@pytest.mark.parametrize("problem_type", ["binary", "multiclass"])
def test_hard_voting_ensembler_basic(problem_type):
    from tabarena.simulation.ensemble import HardVotingEnsembler

    if problem_type == "binary":
        y, preds = _make_binary_task(seed=7)
        metric = get_metric(metric="roc_auc", problem_type=problem_type)
    else:
        y, preds = _make_multiclass_task(seed=7)
        metric = get_metric(metric="log_loss", problem_type=problem_type)

    ensembler = HardVotingEnsembler(problem_type=problem_type, metric=metric)
    ensembler.fit(predictions=preds, labels=y)

    out = ensembler.predict_proba(preds.copy())
    if problem_type == "multiclass":
        assert out.shape == (len(y), preds.shape[2])
        np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-6)
    else:
        assert out.shape == (len(y),)
    assert np.all(out <= 1.0)
    assert np.all(out >= 0.0)
    assert ensembler.model_weights() is None


def test_hard_voting_fails_regression():
    from tabarena.simulation.ensemble import HardVotingEnsembler

    metric = get_metric(metric="rmse", problem_type="regression")
    with pytest.raises(ValueError):
        ensembler = HardVotingEnsembler(problem_type="regression", metric=metric)


def test_hard_voting_fails_multiclass_with_wrong_dimension():
    from tabarena.simulation.ensemble import HardVotingEnsembler

    y, preds = _make_binary_task(seed=7)
    metric = get_metric(metric="log_loss", problem_type="multiclass")
    ensembler = HardVotingEnsembler(problem_type="multiclass", metric=metric)
    with pytest.raises(ValueError):
        ensembler.fit(predictions=preds, labels=y)
    with pytest.raises(ValueError):
        ensembler.predict_proba(predictions=preds)
    with pytest.raises(ValueError):
        ensembler.predict(predictions=preds)


@pytest.mark.parametrize("problem_type", ["binary", "multiclass"])
def test_hard_voting_predictions(problem_type):
    from tabarena.simulation.ensemble import HardVotingEnsembler

    if problem_type == "binary":
        preds = np.array(
            [
                [0.2, 0.7, 0.8, 0.7],
                [0.4, 0.6, 0.5, 0.4],
                [0.3, 0.6, 0.6, 0.4],
            ],
            dtype=np.float32,
        )
        expected_proba = np.array(
            [0, 1, 5 / 6, 1 / 3],
            dtype=np.float32,
        )
        expected_classes = np.array([0, 1, 1, 0])
        metric = get_metric(metric="roc_auc", problem_type=problem_type)
    else:
        preds = np.array(
            [
                [
                    [0.0, 0.1, 0.9],
                    [0.2, 0.5, 0.3],
                    [0.6, 0.1, 0.3],
                ],
                [
                    [0.2, 0.4, 0.4],
                    [0.2, 0.5, 0.3],
                    [0.2, 0.7, 0.1],
                ],
                [
                    [0.1, 0.1, 0.8],
                    [0.2, 0.6, 0.2],
                    [0.5, 0.4, 0.1],
                ],
            ],
            dtype=np.float32,
        )
        expected_proba = np.array(
            [
                [0, 1 / 6, 5 / 6],
                [0, 1, 0],
                [2 / 3, 1 / 3, 0],
            ],
            dtype=np.float32,
        )
        expected_classes = np.array([2, 1, 0])
        metric = get_metric(metric="log_loss", problem_type=problem_type)

    ensembler = HardVotingEnsembler(problem_type=problem_type, metric=metric)

    # new data (a copy) uses the full refit model
    out_proba = ensembler.predict_proba(preds.copy())
    out_classes = ensembler.predict(preds.copy())
    np.testing.assert_allclose(out_proba, expected_proba, rtol=1e-6)
    assert np.all(out_classes == expected_classes)


# -------------------------
# StackingEnsembler
# -------------------------
def _make_multiclass_task(n_models=6, n_samples=400, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n_samples)
    preds = rng.random((n_models, n_samples, n_classes))
    # make predictions informative
    for m in range(n_models):
        preds[m, np.arange(n_samples), y] += rng.uniform(0.5, 2.0)
    preds /= preds.sum(axis=2, keepdims=True)
    return y, preds.astype(np.float32)


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_stacking_ensembler_basic(problem_type):
    from tabarena.simulation.ensemble import StackingEnsembler

    if problem_type == "binary":
        y, preds = _make_binary_task(seed=7)
        metric = get_metric(metric="roc_auc", problem_type=problem_type)
    elif problem_type == "multiclass":
        y, preds = _make_multiclass_task(seed=7)
        metric = get_metric(metric="log_loss", problem_type=problem_type)
    else:
        y, preds = _make_regression_task(seed=7)
        metric = get_metric(metric="rmse", problem_type=problem_type)

    ensembler = StackingEnsembler(problem_type=problem_type, metric=metric)
    ensembler.fit(predictions=preds, labels=y)
    assert ensembler.info() == {"n_splits_used": 5}

    # new data (a copy) uses the full refit model
    out = ensembler.predict_proba(preds.copy())
    assert out.shape[0] == len(y)
    if problem_type == "multiclass":
        assert out.shape == (len(y), preds.shape[2])
        np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-6)
    else:
        assert out.ndim == 1
    assert ensembler.model_weights() is None
    assert ensembler.models_used().all()


def test_stacking_ensembler_oof_val_predictions():
    """Predicting on the fit input returns out-of-fold predictions (honest val error);
    predicting on any other input uses the full refit model.
    """
    from tabarena.simulation.ensemble import StackingEnsembler

    y, preds = _make_binary_task(seed=8)
    metric = get_metric(metric="roc_auc", problem_type="binary")
    ensembler = StackingEnsembler(problem_type="binary", metric=metric)
    ensembler.fit(predictions=preds, labels=y)

    oof = ensembler.predict_proba(preds)  # same object as fit input -> OOF
    in_sample = ensembler.predict_proba(preds.copy())  # different object -> refit model
    assert not np.array_equal(oof, in_sample)
    # in-sample predictions of the refit model must score better than (or equal to) OOF
    assert metric.error(y, in_sample) <= metric.error(y, oof)


def test_stacking_ensembler_small_data_falls_back():
    """With classes rarer than 2 per fold, CV is skipped and the fit input predicts
    in-sample instead of failing.
    """
    from tabarena.simulation.ensemble import StackingEnsembler

    metric = get_metric(metric="roc_auc", problem_type="binary")
    y = np.array([True] + [False] * 9)  # minority class count 1 -> no stratified CV possible
    rng = np.random.default_rng(9)
    preds = rng.random((3, 10)).astype(np.float32)

    ensembler = StackingEnsembler(problem_type="binary", metric=metric)
    ensembler.fit(predictions=preds, labels=y)
    assert ensembler.info() == {"n_splits_used": 1}
    np.testing.assert_array_equal(ensembler.predict_proba(preds), ensembler.predict_proba(preds.copy()))


def test_stacking_ensembler_custom_meta_model():
    """The meta-model is pluggable per problem kind (e.g. a foundation model)."""
    from sklearn.tree import DecisionTreeRegressor

    from tabarena.simulation.ensemble import StackingEnsembler

    y, preds = _make_regression_task(seed=10)
    metric = get_metric(metric="rmse", problem_type="regression")
    ensembler = StackingEnsembler(
        problem_type="regression",
        metric=metric,
        regressor_cls=DecisionTreeRegressor,
        regressor_kwargs={"max_depth": 3, "random_state": 0},
    )
    ensembler.fit(predictions=preds, labels=y)
    assert np.isfinite(metric.error(y, ensembler.predict_proba(preds.copy())))


def test_stacking_ensembler_rejected_on_fast_log_loss():
    """linear=False: the guardrail rejects stacking on metric-preprocessed spaces."""
    from tabarena.metrics._fast_log_loss import fast_log_loss
    from tabarena.simulation.ensemble import StackingEnsembler
    from tabarena.simulation.ensemble_selection_config_scorer import TaskEvaluator

    evaluator = TaskEvaluator(
        ensembler_cls=StackingEnsembler,
        ensembler_kwargs={},
        eval_metric=fast_log_loss,
        fit_eval_metric=fast_log_loss,
        problem_type="multiclass",
    )
    with pytest.raises(ValueError, match="non-linear"):
        evaluator.init_ens()


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_stacking_with_autogluon_abstract_model(problem_type):
    """StackingEnsembler stacks with any AutoGluon AbstractModel via the adapters.

    Uses ``RFModel`` because it ships with the base ``autogluon.tabular`` install, so this
    runs without the optional model libraries (LightGBM, CatBoost, ...).
    """
    from autogluon.tabular.models import RFModel

    from tabarena.simulation.ensemble import (
        AutoGluonStackerClassifier,
        AutoGluonStackerRegressor,
        StackingEnsembler,
    )

    hyperparameters = {"n_estimators": 10}
    if problem_type == "binary":
        y, preds = _make_binary_task(seed=11)
        metric = get_metric(metric="roc_auc", problem_type=problem_type)
    elif problem_type == "multiclass":
        rng = np.random.default_rng(11)
        n_samples, n_classes, n_models = 300, 3, 5
        # non-contiguous labels exercise the classes_ mapping
        y = rng.choice([0, 2, 5], size=n_samples)
        preds = rng.random((n_models, n_samples, n_classes)).astype(np.float32)
        preds /= preds.sum(axis=2, keepdims=True)
        metric = get_metric(metric="log_loss", problem_type=problem_type)
    else:
        y, preds = _make_regression_task(seed=11)
        metric = get_metric(metric="rmse", problem_type=problem_type)

    ensembler = StackingEnsembler(
        problem_type=problem_type,
        metric=metric,
        classifier_cls=AutoGluonStackerClassifier,
        classifier_kwargs={"model_cls": RFModel, "hyperparameters": hyperparameters},
        regressor_cls=AutoGluonStackerRegressor,
        regressor_kwargs={"model_cls": RFModel, "hyperparameters": hyperparameters},
        n_splits=2,
    )
    if problem_type == "multiclass":
        # StackingEnsembler's class-count bookkeeping is positional (0..k-1); remap
        y = np.searchsorted(np.unique(y), y)
    ensembler.fit(predictions=preds, labels=y)
    out = ensembler.predict_proba(preds.copy())  # full refit model (non-fit input)
    assert np.isfinite(np.asarray(out, dtype=float)).all()
    if problem_type == "multiclass":
        assert out.shape == (len(y), 3)
        np.testing.assert_allclose(np.asarray(out).sum(axis=1), 1.0, rtol=1e-5)
    else:
        assert np.asarray(out).ndim == 1
    # label-space predictions work through the ensembler
    labels_pred = ensembler.predict(preds.copy(), problem_type=problem_type)
    assert len(labels_pred) == len(y)


def test_autogluon_stacker_leaves_no_cwd_artifacts(monkeypatch, tmp_path):
    """The adapter must not drop an ``AutogluonModels/`` dir into the cwd, which is what
    AbstractModel does when constructed without a path.
    """
    from autogluon.tabular.models import RFModel

    from tabarena.simulation.ensemble import AutoGluonStackerRegressor

    monkeypatch.chdir(tmp_path)
    y, preds = _make_regression_task(n_models=3, n_samples=50, seed=12)
    regressor = AutoGluonStackerRegressor(model_cls=RFModel, hyperparameters={"n_estimators": 5})
    regressor.fit(preds.T, y)

    assert np.isfinite(regressor.predict(preds.T)).all()
    assert list(tmp_path.iterdir()) == []


# -------------------------
# Hill climbing (autogluon/autogluon#4505)
# -------------------------
def test_hill_climbing_beats_or_matches_single_best_on_synthetic():
    """HC starts from the best single model and only accepts improving blends, so
    validation error must be <= single-best error on the fit split.
    """
    from tabarena.simulation.ensemble import HillClimbingEnsembler, SingleBestEnsembler

    y, preds = _make_binary_task(n_models=12, n_samples=800, seed=7)
    metric = get_metric(metric="roc_auc", problem_type="binary")

    single = SingleBestEnsembler(problem_type="binary", metric=metric)
    single.fit(predictions=preds, labels=y)
    single_err = metric.error(y, single.predict_proba(preds))

    hc = HillClimbingEnsembler(
        problem_type="binary",
        metric=metric,
        precision=0.05,
        max_rounds=20,
        random_state=0,
    )
    assert hc.include_caruana_step is True
    hc.fit(predictions=preds, labels=y)
    hc_err = metric.error(y, hc.predict_proba(preds))

    assert hc_err <= single_err + 1e-12
    assert hc.model_weights() is not None
    assert np.isclose(hc.model_weights().sum(), 1.0)
    assert hc.models_used().any()


def test_hill_climbing_regression_and_weights_sum():
    from tabarena.simulation.ensemble import HillClimbingEnsembler

    y, preds = _make_regression_task(n_models=10, n_samples=600, seed=3)
    metric = get_metric(metric="rmse", problem_type="regression")
    hc = HillClimbingEnsembler(
        problem_type="regression",
        metric=metric,
        precision=0.05,
        max_rounds=15,
        random_state=1,
    )
    hc.fit(predictions=preds, labels=y)
    w = hc.model_weights()
    assert w is not None
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-12).all()
    # Prediction is a weighted sum of base preds
    combined = hc.predict_proba(preds)
    expected = sum(p * wi for p, wi in zip(preds, w, strict=True) if wi != 0)
    np.testing.assert_allclose(combined, expected, rtol=1e-5, atol=1e-5)


def test_hill_climbing_task_evaluator_runs():
    """Smoke: HillClimbingEnsembler plugs into TaskEvaluator like other ensemblers."""
    from tabarena.simulation.ensemble import HillClimbingEnsembler

    y, preds = _make_binary_task(seed=11)
    metric = get_metric(metric="roc_auc", problem_type="binary")
    results, ensemble = _run_task_evaluator(
        HillClimbingEnsembler,
        {"precision": 0.05, "max_rounds": 10, "random_state": 0},
        problem_type="binary",
        eval_metric=metric,
        fit_eval_metric=metric,
        y=y,
        preds=preds,
    )
    assert "metric_error" in results
    assert "ensemble_weights" in results
    assert results["ensemble_weights"] is not None
    assert ensemble is not None


def test_hill_climbing_invariants_and_edge_cases():
    """Trajectory monotonicity, sparsity cap, single-model, determinism, hyperparam guards."""
    from tabarena.simulation.ensemble import HillClimbingEnsembler

    y, preds = _make_binary_task(n_models=12, n_samples=600, seed=13)
    metric = get_metric(metric="roc_auc", problem_type="binary")

    hc = HillClimbingEnsembler(problem_type="binary", metric=metric, precision=0.05, max_rounds=25, random_state=0)
    hc.fit(predictions=preds, labels=y)
    traj = np.asarray(hc.trajectory_)
    assert traj.ndim == 1 and len(traj) >= 1
    assert np.all(np.diff(traj) <= 1e-15)
    single_best_err = min(metric.error(y, p) for p in preds)
    assert np.isclose(traj[0], single_best_err, rtol=1e-10, atol=1e-12)
    assert traj[-1] <= single_best_err + 1e-12
    assert np.isclose(hc.model_weights().sum(), 1.0)
    assert (hc.model_weights() >= -1e-12).all()
    assert hc.info()["n_rounds"] >= 1

    # max_models sparsity
    hc_sparse = HillClimbingEnsembler(
        problem_type="binary",
        metric=metric,
        precision=0.05,
        max_rounds=25,
        max_models=3,
        random_state=0,
    )
    hc_sparse.fit(predictions=preds, labels=y)
    assert int(hc_sparse.models_used().sum()) <= 3
    assert metric.error(y, hc_sparse.predict_proba(preds)) <= single_best_err + 1e-12

    # single model pool
    hc_one = HillClimbingEnsembler(problem_type="binary", metric=metric, precision=0.1, max_rounds=5, random_state=0)
    hc_one.fit(predictions=preds[:1], labels=y)
    np.testing.assert_allclose(hc_one.model_weights(), [1.0])

    # determinism
    a = HillClimbingEnsembler(problem_type="binary", metric=metric, precision=0.05, max_rounds=15, random_state=42)
    b = HillClimbingEnsembler(problem_type="binary", metric=metric, precision=0.05, max_rounds=15, random_state=42)
    a.fit(predictions=preds, labels=y)
    b.fit(predictions=preds, labels=y)
    np.testing.assert_allclose(a.model_weights(), b.model_weights())

    # does not mutate caller arrays
    preds_copy = preds.copy()
    HillClimbingEnsembler(problem_type="binary", metric=metric, precision=0.1, max_rounds=5, random_state=0).fit(
        predictions=preds, labels=y
    )
    assert np.array_equal(preds, preds_copy)

    # hyperparam validation
    with pytest.raises(ValueError, match="precision"):
        HillClimbingEnsembler(problem_type="binary", metric=metric, precision=0)
    with pytest.raises(ValueError, match="max_rounds"):
        HillClimbingEnsembler(problem_type="binary", metric=metric, max_rounds=0)
    with pytest.raises(ValueError, match="max_models"):
        HillClimbingEnsembler(problem_type="binary", metric=metric, max_models=0)
    with pytest.raises(ValueError, match="at least one"):
        HillClimbingEnsembler(problem_type="binary", metric=metric).fit(predictions=np.zeros((0, 10)), labels=y[:10])


def test_hill_climbing_multiclass_simplex():
    """Multiclass blends stay on the probability simplex; fit error ≤ single-best."""
    from tabarena.simulation.ensemble import HillClimbingEnsembler

    y, preds = _make_multiclass_task(n_models=8, n_samples=500, n_classes=4, seed=14)
    metric = get_metric(metric="log_loss", problem_type="multiclass")
    hc = HillClimbingEnsembler(problem_type="multiclass", metric=metric, precision=0.05, max_rounds=20, random_state=0)
    hc.fit(predictions=preds, labels=y)
    out = hc.predict_proba(preds)
    assert out.shape == (len(y), preds.shape[2])
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)
    assert (out >= -1e-8).all()
    single_best_err = min(metric.error(y, p) for p in preds)
    assert metric.error(y, out) <= single_best_err + 1e-10


def test_hill_climbing_time_limit_returns_valid_weights():
    """time_limit may stop early but must still yield a valid weight vector."""
    from tabarena.simulation.ensemble import HillClimbingEnsembler

    y, preds = _make_binary_task(n_models=30, n_samples=2000, seed=15)
    metric = get_metric(metric="roc_auc", problem_type="binary")
    hc = HillClimbingEnsembler(problem_type="binary", metric=metric, precision=0.01, max_rounds=200, random_state=0)
    hc.fit(predictions=preds, labels=y, time_limit=0.02)
    w = hc.model_weights()
    assert w is not None
    assert np.isclose(w.sum(), 1.0)
    assert np.isfinite(w).all()


def test_hill_climbing_warm_start_from_greedy_weights():
    """Warm-start HC from Greedy weights: refined val error must be <= Greedy val error."""
    from tabarena.simulation.ensemble import GreedyEnsembler, HillClimbingEnsembler

    y, preds = _make_binary_task(n_models=12, n_samples=800, seed=21)
    metric = get_metric(metric="roc_auc", problem_type="binary")

    greedy = GreedyEnsembler(
        problem_type="binary", metric=metric, ensemble_size=30, random_state=np.random.RandomState(0)
    )
    greedy.fit(predictions=preds, labels=y)
    g_err = metric.error(y, greedy.predict_proba(preds))

    hc = HillClimbingEnsembler(
        problem_type="binary",
        metric=metric,
        precision=0.05,
        max_rounds=20,
        initial_weights=greedy.model_weights(),
        random_state=0,
    )
    hc.fit(predictions=preds, labels=y)
    hc_err = metric.error(y, hc.predict_proba(preds))
    assert hc_err <= g_err + 1e-12
    assert hc.info()["warm_started"] is True
    assert hc.info()["init_val_error"] is not None
