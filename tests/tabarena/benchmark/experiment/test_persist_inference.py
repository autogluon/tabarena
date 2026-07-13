"""Tests for the untimed inference-side persist + ``prepare_for_inference`` dispatch.

A fake predictor/trainer stands in for AutoGluon: these exercise ``AGWrapper``'s hook logic
only (persist-outcome recording, dispatch to persisted objects including bagged children,
the memory-guard and disabled paths) — the persist mechanics themselves are AutoGluon's.
"""

from __future__ import annotations

from types import SimpleNamespace

from autogluon.core.metrics import get_metric
from autogluon.core.models import AbstractModel

from tabarena.benchmark.exec_models.autogluon import AGModelWrapper, AGWrapper


class _PreparableModel:
    def __init__(self, name: str, log: list[str]):
        self.name = name
        self._log = log

    def prepare_for_inference(self) -> None:
        self._log.append(self.name)


class _PlainModel:
    """Persisted model without the optional hook — must be skipped without error."""


class _FakePredictor:
    def __init__(self, *, persist_returns: list[str], trainer_models: dict):
        self._persist_returns = persist_returns
        self._trainer = SimpleNamespace(models=trainer_models)
        self.calls: list[tuple] = []

    def persist(self, models):
        self.calls.append(("persist", models))
        return list(self._persist_returns)

    def unpersist(self):
        self.calls.append(("unpersist",))


def _make_wrapper(*, persist: bool = True) -> AGWrapper:
    return AGWrapper(
        persist=persist,
        problem_type="regression",
        eval_metric=get_metric("rmse", problem_type="regression"),
    )


def test_persist_defaults_to_true():
    assert _make_wrapper().persist is True


def test_pre_predict_persists_and_dispatches_prepare_for_inference():
    prepared: list[str] = []
    bag = _PreparableModel("bag", prepared)
    # A persisted bag holds loaded child objects; unloaded children stay strings.
    bag.models = [_PreparableModel("child", prepared), "unloaded_child", _PlainModel()]
    wrapper = _make_wrapper()
    wrapper.predictor = _FakePredictor(persist_returns=["bag"], trainer_models={"bag": bag})

    wrapper.pre_predict()

    assert wrapper.predictor.calls == [("persist", "best")]
    assert wrapper._persisted_models == ["bag"]
    assert prepared == ["bag", "child"]  # hook on the bag and its loaded child; others skipped


def test_pre_predict_records_memory_guard_skip():
    wrapper = _make_wrapper()
    wrapper.predictor = _FakePredictor(persist_returns=[], trainer_models={})
    wrapper.pre_predict()
    assert wrapper._persisted_models == []  # persist attempted but skipped by the memory guard


def test_pre_predict_disabled_does_nothing():
    wrapper = _make_wrapper(persist=False)
    wrapper.predictor = _FakePredictor(persist_returns=["m"], trainer_models={})
    wrapper.pre_predict()
    assert wrapper.predictor.calls == []
    assert wrapper._persisted_models is None


def test_post_predict_unpersists_only_when_enabled():
    wrapper = _make_wrapper()
    wrapper.predictor = _FakePredictor(persist_returns=[], trainer_models={})
    wrapper.post_predict()
    assert wrapper.predictor.calls == [("unpersist",)]

    wrapper = _make_wrapper(persist=False)
    wrapper.predictor = _FakePredictor(persist_returns=[], trainer_models={})
    wrapper.post_predict()
    assert wrapper.predictor.calls == []


# --- outer/direct fits (AGModelWrapper) — model already in memory, same hook ---------------


def _make_outer_wrapper() -> AGModelWrapper:
    return AGModelWrapper(
        model_cls=AbstractModel,
        problem_type="regression",
        eval_metric=get_metric("rmse", problem_type="regression"),
    )


def test_outer_wrapper_dispatches_prepare_for_inference():
    prepared: list[str] = []
    wrapper = _make_outer_wrapper()
    wrapper.model = _PreparableModel("direct", prepared)
    wrapper.pre_predict()
    assert prepared == ["direct"]


def test_outer_wrapper_without_hook_is_a_noop():
    wrapper = _make_outer_wrapper()
    wrapper.model = _PlainModel()
    wrapper.pre_predict()  # no hook declared -> nothing to do, no error


def test_outer_wrapper_prepare_failure_is_non_fatal():
    class _Boom:
        def prepare_for_inference(self):
            raise RuntimeError("prep bug")

    wrapper = _make_outer_wrapper()
    wrapper.model = _Boom()
    wrapper.pre_predict()  # logged, inference proceeds unprepared
