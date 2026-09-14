"""Tests for the untimed inference bracket: persist, ``prepare_for_inference`` dispatch and release.

A fake predictor/trainer stands in for AutoGluon (shared fakes in
``tests/tabarena/benchmark/exec_models/conftest.py``): these exercise the hook logic of
``persist_inference`` and the AutoGluon exec models only (persist-outcome recording, dispatch to
persisted objects including bagged children, the memory-guard, failure and disabled paths, release
in ``cleanup``). The persist mechanics themselves are AutoGluon's.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

from autogluon.core.metrics import get_metric
from autogluon.core.models import AbstractModel

from tabarena.benchmark.exec_models.autogluon import AGModelWrapper, AGSingleWrapper, AGWrapper
from tabarena.benchmark.exec_models.persist_inference import (
    InferencePersistence,
    persist_for_inference,
    release_after_inference,
)
from tests.tabarena.benchmark.exec_models.conftest import (
    _FakeBag,
    _FakeFittedModel,
    _FakePredictor,
    _PlainModel,
    _PreparableModel,
)


def _rmse():
    return get_metric("rmse", problem_type="regression")


def _make_wrapper(*, persist: bool = True) -> AGWrapper:
    return AGWrapper(persist=persist, problem_type="regression", eval_metric=_rmse())


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

    assert wrapper.predictor.calls == [("persist", "best", 0.4)]
    assert wrapper._persisted_models == ["bag"]
    assert wrapper._prepared_models == ["bag", "child"]
    assert prepared == ["bag", "child"]  # hook on the bag and its loaded child; others skipped


def test_pre_predict_records_memory_guard_skip():
    wrapper = _make_wrapper()
    wrapper.predictor = _FakePredictor(persist_returns=[], trainer_models={})
    wrapper.pre_predict()
    assert wrapper._persisted_models == []  # persist attempted but skipped by the memory guard
    assert wrapper._prepared_models == []


def test_pre_predict_forwards_persist_max_memory():
    wrapper = _make_wrapper()
    wrapper.persist_max_memory = None
    wrapper.predictor = _FakePredictor(persist_returns=["m"], trainer_models={})
    wrapper.pre_predict()
    assert wrapper.predictor.calls == [("persist", "best", None)]


def test_single_wrapper_skips_the_persist_memory_guard():
    assert AGWrapper.persist_max_memory == 0.4
    assert AGSingleWrapper.persist_max_memory is None


def test_pre_predict_disabled_does_nothing():
    wrapper = _make_wrapper(persist=False)
    wrapper.predictor = _FakePredictor(persist_returns=["m"], trainer_models={})
    wrapper.pre_predict()
    assert wrapper.predictor.calls == []
    assert wrapper._persisted_models is None
    assert wrapper._prepared_models is None


def test_pre_predict_loads_string_children_when_persist_short_circuits():
    prepared: list[str] = []
    bag = _FakeBag("bag", prepared, children=["c1"])
    wrapper = _make_wrapper()
    # persist returns [] because the bag is already resident; its child is still a path string.
    wrapper.predictor = _FakePredictor(persist_returns=[], trainer_models={"bag": bag})

    wrapper.pre_predict()

    assert wrapper._persisted_models == ["bag"]
    assert bag.persist_child_models_calls == 1
    assert wrapper._prepared_models == ["bag", "c1"]
    assert prepared == ["bag", "c1"]


def test_prepare_failure_is_isolated_and_recorded():
    prepared: list[str] = []

    class _Boom:
        name = "boom"

        def prepare_for_inference(self):
            raise RuntimeError("prep bug")

    bag = _FakeBag("bag", prepared, children=[_Boom(), _PreparableModel("c2", prepared)])
    wrapper = _make_wrapper()
    wrapper.predictor = _FakePredictor(persist_returns=["bag"], trainer_models={"bag": bag})

    wrapper.pre_predict()

    assert wrapper._persisted_models == ["bag"]  # the persist outcome stays recorded
    assert wrapper._prepared_models == ["bag", "c2"]  # only the failing object is missing


def test_persist_failure_records_none_and_unpersists():
    wrapper = _make_wrapper()
    wrapper.predictor = _FakePredictor(persist_returns=["m"], trainer_models={}, persist_raises=RuntimeError("oom"))

    wrapper.pre_predict()  # no crash

    assert wrapper._persisted_models is None
    assert wrapper._prepared_models is None
    # Best-effort unpersist so None never coexists with partially resident models.
    assert wrapper.predictor.calls == [("persist", "best", 0.4), ("unpersist",)]


def test_persist_for_inference_metadata_block():
    predictor = _FakePredictor(persist_returns=["m"], trainer_models={"m": _PlainModel()})
    outcome = persist_for_inference(predictor, max_memory=None)
    assert outcome == InferencePersistence(persisted_models=["m"], prepared_models=[])
    assert outcome.as_metadata(persist=True) == {
        "persist": True,
        "persisted_models": ["m"],
        "prepared_for_inference": [],
    }


def test_unpersist_happens_in_cleanup_not_post_predict(tmp_path):
    path = tmp_path / "ag"
    path.mkdir()
    wrapper = _make_wrapper()
    wrapper.predictor = _FakePredictor(persist_returns=[], trainer_models={}, path=str(path))
    wrapper.post_predict()
    assert wrapper.predictor.calls == []  # served models stay resident for metadata and bag artifacts

    wrapper.cleanup()
    assert wrapper.predictor.calls == [("unpersist",)]
    assert not path.exists()

    path.mkdir()
    wrapper = _make_wrapper(persist=False)
    wrapper.predictor = _FakePredictor(persist_returns=[], trainer_models={}, path=str(path))
    wrapper.post_predict()
    wrapper.cleanup()
    assert wrapper.predictor.calls == []
    assert not path.exists()


def test_cleanup_without_a_predictor_is_safe():
    wrapper = _make_wrapper()
    wrapper.cleanup()  # a failed fit never assigned ``predictor``; cleanup must not raise


def test_cleanup_tolerates_a_failing_empty_cache(tmp_path, monkeypatch):
    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            raise RuntimeError("sticky CUDA fault")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=_Cuda()))
    wrapper = _make_wrapper()
    wrapper.predictor = _FakePredictor(persist_returns=[], trainer_models={}, path=str(tmp_path / "ag"))
    wrapper.cleanup()  # logged, not raised
    assert wrapper.predictor.calls == [("unpersist",)]


def test_cleanup_releases_shared_weights_only_when_opted_in(monkeypatch):
    weights = importlib.import_module("tabarena.models._weights")
    calls: list[str] = []
    monkeypatch.setattr(weights, "release", lambda **kw: calls.append("release") or 0)

    wrapper = _make_wrapper()
    wrapper.predictor = _FakePredictor(persist_returns=[], trainer_models={})
    wrapper.cleanup()
    assert calls == []

    wrapper.release_shared_weights_on_cleanup = True
    wrapper.cleanup()
    assert calls == ["release"]


def test_release_after_inference_tolerates_none_predictor():
    release_after_inference(None)


# --- AGSingleWrapper: served objects are reused by the post-evaluate consumers -----------


def _make_single_wrapper() -> AGSingleWrapper:
    class _DummyModel(AbstractModel):
        ag_key = "DUMMYPERSIST"
        ag_name = "DummyPersist"

    return AGSingleWrapper(
        model_cls=_DummyModel,
        model_hyperparameters={},
        problem_type="regression",
        eval_metric=_rmse(),
    )


def test_load_model_returns_persisted_object_until_cleanup(tmp_path):
    prepared: list[str] = []
    bag = _FakeBag("bag", prepared, children=["c1"])
    wrapper = _make_single_wrapper()
    wrapper.predictor = _FakePredictor(
        persist_returns=["bag"], trainer_models={"bag": bag}, path=str(tmp_path / "ag"), model_best="bag"
    )

    wrapper.pre_predict()
    assert wrapper._load_model() is bag  # the served object, no reload from disk

    wrapper.cleanup()
    assert wrapper._load_model() is not bag  # released; a fresh load would come from disk


def test_metadata_fit_records_prepared_models():
    wrapper = _make_single_wrapper()
    wrapper.predictor = _FakePredictor(persist_returns=["bag"], trainer_models={}, model_best="bag")
    wrapper._prepared_models = ["bag", "c1"]

    metadata = wrapper.get_metadata_fit(model=_FakeFittedModel({}), info={})

    assert metadata["persist"] is True
    assert metadata["persisted_models"] is None  # pre_predict did not run in this test
    assert metadata["prepared_for_inference"] == ["bag", "c1"]
    assert metadata["per_child_test_source"] is None
    assert set(metadata["shared_weights"]) == {"registry", "children"}


# --- outer/direct fits (AGModelWrapper): model already in memory, same hook ---------------


def _make_outer_wrapper() -> AGModelWrapper:
    return AGModelWrapper(model_cls=AbstractModel, problem_type="regression", eval_metric=_rmse())


def test_outer_wrapper_dispatches_prepare_for_inference():
    prepared: list[str] = []
    wrapper = _make_outer_wrapper()
    wrapper.model = _PreparableModel("direct", prepared)
    wrapper.pre_predict()
    assert prepared == ["direct"]
    assert wrapper.get_metadata() == {"persist": False, "persisted_models": None, "prepared_for_inference": ["direct"]}


def test_outer_wrapper_without_hook_is_a_noop():
    wrapper = _make_outer_wrapper()
    wrapper.model = _PlainModel()
    wrapper.pre_predict()  # no hook declared -> nothing to do, no error
    assert wrapper.get_metadata()["prepared_for_inference"] == []


def test_outer_wrapper_prepare_failure_is_non_fatal():
    class _Boom:
        def prepare_for_inference(self):
            raise RuntimeError("prep bug")

    wrapper = _make_outer_wrapper()
    wrapper.model = _Boom()
    wrapper.pre_predict()  # logged, inference proceeds unprepared
    assert wrapper.get_metadata()["prepared_for_inference"] == []
