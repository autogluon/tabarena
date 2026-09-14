"""The arena context owns the official validation protocol: it stamps, refuses, and checks the tasks.

Covers the constructor (declared protocol, opt-out, refusal of a foreign protocol, the bare context),
``build_jobs`` / ``run_jobs`` stamping per experiment flavour, the notice, and the split-structure check
of the task collection. No models are fit.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest
from autogluon.tabular.models import LGBModel

from tabarena.benchmark.exec_models.external import ExternalSystemModel
from tabarena.benchmark.experiment import (
    AGExperiment,
    AGModelBagExperiment,
    AGModelExperiment,
    AGModelOuterExperiment,
    ExternalSystemExperiment,
    Job,
    ValidationProtocol,
    ValidationProtocolError,
)
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.benchmark.validation_protocol import (
    BEYONDARENA_VALIDATION_PROTOCOL,
    TABARENA_V0PT1_VALIDATION_PROTOCOL,
)
from tabarena.contexts import AbstractArenaContext, BeyondArenaContext, TabArenaContext

CUSTOM = ValidationProtocol.custom(3)


def _collection() -> TaskMetadataCollection:
    """Two datasets unknown to the official TabArena suite, two folds each."""
    return TaskMetadataCollection.from_legacy_df(
        pd.DataFrame(
            {
                "tid": [0, 1],
                "dataset": ["small_ds", "big_ds"],
                "n_folds": [2, 2],
                "n_repeats": [1, 1],
                "n_samples_train_per_fold": [100, 50_000],
                "n_samples_test_per_fold": [50, 25_000],
                "NumberOfInstances": [150, 75_000],
                "problem_type": ["binary", "regression"],
                "n_features": [10, 10],
                "n_classes": [2, 0],
            },
        )
    )


def _ctx(**kwargs) -> TabArenaContext:
    return TabArenaContext(methods=[], task_metadata=_collection(), **kwargs)


def _bag(name: str = "lgb_BAG_L1", **kwargs) -> AGModelBagExperiment:
    return AGModelBagExperiment(name=name, model_cls=LGBModel, model_hyperparameters={}, **kwargs)


def _holdout(**kwargs) -> AGModelExperiment:
    return AGModelExperiment(name="lgb_HOLDOUT", model_cls=LGBModel, model_hyperparameters={}, **kwargs)


def _outer() -> AGModelOuterExperiment:
    return AGModelOuterExperiment(name="lgb", model_cls=LGBModel, model_hyperparameters={})


def _predictor() -> AGExperiment:
    return AGExperiment(name="ag", fit_kwargs={"hyperparameters": {"GBM": {}}})


def _system() -> ExternalSystemExperiment:
    return ExternalSystemExperiment(name="sys", system_cls=ExternalSystemModel, system_hyperparameters={})


class TestDeclaredProtocols:
    def test_arenas_declare_their_official_protocol(self):
        assert TabArenaContext.OFFICIAL_VALIDATION_PROTOCOL == TABARENA_V0PT1_VALIDATION_PROTOCOL
        assert BeyondArenaContext.OFFICIAL_VALIDATION_PROTOCOL == BEYONDARENA_VALIDATION_PROTOCOL
        assert AbstractArenaContext.OFFICIAL_VALIDATION_PROTOCOL is None
        assert frozenset({"bagged", "system"}) == AbstractArenaContext.OFFICIAL_VALIDATION_FLAVOURS
        assert TabArenaContext.OFFICIAL_BUNDLE_HINT == "TabArenaV0pt1ExperimentBundle"
        assert BeyondArenaContext.OFFICIAL_BUNDLE_HINT == "BeyondArenaExperimentBundle"

    def test_default_context_enforces_its_protocol(self):
        ctx = _ctx()
        assert ctx.validation_protocol == TABARENA_V0PT1_VALIDATION_PROTOCOL
        assert ctx.validation_protocol.arena == "TabArena"
        assert ctx.validation_protocol.enforced is True
        assert ctx.enforces_validation_protocol is True

    def test_opt_out_keeps_the_official_protocol_but_stops_enforcing(self):
        ctx = _ctx(official_validation_protocol=False)
        assert ctx.validation_protocol == TABARENA_V0PT1_VALIDATION_PROTOCOL
        assert ctx.validation_protocol.enforced is False
        assert ctx.enforces_validation_protocol is False

    def test_a_foreign_protocol_is_refused_while_enforcing(self):
        with pytest.raises(ValueError, match="official_validation_protocol=False"):
            _ctx(validation_protocol=CUSTOM)

    def test_the_official_protocol_may_be_passed_explicitly(self):
        ctx = _ctx(validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL)
        assert ctx.enforces_validation_protocol is True

    def test_a_custom_protocol_runs_once_opted_out(self):
        ctx = _ctx(validation_protocol={"num_bag_folds": 3}, official_validation_protocol=False)
        assert ctx.validation_protocol == CUSTOM
        assert ctx.enforces_validation_protocol is False

    def test_a_bare_context_enforces_only_what_it_is_given(self):
        bare = AbstractArenaContext(methods=[], task_metadata=_collection())
        assert bare.validation_protocol is None
        assert bare.enforces_validation_protocol is False
        given = AbstractArenaContext(methods=[], task_metadata=_collection(), validation_protocol=CUSTOM)
        assert given.validation_protocol == CUSTOM
        assert given.validation_protocol.arena == "Arena"
        assert given.enforces_validation_protocol is True


class TestBuildJobsStamping:
    def test_unstamped_bagged_experiments_receive_the_enforced_protocol(self):
        exp = _bag()
        jobs = _ctx().build_jobs([exp], subset="lite")
        assert exp.validation_protocol == TABARENA_V0PT1_VALIDATION_PROTOCOL
        assert exp.validation_protocol.enforced is True
        assert exp.validation_protocol.arena == "TabArena"
        assert all(job.experiment is exp for job in jobs)
        assert exp.to_yaml_dict()["validation_protocol"]["enforced"] is True

    def test_holdout_and_predictor_receive_the_protocol_without_enforcement(self):
        holdout, predictor = _holdout(), _predictor()
        _ctx().build_jobs([holdout, predictor], subset="lite")
        for exp in (holdout, predictor):
            assert exp.validation_protocol == TABARENA_V0PT1_VALIDATION_PROTOCOL
            assert exp.validation_protocol.enforced is False

    def test_outer_and_system_experiments_are_untouched(self):
        outer, system = _outer(), _system()
        _ctx().build_jobs([outer, system], subset="lite")
        assert outer.validation_protocol is None
        assert system.validation_protocol is None

    def test_a_matching_explicit_protocol_is_relabelled_not_refused(self):
        exp = _bag(validation_protocol=ValidationProtocol(name="my-8x1"))
        _ctx().build_jobs([exp], subset="lite")
        assert exp.validation_protocol == TABARENA_V0PT1_VALIDATION_PROTOCOL
        assert exp.validation_protocol.name == "TabArena-v0.1"
        assert exp.validation_protocol.enforced is True

    def test_a_foreign_bagged_protocol_is_refused_with_the_pairing_hint(self):
        exp = _bag(validation_protocol=CUSTOM)
        with pytest.raises(ValidationProtocolError) as excinfo:
            _ctx().build_jobs([exp, _bag("other_BAG_L1")], subset="lite")
        message = str(excinfo.value)
        assert "'lgb_BAG_L1'" in message
        assert "[3x1]" in message and "[8x1]" in message
        assert "TabArenaV0pt1ExperimentBundle" in message
        assert "official_validation_protocol=False" in message
        assert "'other_BAG_L1'" not in message

    def test_a_split_structure_override_is_refused(self):
        exp = _bag(method_kwargs={"validation_metadata": {"group_on": "grp"}})
        with pytest.raises(ValidationProtocolError, match="validation_metadata"):
            _ctx().build_jobs([exp], subset="lite")

    def test_opted_out_context_keeps_custom_protocols_and_marks_them(self):
        custom, unstamped = _bag(validation_protocol=CUSTOM), _bag("other_BAG_L1")
        _ctx(official_validation_protocol=False).build_jobs([custom, unstamped], subset="lite")
        assert custom.validation_protocol == CUSTOM
        assert custom.validation_protocol.enforced is False
        assert unstamped.validation_protocol == TABARENA_V0PT1_VALIDATION_PROTOCOL
        assert unstamped.validation_protocol.enforced is False

    def test_context_level_custom_protocol_reaches_unstamped_experiments(self):
        exp = _bag()
        _ctx(validation_protocol=CUSTOM, official_validation_protocol=False).build_jobs([exp], subset="lite")
        assert exp.validation_protocol == CUSTOM

    def test_stubs_without_a_flavour_are_ignored(self):
        class _Stub:
            model_constraints = None
            name = "stub"

        jobs = _ctx().build_jobs([_Stub()], subset="lite")
        assert len(jobs) == 2

    def test_notice_summarizes_what_was_stamped(self, capsys):
        _ctx().build_jobs([_bag(), _holdout(), _outer(), _system()], subset="lite")
        out = capsys.readouterr().out
        assert "TabArena-v0.1 [8x1] enforced on 1 bagged experiment(s)" in out
        assert "1 system experiment(s) own their validation" in out
        assert "1 holdout experiment(s) run outside the official protocol" in out
        assert "1 outer experiment(s) run outside the official protocol" in out

    def test_a_bare_context_without_protocol_stamps_nothing(self, capsys):
        exp = _bag()
        AbstractArenaContext(methods=[], task_metadata=_collection()).build_jobs([exp], subset="lite")
        assert exp.validation_protocol is None
        assert "Validation protocol" not in capsys.readouterr().out


class TestRunJobsStamping:
    def test_hand_built_jobs_are_prepared_before_running(self, monkeypatch):
        seen: dict = {}

        class _FakeRunner:
            def __init__(self, **kwargs):
                seen["kwargs"] = kwargs

            def run_jobs(self, jobs):
                seen["jobs"] = jobs
                return []

        monkeypatch.setattr("tabarena.benchmark.experiment.ExperimentBatchRunner", _FakeRunner)
        exp = _bag()
        jobs = [Job.create(exp, "small_ds", fold=0), Job.create(exp, "small_ds", fold=1)]
        assert _ctx().run_jobs(jobs, expname=None, register=False) == []
        assert exp.validation_protocol == TABARENA_V0PT1_VALIDATION_PROTOCOL
        assert exp.validation_protocol.enforced is True
        assert seen["jobs"] == jobs

    def test_a_foreign_protocol_is_refused_before_anything_runs(self, monkeypatch):
        class _FakeRunner:
            def __init__(self, **kwargs):
                raise AssertionError("the runner must not be built")

        monkeypatch.setattr("tabarena.benchmark.experiment.ExperimentBatchRunner", _FakeRunner)
        jobs = [Job.create(_bag(validation_protocol=CUSTOM), "small_ds", fold=0)]
        with pytest.raises(ValidationProtocolError):
            _ctx().run_jobs(jobs, expname=None, register=False)


class TestTaskCollectionStructure:
    @staticmethod
    def _official_anneal() -> TaskMetadataCollection:
        return TaskMetadataCollection.from_preset("TabArena-v0.1").subset_tasks(
            dataset_names=["anneal"], split_indices="lite"
        )

    def test_the_official_collection_passes(self):
        ctx = TabArenaContext(methods=[], task_metadata=self._official_anneal())
        assert len(ctx.build_jobs([_bag()])) == 1

    def test_changed_split_structure_is_refused(self):
        task = replace(self._official_anneal().tasks[0], time_on="some_column")
        collection = TaskMetadataCollection([task])
        with pytest.raises(ValidationProtocolError, match="time_on='some_column'"):
            TabArenaContext(methods=[], task_metadata=collection).build_jobs([_bag()])
        # Opting out runs the collection as given.
        opted_out = TabArenaContext(methods=[], task_metadata=collection, official_validation_protocol=False)
        assert len(opted_out.build_jobs([_bag()])) == 1

    def test_a_split_outside_the_official_grid_is_refused(self):
        task = self._official_anneal().tasks[0]
        split = next(iter(task.splits_metadata.values()))
        moved = replace(task, splits_metadata={"r0f99": replace(split, fold=99)})
        with pytest.raises(ValidationProtocolError, match="fold=99"):
            TabArenaContext(methods=[], task_metadata=TaskMetadataCollection([moved])).build_jobs([_bag()])

    def test_custom_datasets_on_an_official_context_pass(self):
        # `small_ds` / `big_ds` are unknown to the TabArena suite: nothing to compare, their structure is
        # recorded per result instead.
        assert len(_ctx().build_jobs([_bag()], subset="lite")) == 2
