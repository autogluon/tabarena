"""The experiment side of the validation protocol: stamping, rejections, records, legacy YAML.

Covers ``Experiment.validation_protocol`` (constructor, setter, ``_locals`` sync, YAML dict), the
``VALIDATION_FLAVOUR`` of each experiment class, what the bagged and holdout experiments reject, the
eager checks in ``task_cache_scope``, the kwargs handed to the wrapper, the result record, the seed
blocks of the config generators, and the migration of experiments serialized before the protocol
existed. No models are fit.
"""

from __future__ import annotations

import pytest
from autogluon.tabular.models import LGBModel

from tabarena.benchmark.exec_models.external import ExternalSystemModel
from tabarena.benchmark.experiment import (
    AGExperiment,
    AGModelBagExperiment,
    AGModelExperiment,
    AGModelOuterExperiment,
    Experiment,
    ExternalSystemExperiment,
    ValidationProtocol,
    ValidationProtocolError,
)
from tabarena.benchmark.experiment.experiment_constructor import _migrate_legacy_validation_kwargs
from tabarena.benchmark.task.metadata import ValidationMetadata
from tabarena.benchmark.validation_protocol import (
    BEYONDARENA_VALIDATION_PROTOCOL,
    CACHE_AUX_NAME,
    TABARENA_V0PT1_VALIDATION_PROTOCOL,
)
from tabarena.utils.cache import CacheFunctionPickle
from tabarena.utils.config_utils import (
    _apply_seed_to_bag_configs,
    generate_bag_experiments,
    generate_holdout_experiments,
)


class _FakeTask:
    """Enough of a task for ``init_method_kwargs`` and ``task_cache_scope``: no text, plain IID metadata."""

    has_text = False

    def get_validation_metadata(self) -> ValidationMetadata:
        return ValidationMetadata(target_name="t")


def _bag(model_hyperparameters: dict | None = None, **kwargs) -> AGModelBagExperiment:
    return AGModelBagExperiment(
        name="lgb_BAG_L1", model_cls=LGBModel, model_hyperparameters=model_hyperparameters or {}, **kwargs
    )


def _holdout(model_hyperparameters: dict | None = None, **kwargs) -> AGModelExperiment:
    return AGModelExperiment(
        name="lgb_HOLDOUT", model_cls=LGBModel, model_hyperparameters=model_hyperparameters or {}, **kwargs
    )


def _outer(**kwargs) -> AGModelOuterExperiment:
    return AGModelOuterExperiment(name="lgb", model_cls=LGBModel, model_hyperparameters={}, **kwargs)


def _system(**kwargs) -> ExternalSystemExperiment:
    return ExternalSystemExperiment(name="sys", system_cls=ExternalSystemModel, system_hyperparameters={}, **kwargs)


class TestFlavours:
    @pytest.mark.parametrize(
        ("make", "flavour"),
        [
            (_bag, "bagged"),
            (_holdout, "holdout"),
            (_outer, "outer"),
            (lambda: AGExperiment(name="ag", fit_kwargs={"hyperparameters": {"GBM": {}}}), "predictor"),
            (_system, "system"),
        ],
        ids=["bagged", "holdout", "outer", "predictor", "system"],
    )
    def test_each_experiment_class_declares_its_flavour(self, make, flavour):
        assert flavour == make().VALIDATION_FLAVOUR

    def test_a_bare_experiment_has_no_flavour(self):
        assert Experiment.VALIDATION_FLAVOUR is None


class TestProtocolOnTheExperiment:
    def test_default_is_no_protocol_and_no_counts_in_fit_kwargs(self):
        exp = _bag()
        assert exp.validation_protocol is None
        assert "num_bag_folds" not in exp.method_kwargs["fit_kwargs"]
        assert "validation_protocol" not in exp.to_yaml_dict()

    def test_constructor_normalizes_a_dict_and_syncs_locals(self):
        exp = _bag(validation_protocol={"num_bag_folds": 3, "num_bag_sets": 2})
        assert exp.validation_protocol == ValidationProtocol.custom(num_bag_folds=3, num_bag_sets=2)
        assert exp._locals["validation_protocol"] is exp.validation_protocol
        assert exp.to_yaml_dict()["validation_protocol"] == exp.validation_protocol.to_dict()

    def test_setter_stamps_and_unstamps(self):
        exp = _bag()
        exp.set_validation_protocol(TABARENA_V0PT1_VALIDATION_PROTOCOL.with_origin(arena="TabArena", enforced=True))
        assert exp.validation_protocol == TABARENA_V0PT1_VALIDATION_PROTOCOL
        assert exp.validation_protocol.enforced is True
        assert exp.to_yaml_dict()["validation_protocol"]["arena"] == "TabArena"
        exp.set_validation_protocol(None)
        assert exp.validation_protocol is None
        assert "validation_protocol" not in exp.to_yaml_dict()

    def test_count_constructor_kwargs_name_the_protocol_replacement(self):
        with pytest.raises(
            TypeError, match=r"validation_protocol=ValidationProtocol\(num_bag_folds=2, num_bag_sets=1\)"
        ):
            _bag(num_bag_folds=2, num_bag_sets=1)

    @pytest.mark.parametrize("make", [_bag, _holdout], ids=["bagged", "holdout"])
    @pytest.mark.parametrize("key", ["num_bag_folds", "num_bag_sets", "adapt_num_bag_folds_to_n_classes"])
    def test_counts_in_fit_kwargs_are_rejected(self, make, key):
        with pytest.raises(AssertionError, match="validation protocol"):
            make(method_kwargs={"fit_kwargs": {key: 2}})

    @pytest.mark.parametrize("make", [_bag, _holdout], ids=["bagged", "holdout"])
    @pytest.mark.parametrize("key", ["num_folds", "max_sets", "custom_splits"])
    def test_bagging_structure_overrides_in_ag_args_ensemble_are_rejected(self, make, key):
        with pytest.raises(AssertionError, match="bagging structure"):
            make(model_hyperparameters={"ag_args_ensemble": {key: 2}})

    def test_use_child_oof_is_not_a_structure_override(self):
        """A single leave-one-out child (``use_child_oof``, KNN) is a declared outlier, not a rewrite of the bag."""
        exp = _bag(model_hyperparameters={"ag_args_ensemble": {"use_child_oof": True}})
        assert exp.method_kwargs["model_hyperparameters"]["ag_args_ensemble"]["use_child_oof"] is True


class TestDeclaredOutlier:
    def test_knn_declares_its_single_leave_one_out_child(self):
        from tabarena.models.knn.model import KNNNewModel

        assert KNNNewModel._default_ag_args_ensemble_extra == {"use_child_oof": True}


class TestTimeLimitPlacement:
    def test_bagged_places_the_limit_on_the_bag(self):
        hp = _bag(time_limit=60).method_kwargs["model_hyperparameters"]
        assert hp["ag_args_ensemble"]["ag.max_time_limit"] == 60
        assert "ag.max_time_limit" not in hp

    def test_holdout_places_the_limit_on_the_model(self):
        hp = _holdout(time_limit=60).method_kwargs["model_hyperparameters"]
        assert hp["ag.max_time_limit"] == 60
        assert "ag_args_ensemble" not in hp


class TestFitTimeChecks:
    @pytest.mark.parametrize("make", [_bag, _holdout], ids=["bagged", "holdout"])
    def test_bagged_and_holdout_need_a_protocol_before_any_fit(self, make):
        with pytest.raises(ValidationProtocolError, match="arena context"):
            make().task_cache_scope(task=_FakeTask(), cache_task_key="t")

    @pytest.mark.parametrize("make", [_outer, _system], ids=["outer", "system"])
    def test_flavours_without_inner_validation_need_none(self, make):
        make().task_cache_scope(task=_FakeTask(), cache_task_key="t")

    def test_a_non_task_specific_protocol_runs_on_any_task(self):
        _bag(validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL).task_cache_scope(
            task=_FakeTask(), cache_task_key="t"
        )

    def test_a_task_specific_protocol_needs_a_split_aware_task(self):
        with pytest.raises(ValueError, match="task_specific_validation"):
            _bag(validation_protocol=BEYONDARENA_VALIDATION_PROTOCOL).task_cache_scope(
                task=_FakeTask(), cache_task_key="t"
            )


class TestMethodKwargs:
    def test_the_protocol_is_handed_to_autogluon_wrappers(self):
        exp = _bag(validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL)
        method_kwargs = exp.init_method_kwargs(task=_FakeTask())
        assert method_kwargs["validation_protocol"] == TABARENA_V0PT1_VALIDATION_PROTOCOL
        assert method_kwargs["validation_metadata"].target_name == "t"
        assert "use_task_specific_validation" not in method_kwargs

    def test_nothing_is_handed_over_without_a_protocol(self):
        assert "validation_protocol" not in _bag().init_method_kwargs(task=_FakeTask())

    @pytest.mark.parametrize("make", [_outer, _system], ids=["outer", "system"])
    def test_outer_and_system_wrappers_get_no_protocol(self, make):
        method_kwargs = make(validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL).init_method_kwargs(
            task=_FakeTask()
        )
        assert "validation_protocol" not in method_kwargs


class TestRecord:
    def test_a_stamped_bagged_experiment_records_the_protocol_key(self):
        record = _bag(validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL).validation_record()
        assert record["flavour"] == "bagged"
        assert record["key"] == "8x1"
        assert record["protocol"]["num_bag_folds"] == 8

    def test_an_unstamped_bagged_experiment_records_custom(self):
        assert _bag().validation_record() == {"flavour": "bagged", "protocol": None, "key": "custom"}

    def test_a_holdout_key_never_reads_as_the_bagged_protocol(self):
        assert (
            _holdout(validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL).validation_record()["key"] == "holdout:8x1"
        )

    def test_outer_and_system_records(self):
        assert _outer().validation_record() == {"flavour": "outer", "protocol": None, "key": "outer"}
        assert _system().validation_record() == {"flavour": "system", "protocol": None, "key": "system"}

    def test_init_and_run_recorded_completes_the_fitted_part(self):
        class _Runner:
            @staticmethod
            def init_and_run(**_kwargs):
                return {"metric_error": 0.1, "validation_protocol": {"regime": "default", "num_bag_folds_fitted": 8}}

        exp = _bag(validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL)
        exp.experiment_cls = _Runner
        out = exp._init_and_run_recorded(anything=1)
        assert out["metric_error"] == 0.1
        assert out["validation_protocol"]["flavour"] == "bagged"
        assert out["validation_protocol"]["key"] == "8x1"
        assert out["validation_protocol"]["num_bag_folds_fitted"] == 8

    def test_init_and_run_recorded_without_a_fitted_part(self):
        class _Runner:
            @staticmethod
            def init_and_run(**_kwargs):
                return {"metric_error": 0.1}

        exp = _outer()
        exp.experiment_cls = _Runner
        assert exp._init_and_run_recorded()["validation_protocol"] == exp.validation_record()


class TestGenerators:
    def test_generate_bag_experiments_forwards_the_protocol(self):
        (exp,) = generate_bag_experiments(
            model_cls=LGBModel,
            configs=[{}],
            time_limit=60,
            validation_protocol=ValidationProtocol.custom(num_bag_folds=3),
        )
        assert exp.validation_protocol == ValidationProtocol.custom(num_bag_folds=3)

    def test_generate_bag_experiments_defaults_to_no_protocol(self):
        (exp,) = generate_bag_experiments(model_cls=LGBModel, configs=[{}], time_limit=60)
        assert exp.validation_protocol is None
        assert exp.method_kwargs["model_hyperparameters"]["ag_args_ensemble"]["ag.max_time_limit"] == 60

    def test_generate_holdout_experiments_forwards_the_protocol(self):
        (exp,) = generate_holdout_experiments(LGBModel, [{}], validation_protocol=BEYONDARENA_VALIDATION_PROTOCOL)
        assert exp.validation_protocol == BEYONDARENA_VALIDATION_PROTOCOL


class TestSeedBlocks:
    """``fold-config-wise`` seeds reserve one block per config: 8 without a protocol (the official runs)."""

    def test_block_is_eight_without_a_protocol(self):
        configs = _apply_seed_to_bag_configs([{}, {}], "fold-config-wise")
        assert configs[1]["ag_args_ensemble"]["model_random_seed"] == 8

    @pytest.mark.parametrize("protocol", [TABARENA_V0PT1_VALIDATION_PROTOCOL, BEYONDARENA_VALIDATION_PROTOCOL])
    def test_official_protocols_keep_the_block_of_eight(self, protocol):
        configs = _apply_seed_to_bag_configs([{}, {}], "fold-config-wise", validation_protocol=protocol)
        assert configs[1]["ag_args_ensemble"]["model_random_seed"] == 8

    def test_block_follows_a_custom_protocol(self):
        configs = _apply_seed_to_bag_configs(
            [{}, {}], "fold-config-wise", validation_protocol=ValidationProtocol.custom(num_bag_folds=3, num_bag_sets=2)
        )
        assert configs[1]["ag_args_ensemble"]["model_random_seed"] == 6

    def test_config_wise_offset_is_one_whatever_the_protocol(self):
        configs = _apply_seed_to_bag_configs(
            [{}, {}], "config-wise", validation_protocol=ValidationProtocol.custom(num_bag_folds=3, num_bag_sets=2)
        )
        assert configs[1]["ag_args_ensemble"]["model_random_seed"] == 1


class TestLegacyYaml:
    """Experiments serialized before the protocol object existed load with a derived ``"legacy"`` protocol."""

    def test_explicit_counts_become_the_protocol(self):
        kwargs = _migrate_legacy_validation_kwargs(
            AGModelBagExperiment,
            {"name": "x", "num_bag_folds": 2, "num_bag_sets": 3, "dynamic_tabarena_validation_protocol": False},
        )
        assert set(kwargs) == {"name", "validation_protocol"}
        assert kwargs["validation_protocol"] == ValidationProtocol.custom(num_bag_folds=2, num_bag_sets=3)
        assert kwargs["validation_protocol"].name == "legacy"

    def test_auto_counts_under_the_dynamic_flag_are_the_pre_protocol_policy(self):
        kwargs = _migrate_legacy_validation_kwargs(
            AGModelBagExperiment,
            {
                "num_bag_folds": "auto",
                "num_bag_sets": "auto",
                "dynamic_tabarena_validation_protocol": True,
                "method_kwargs": {"fit_kwargs": {"adapt_num_bag_folds_to_n_classes": True, "num_cpus": 1}},
            },
        )
        assert kwargs["validation_protocol"] == BEYONDARENA_VALIDATION_PROTOCOL
        assert kwargs["method_kwargs"]["fit_kwargs"] == {"num_cpus": 1}

    def test_auto_counts_without_the_dynamic_flag_are_eight_by_one(self):
        kwargs = _migrate_legacy_validation_kwargs(
            AGModelBagExperiment,
            {"num_bag_folds": "auto", "num_bag_sets": "auto", "dynamic_tabarena_validation_protocol": False},
        )
        assert kwargs["validation_protocol"] == TABARENA_V0PT1_VALIDATION_PROTOCOL

    def test_a_holdout_dynamic_flag_becomes_a_task_specific_protocol(self):
        kwargs = _migrate_legacy_validation_kwargs(AGModelExperiment, {"dynamic_tabarena_validation_protocol": True})
        assert kwargs["validation_protocol"].task_specific_validation is True

    @pytest.mark.parametrize("cls", [AGModelOuterExperiment, ExternalSystemExperiment])
    def test_flavours_without_inner_validation_only_drop_the_flag(self, cls):
        kwargs = {"name": "x", "dynamic_tabarena_validation_protocol": False}
        assert _migrate_legacy_validation_kwargs(cls, kwargs) == {"name": "x"}

    def test_a_yaml_that_already_names_the_protocol_wins(self):
        protocol = ValidationProtocol.custom(num_bag_folds=4)
        kwargs = _migrate_legacy_validation_kwargs(
            AGModelBagExperiment, {"num_bag_folds": 2, "validation_protocol": protocol}
        )
        assert kwargs["validation_protocol"] == protocol

    def test_a_yaml_without_the_knobs_is_untouched(self):
        kwargs = {"name": "x", "method_kwargs": {"fit_kwargs": {"num_cpus": 1}}}
        assert _migrate_legacy_validation_kwargs(AGModelBagExperiment, dict(kwargs)) == kwargs


class TestCachedResultGuard:
    """A cached result is reused only when it was fit under the protocol the experiment now runs."""

    @staticmethod
    def _cacher(tmp_path, aux: dict | None) -> CacheFunctionPickle:
        cacher = CacheFunctionPickle(cache_name="results", cache_path=str(tmp_path / "data" / "lgb" / "1" / "0_0"))
        cacher.save_cache({"metric_error": 0.1})
        if aux is not None:
            cacher.save_aux(CACHE_AUX_NAME, aux)
        return cacher

    @staticmethod
    def _load(exp, cacher):
        return exp.run(task=None, fold=0, task_name="ds", cache_task_key="ds", cacher=cacher)

    def test_a_matching_side_record_is_reused(self, tmp_path):
        exp = _bag(validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL)
        cacher = self._cacher(tmp_path, {"key": "8x1", "flavour": "bagged"})
        assert self._load(exp, cacher) == {"metric_error": 0.1}

    def test_a_legacy_result_without_side_record_is_trusted(self, tmp_path):
        exp = _bag(validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL)
        assert self._load(exp, self._cacher(tmp_path, None)) == {"metric_error": 0.1}

    def test_a_result_fit_under_another_protocol_is_refused(self, tmp_path):
        exp = _bag(validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL)
        cacher = self._cacher(tmp_path, {"key": "3x1", "flavour": "bagged"})
        with pytest.raises(ValidationProtocolError, match="'3x1'.*'8x1'.*ignore_cache=True"):
            self._load(exp, cacher)

    def test_a_holdout_result_never_passes_as_the_bagged_protocol(self, tmp_path):
        exp = _bag(validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL)
        cacher = self._cacher(tmp_path, {"key": "holdout:8x1", "flavour": "holdout"})
        with pytest.raises(ValidationProtocolError):
            self._load(exp, cacher)

    def test_aux_record_is_the_key_and_flavour(self):
        assert _bag(validation_protocol=ValidationProtocol.custom(num_bag_folds=3)).validation_aux_record() == {
            "key": "3x1",
            "flavour": "bagged",
        }
        assert _outer().validation_aux_record() == {"key": "outer", "flavour": "outer"}
