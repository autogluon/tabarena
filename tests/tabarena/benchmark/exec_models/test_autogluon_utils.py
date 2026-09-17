"""Tests for the configured-model-class and preset helpers of ``autogluon_utils`` (registry lookups only)."""

from __future__ import annotations

import pytest
from autogluon.core.models import AbstractModel
from autogluon.tabular.models import CatBoostModel, LGBModel, RealMLPModel, TabICLModel, XGBoostModel

from tabarena.benchmark.exec_models import autogluon_utils as agu


class _Custom(AbstractModel):
    ag_key = "_CUSTOM_UTILS"
    ag_name = "_CustomUtils"


def test_resolve_model_cls():
    assert agu.resolve_model_cls(LGBModel) is LGBModel
    assert agu.resolve_model_cls("GBM") is LGBModel
    with pytest.raises(Exception, match="NOT_A_REAL_AG_KEY"):
        agu.resolve_model_cls("NOT_A_REAL_AG_KEY")


def test_resolve_effective_fit_kwargs_applies_presets_without_mutating_input():
    fit_kwargs = {"presets": "extreme_quality", "num_cpus": 4}
    effective = agu.resolve_effective_fit_kwargs(fit_kwargs)
    assert fit_kwargs == {"presets": "extreme_quality", "num_cpus": 4}
    assert effective["num_bag_folds"] == 8 and effective["num_cpus"] == 4
    assert isinstance(effective["hyperparameters"], dict)
    assert {"GBM", "CAT", "REALMLP"} <= set(effective["hyperparameters"])


def test_resolve_effective_fit_kwargs_explicit_wins_and_none_resolves_to_default():
    effective = agu.resolve_effective_fit_kwargs({"presets": "eq", "num_bag_folds": 3})
    assert effective["num_bag_folds"] == 3
    effective = agu.resolve_effective_fit_kwargs({})
    assert {"GBM", "CAT", "XGB"} <= set(effective["hyperparameters"])
    effective = agu.resolve_effective_fit_kwargs({"hyperparameters": "light"})
    assert isinstance(effective["hyperparameters"], dict)
    assert agu.resolve_effective_fit_kwargs(None)["hyperparameters"]


def test_iter_configured_model_classes_dict_forms():
    hyperparameters = {
        "GBM": [{"a": 1}, {"b": 2}],
        "CAT": {},
        XGBoostModel: {"n": 3},
        _Custom: [],
        "NOT_A_REAL_AG_KEY": {},
        "XT": "not-a-dict",
    }
    out = list(agu.iter_configured_model_classes(hyperparameters))
    classes = [cls for cls, _ in out]
    assert classes[:4] == [LGBModel, CatBoostModel, XGBoostModel, _Custom]
    assert dict(out)[LGBModel] == {"a": 1}
    assert dict(out)[CatBoostModel] == {}
    assert dict(out)[_Custom] is None
    assert all(not isinstance(cls, str) for cls in classes)
    # The yielded config is a copy.
    dict(out)[XGBoostModel]["n"] = 99
    assert hyperparameters[XGBoostModel] == {"n": 3}


def test_iter_configured_model_classes_stack_levels_and_names():
    out = list(agu.iter_configured_model_classes({1: {"GBM": {}}, "default": {"CAT": {}, "GBM": {"x": 1}}}))
    assert [cls for cls, _ in out] == [LGBModel, CatBoostModel]  # deduplicated across levels
    assert list(agu.iter_configured_model_classes(None)) == []
    assert list(agu.iter_configured_model_classes("not_a_config_name")) == []
    classes = [cls for cls, _ in agu.iter_configured_model_classes("zeroshot")]
    assert LGBModel in classes and CatBoostModel in classes


def test_configured_model_classes_resolves_presets():
    classes = [cls for cls, _ in agu.configured_model_classes({"presets": "extreme_quality"})]
    assert {LGBModel, CatBoostModel, RealMLPModel, TabICLModel} <= set(classes)
    assert [cls for cls, _ in agu.configured_model_classes({"hyperparameters": {"GBM": {}}})] == [LGBModel]
    assert [cls for cls, _ in agu.configured_model_classes({"hyperparameters": {"NOT_A_REAL_AG_KEY": {}}})] == []


def test_preset_uses_bagging():
    assert agu.preset_uses_bagging({"presets": "extreme_quality"}) is True
    assert agu.preset_uses_bagging({"presets": "best_quality"}) is True  # auto_stack
    assert agu.preset_uses_bagging({"presets": "medium_quality"}) is False
    assert agu.preset_uses_bagging({"hyperparameters": {"GBM": {}}}) is False
    assert agu.preset_uses_bagging({"hyperparameters": {"GBM": {}}, "num_bag_folds": 8}) is True
    assert agu.preset_uses_bagging({"num_bag_folds": 1}) is False


# ---- validation_structure_from_metadata -------------------------------------------------------------------------


def _metadata(**fields):
    from tabarena.benchmark.task.metadata import ValidationMetadata

    return ValidationMetadata(target_name="y", **fields)


def test_validation_structure_from_metadata_group_per_group_sizes_on_groups():
    from tabarena.benchmark.task.metadata import GroupLabelTypes

    structure = agu.validation_structure_from_metadata(
        _metadata(group_on="patient", group_labels=GroupLabelTypes.PER_GROUP, group_time_on="visit")
    )
    assert structure is not None
    assert structure.group_on == "patient"
    assert structure.time_on is None
    assert structure.size_validation_on_groups is True
    assert structure.temporal_forward_only is False
    # the within-group time column orders rows for the feature generator; it is not a split directive
    assert not hasattr(structure, "group_time_on")


def test_validation_structure_from_metadata_group_per_sample_sizes_on_rows():
    from tabarena.benchmark.task.metadata import GroupLabelTypes

    structure = agu.validation_structure_from_metadata(
        _metadata(group_on=["site", "subject"], group_labels=GroupLabelTypes.PER_SAMPLE, stratify_on="y")
    )
    assert structure.group_on == ["site", "subject"]
    assert structure.stratify_on == "y"
    assert structure.size_validation_on_groups is False


def test_validation_structure_from_metadata_explicit_sizing_wins():
    from tabarena.benchmark.task.metadata import GroupLabelTypes

    structure = agu.validation_structure_from_metadata(
        _metadata(group_on="patient", group_labels=GroupLabelTypes.PER_GROUP), size_validation_on_groups=False
    )
    assert structure.size_validation_on_groups is False


def test_validation_structure_from_metadata_temporal_forwards_the_flag():
    plain = agu.validation_structure_from_metadata(_metadata(time_on="date"))
    forward = agu.validation_structure_from_metadata(_metadata(time_on="date"), temporal_forward_only=True)
    assert plain.time_on == "date" and plain.temporal_forward_only is False
    assert forward.temporal_forward_only is True
    assert plain.group_on is None and plain.size_validation_on_groups is False


def test_validation_structure_from_metadata_stratify_only_is_no_structure():
    assert agu.validation_structure_from_metadata(_metadata(stratify_on="y")) is None
    assert agu.validation_structure_from_metadata(_metadata()) is None


def test_split_random_state_is_data_foundry_seed():
    # data_foundry keeps the seed as a local constant inside its split builders (curation_recommendations.py);
    # this pins the value TabArena's resolved splits were built with.
    assert agu.SPLIT_RANDOM_STATE == 4267
