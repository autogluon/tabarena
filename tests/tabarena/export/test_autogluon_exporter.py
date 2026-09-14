"""``AutoGluonExporter``: hyperparameters from the experiments, bagging counts from their protocol."""

from __future__ import annotations

import pytest
from autogluon.tabular.models import LGBModel

from tabarena.benchmark.experiment import AGModelBagExperiment, ValidationProtocol
from tabarena.benchmark.validation_protocol import (
    BEYONDARENA_VALIDATION_PROTOCOL,
    TABARENA_V0PT1_VALIDATION_PROTOCOL,
)
from tabarena.export.autogluon import AutoGluonExporter


def _experiments(protocol: ValidationProtocol | None, n: int = 2) -> list[AGModelBagExperiment]:
    return [
        AGModelBagExperiment(
            name=f"lgb_{i}",
            model_cls=LGBModel,
            model_hyperparameters={"learning_rate": 0.1 * (i + 1)},
            validation_protocol=protocol,
        )
        for i in range(n)
    ]


def test_preset_takes_the_counts_from_the_shared_protocol():
    preset = AutoGluonExporter(_experiments(TABARENA_V0PT1_VALIDATION_PROTOCOL)).export_preset()
    assert (preset["num_bag_folds"], preset["num_bag_sets"]) == (8, 1)
    assert "adapt_num_bag_folds_to_n_classes" not in preset
    assert [hp["learning_rate"] for hp in preset["hyperparameters"]["GBM"]] == [0.1, 0.2]
    assert [hp["ag_args"]["priority"] for hp in preset["hyperparameters"]["GBM"]] == [-1, -2]


def test_beyondarena_protocol_exports_its_default_regime_and_class_adaptation():
    preset = AutoGluonExporter(_experiments(BEYONDARENA_VALIDATION_PROTOCOL)).export_preset()
    assert (preset["num_bag_folds"], preset["num_bag_sets"]) == (8, 1)
    assert preset["adapt_num_bag_folds_to_n_classes"] is True


def test_unstamped_experiments_need_an_explicit_protocol():
    exporter = AutoGluonExporter(_experiments(None))
    with pytest.raises(ValueError, match="validation_protocol="):
        exporter.export_preset()
    assert exporter.export_preset(validation_protocol=ValidationProtocol.custom(3, 2))["num_bag_folds"] == 3


def test_disagreeing_protocols_are_refused():
    experiments = _experiments(TABARENA_V0PT1_VALIDATION_PROTOCOL) + _experiments(ValidationProtocol.custom(3))
    with pytest.raises(ValueError, match="share one validation protocol"):
        AutoGluonExporter(experiments).export_fit_kwargs()


def test_no_experiments_export_no_fit_kwargs():
    assert AutoGluonExporter([]).export_fit_kwargs() == {}
