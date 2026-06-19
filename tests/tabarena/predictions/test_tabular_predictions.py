from __future__ import annotations

import copy
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tabarena.predictions import (
    TabularModelPredictions,
    TabularPredictionsInMemory,
    TabularPredictionsInMemoryOpt,
    TabularPredictionsMemmap,
)
from tabarena.utils.test_utils import generate_artificial_dict, generate_dummy

if TYPE_CHECKING:
    from tabarena.predictions.tabular_predictions import TabularPredictionsDict

num_models = 13
num_folds = 3
dataset_shapes = {
    "d1": ((20,), (50,)),
    "d2": ((10,), (5,)),
    "d3": ((4, 3), (8, 3)),
}
models = [f"{i}" for i in range(num_models)]
pred_dict = generate_artificial_dict(num_folds, models, dataset_shapes)


def _assert_equivalent_pred_dict(pred_dict_1: TabularPredictionsDict, pred_dict_2: TabularPredictionsDict):
    assert sorted(pred_dict_1.keys()) == sorted(pred_dict_2.keys())
    for dataset in pred_dict_1:
        assert sorted(pred_dict_1[dataset].keys()) == sorted(pred_dict_2[dataset].keys())
        for fold, fold_dict in pred_dict_1[dataset].items():
            assert sorted(pred_dict_1[dataset][fold].keys()) == sorted(pred_dict_2[dataset][fold].keys())
            for split, model_dict in fold_dict.items():
                for model, _model_value in model_dict.items():
                    assert np.allclose(
                        pred_dict_1[dataset][fold][split][model], pred_dict_2[dataset][fold][split][model]
                    )


def _assert_equivalent_preds(preds_1: TabularModelPredictions, preds_2: TabularModelPredictions):
    print(f"{preds_1.__class__.__name__} vs {preds_2.__class__.__name__}")
    cur_models = preds_1.models
    cur_models_reverse = copy.deepcopy(cur_models)
    cur_models_reverse.reverse()
    assert preds_1.datasets == preds_2.datasets
    assert sorted(cur_models) == sorted(preds_2.models)
    assert preds_1.folds == preds_2.folds
    for dataset in preds_1.datasets:
        for fold in preds_1.folds:
            test_1 = preds_1.predict_test(dataset=dataset, fold=fold, models=cur_models)
            test_2 = preds_2.predict_test(dataset=dataset, fold=fold, models=cur_models)
            assert np.allclose(test_1, test_2)
            val_1 = preds_1.predict_val(dataset=dataset, fold=fold, models=cur_models)
            val_2 = preds_2.predict_val(dataset=dataset, fold=fold, models=cur_models)
            assert np.allclose(val_1, val_2)
            test_1_reverse = preds_1.predict_test(dataset=dataset, fold=fold, models=cur_models_reverse)
            test_2_reverse = preds_2.predict_test(dataset=dataset, fold=fold, models=cur_models_reverse)
            assert np.allclose(test_1_reverse, test_2_reverse)
            assert not np.allclose(test_1, test_1_reverse)
            assert np.allclose(test_1, np.flip(test_1_reverse, axis=0))
            val_1_reverse = preds_1.predict_val(dataset=dataset, fold=fold, models=cur_models_reverse)
            val_2_reverse = preds_2.predict_val(dataset=dataset, fold=fold, models=cur_models_reverse)
            assert np.allclose(val_1_reverse, val_2_reverse)
            assert not np.allclose(val_1, val_1_reverse)
            assert np.allclose(val_1, np.flip(val_1_reverse, axis=0))
            for model in cur_models:
                test_1 = preds_1.predict_test(dataset=dataset, fold=fold, models=[model])
                test_2 = preds_2.predict_test(dataset=dataset, fold=fold, models=[model])
                assert np.allclose(test_1, test_2)
                val_1 = preds_1.predict_val(dataset=dataset, fold=fold, models=[model])
                val_2 = preds_2.predict_val(dataset=dataset, fold=fold, models=[model])
                assert np.allclose(val_1, val_2)
    _assert_equivalent_pred_dict(preds_1.to_dict(), preds_2.to_dict())


@pytest.mark.parametrize(
    "cls",
    [
        TabularPredictionsMemmap,
        TabularPredictionsInMemory,
        TabularPredictionsInMemoryOpt,
    ],
)
def test_predictions_shape(cls):
    with tempfile.TemporaryDirectory() as tmpdirname:
        preds: TabularModelPredictions = cls.from_dict(pred_dict, output_dir=tmpdirname)
        assert sorted(preds.datasets) == ["d1", "d2", "d3"]
        assert sorted(preds.models) == sorted(models)
        assert preds.folds == list(range(num_folds))

        single_models = models[0:1]
        # we test the shape and values obtained when querying predictions when querying 13 and 1 models
        for test_models in [models, single_models]:
            for dataset, (val_shape, test_shape) in dataset_shapes.items():
                val_pred_proba = preds.predict_val(dataset=dataset, fold=2, models=test_models)
                test_pred_proba = preds.predict_test(dataset=dataset, fold=2, models=test_models)
                assert val_pred_proba.shape == (len(test_models), *list(val_shape))
                assert test_pred_proba.shape == (len(test_models), *list(test_shape))
                for i, model in enumerate(test_models):
                    assert np.allclose(val_pred_proba[i], generate_dummy(val_shape, test_models)[model])
                    assert np.allclose(test_pred_proba[i], generate_dummy(test_shape, test_models)[model])


@pytest.mark.parametrize(
    "cls",
    [
        TabularPredictionsMemmap,
        TabularPredictionsInMemory,
        TabularPredictionsInMemoryOpt,
    ],
)
def test_restrict_datasets(cls):
    with tempfile.TemporaryDirectory() as tmpdirname:
        preds: TabularModelPredictions = cls.from_dict(pred_dict, output_dir=tmpdirname)
        preds.restrict_datasets(["d1", "d3"])
        assert sorted(preds.datasets) == ["d1", "d3"]

        preds.restrict_datasets([])
        assert sorted(preds.datasets) == []
        assert sorted(preds.models) == []
        assert sorted(preds.folds) == []


@pytest.mark.parametrize(
    "cls",
    [
        TabularPredictionsMemmap,
        TabularPredictionsInMemory,
        TabularPredictionsInMemoryOpt,
    ],
)
def test_restrict_models(cls):
    with tempfile.TemporaryDirectory() as tmpdirname:
        preds: TabularModelPredictions = cls.from_dict(pred_dict, output_dir=tmpdirname)
        preds.restrict_models(["2", "3", "6"])
        assert sorted(preds.models) == ["2", "3", "6"]

        preds.restrict_models([])
        assert sorted(preds.datasets) == []
        assert sorted(preds.models) == []
        assert sorted(preds.folds) == []


@pytest.mark.parametrize(
    "cls",
    [
        TabularPredictionsMemmap,
        TabularPredictionsInMemory,
        TabularPredictionsInMemoryOpt,
    ],
)
def test_restrict_folds(cls):
    with tempfile.TemporaryDirectory() as tmpdirname:
        preds: TabularModelPredictions = cls.from_dict(pred_dict, output_dir=tmpdirname)
        preds.restrict_folds([0, 2])
        assert sorted(preds.folds) == [0, 2]

        preds.restrict_folds([])
        assert sorted(preds.datasets) == []
        assert sorted(preds.models) == []
        assert sorted(preds.folds) == []


@pytest.mark.parametrize(
    "cls",
    [
        TabularPredictionsMemmap,
        TabularPredictionsInMemory,
        TabularPredictionsInMemoryOpt,
    ],
)
def test_to_dict(cls):
    # Checks that to_dict returns the same dictionary as the original input
    with tempfile.TemporaryDirectory() as tmpdirname:
        preds = cls.from_dict(pred_dict, output_dir=tmpdirname)
        _assert_equivalent_pred_dict(pred_dict_1=pred_dict, pred_dict_2=preds.to_dict())


@pytest.mark.parametrize(
    "cls",
    [
        TabularPredictionsMemmap,
        TabularPredictionsInMemory,
        TabularPredictionsInMemoryOpt,
    ],
)
def test_to_data_dir(cls):
    # Checks that to_dict returns the same dictionary as the original input
    with tempfile.TemporaryDirectory() as tmpdirname:
        preds: TabularModelPredictions = cls.from_dict(pred_dict, output_dir=tmpdirname)
        with tempfile.TemporaryDirectory() as new_data_dir:
            preds.to_data_dir(data_dir=new_data_dir)
            pred_dict2 = cls.from_data_dir(data_dir=new_data_dir).to_dict()
            assert sorted(pred_dict.keys()) == sorted(pred_dict2.keys())
            for dataset in pred_dict:
                assert sorted(pred_dict[dataset].keys()) == sorted(pred_dict2[dataset].keys())
                for fold, fold_dict in pred_dict[dataset].items():
                    assert sorted(pred_dict[dataset][fold].keys()) == sorted(pred_dict2[dataset][fold].keys())
                    for split, model_dict in fold_dict.items():
                        for model, _model_value in model_dict.items():
                            assert np.allclose(
                                pred_dict[dataset][fold][split][model], pred_dict2[dataset][fold][split][model]
                            )


def test_predictions_after_restrict():
    # Checks that all TabularPredictions objects behave identically after performing restriction operations
    preds_cls_list = [
        TabularPredictionsInMemory,
        TabularPredictionsInMemoryOpt,
        TabularPredictionsMemmap,
    ]

    with tempfile.TemporaryDirectory() as tmpdirname:
        preds_list: list[TabularModelPredictions] = [
            c.from_dict(pred_dict, output_dir=tmpdirname) for c in preds_cls_list
        ]

        cur_datasets = copy.deepcopy(preds_list[0].datasets)
        cur_models = copy.deepcopy(preds_list[0].models)
        for preds_2 in preds_list[1:]:
            _assert_equivalent_preds(preds_1=preds_list[0], preds_2=preds_2)

        for p in preds_list:
            p.restrict_datasets(datasets=[cur_datasets[2], cur_datasets[1]])
        for preds_2 in preds_list[1:]:
            _assert_equivalent_preds(preds_1=preds_list[0], preds_2=preds_2)

        for p in preds_list:
            p.restrict_folds(folds=[2, 1])
        for preds_2 in preds_list[1:]:
            _assert_equivalent_preds(preds_1=preds_list[0], preds_2=preds_2)

        for p in preds_list:
            p.restrict_models(models=cur_models[2:7])
        for preds_2 in preds_list[1:]:
            _assert_equivalent_preds(preds_1=preds_list[0], preds_2=preds_2)

        for p in preds_list:
            p.restrict_models(models=[cur_models[2], cur_models[6], cur_models[4]])
        for preds_2 in preds_list[1:]:
            _assert_equivalent_preds(preds_1=preds_list[0], preds_2=preds_2)

        for p in preds_list:
            p.restrict_folds(folds=[2])
        for preds_2 in preds_list[1:]:
            _assert_equivalent_preds(preds_1=preds_list[0], preds_2=preds_2)

        for p in preds_list:
            p.restrict_datasets(datasets=[])
        for preds_2 in preds_list[1:]:
            _assert_equivalent_preds(preds_1=preds_list[0], preds_2=preds_2)

        for p in preds_list:
            assert p.datasets == []
            assert p.folds == []
            assert p.models == []
