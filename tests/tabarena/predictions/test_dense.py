from __future__ import annotations

import copy
import tempfile

import pytest

from tabarena.predictions import TabularModelPredictions, TabularPredictionsInMemory, TabularPredictionsMemmap
from tabarena.simulation.dense_utils import (
    force_to_dense,
    get_folds_dense,
    get_models_dense,
    is_dense_folds,
    is_dense_models,
    is_empty,
    list_folds_available,
    list_models_available,
    print_summary,
)
from tabarena.utils.test_utils import generate_artificial_dict

num_models = 13
num_folds = 3
dataset_shapes = {
    "d1": ((20,), (50,)),
    "d2": ((10,), (5,)),
    "d3": ((4, 3), (8, 3)),
}
models = [f"{i}" for i in range(num_models)]
pred_dict = generate_artificial_dict(num_folds, models, dataset_shapes)


def _make_empty_and_assert(pred_proba: TabularModelPredictions):
    pred_proba_copy = copy.deepcopy(pred_proba)
    pred_proba_copy.restrict_folds([])
    assert pred_proba_copy.datasets == []
    assert pred_proba_copy.folds == []
    assert pred_proba_copy.models == []

    pred_proba_copy = copy.deepcopy(pred_proba)
    pred_proba_copy.restrict_datasets([])
    assert pred_proba_copy.datasets == []
    assert pred_proba_copy.folds == []
    assert pred_proba_copy.models == []

    pred_proba_copy = copy.deepcopy(pred_proba)
    pred_proba_copy.restrict_models([])
    assert pred_proba_copy.datasets == []
    assert pred_proba_copy.folds == []
    assert pred_proba_copy.models == []


@pytest.mark.skip(
    "skipping for now as dense tooling has not been fully updated, in particular now the case where "
    "some models are available only validation and not test splits is not supported anymore."
)
@pytest.mark.parametrize(
    "cls",
    [
        TabularPredictionsMemmap,
        TabularPredictionsInMemory,
    ],
)
def test_sparse_to_dense(cls):
    """Tests sparse input."""
    num_models = 13
    num_folds = 3
    dataset_shapes = {
        "d1": ((20,), (50,)),
        "d2": ((10,), (5,)),
        "d3": ((4, 3), (8, 3)),
    }
    models = [f"{i}" for i in range(num_models)]
    pred_dict = generate_artificial_dict(num_folds, models, dataset_shapes)
    pred_dict["d1"].pop(0)
    pred_dict["d1"][1]["pred_proba_dict_val"].pop("8")
    pred_dict["d1"][1]["pred_proba_dict_test"].pop("8")
    pred_dict["d2"][1]["pred_proba_dict_test"].pop("5")
    pred_dict["d2"][1]["pred_proba_dict_val"].pop("5")
    pred_dict["d3"].pop(2)
    with tempfile.TemporaryDirectory() as tmpdirname:
        pred_proba: TabularModelPredictions = cls.from_dict(pred_dict=pred_dict, output_dir=tmpdirname)

        init_folds = pred_proba.folds
        init_folds_dense = get_folds_dense(pred_proba)
        init_models = sorted(pred_proba.models)
        init_models_dense = sorted(get_models_dense(pred_proba))
        init_datasets = pred_proba.datasets
        assert list_folds_available(pred_proba, present_in_all=False) == [0, 1, 2]
        assert list_folds_available(pred_proba, present_in_all=True) == [1]
        assert init_folds == [1]
        assert init_folds_dense == [1]
        assert init_models == [
            "0",
            "1",
            "10",
            "11",
            "12",
            "2",
            "3",
            "4",
            "6",
            "7",
            "9",
        ]  # TODO check that this what is expected

        # TODO now fails because of validation/test gap
        assert init_models_dense == ["0", "1", "10", "11", "12", "2", "3", "4", "6", "7", "9"]
        assert sorted(init_datasets) == ["d1", "d2", "d3"]
        # TODO fix me
        # assert not is_dense_models(pred_proba)
        # assert not is_dense_folds(pred_proba)
        assert not is_empty(pred_proba)

        pred_proba_copy = copy.deepcopy(pred_proba)
        with pytest.raises(AssertionError):
            force_to_dense(pred_proba_copy, first_prune_method="task", second_prune_method="dataset")
        pred_proba_copy = copy.deepcopy(pred_proba)
        with pytest.raises(AssertionError):
            force_to_dense(pred_proba_copy, first_prune_method="task", second_prune_method="fold")

        print("Pre Dense")
        print_summary(pred_proba)

        _make_empty_and_assert(pred_proba=pred_proba)

        pred_proba.restrict_folds([0, 1])
        force_to_dense(pred_proba, first_prune_method="task", second_prune_method="dataset")

        print("Post Dense")
        print_summary(pred_proba)

        post_folds = pred_proba.folds
        post_folds_dense = get_folds_dense(pred_proba)
        post_models = pred_proba.models
        post_models_dense = list_models_available(pred_proba, present_in_all=True)
        post_datasets = pred_proba.datasets

        assert post_folds == [0, 1]
        assert post_folds_dense == post_folds
        assert post_models == init_models
        assert post_models_dense == init_models
        assert post_datasets == ["d3"]
        assert is_dense_models(pred_proba)
        assert is_dense_folds(pred_proba)
        assert not is_empty(pred_proba)

        # Additionally check that restricting to nothing results in empty

        _make_empty_and_assert(pred_proba=pred_proba)
