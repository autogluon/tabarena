from __future__ import annotations

import tempfile

import numpy as np

from tabarena.predictions import TabularPredictionsMemmap
from tabarena.predictions.tabular_predictions import TabularModelPredictions
from tabarena.repository import EvaluationRepository
from tabarena.simulation.context_artificial import load_context_artificial
from tabarena.utils.test_utils import generate_artificial_dict

num_models = 13
num_folds = 3
dataset_shapes = {
    "d1": ((20,), (50,)),
    "d2": ((10,), (5,)),
    "d3": ((4, 3), (8, 3)),
}
models = [f"{i}" for i in range(num_models)]


def test_keep_only_models_in_both_validation_and_test():
    pred_dict = generate_artificial_dict(num_folds, models, dataset_shapes)
    pred_dict["d2"][1]["pred_proba_dict_val"].pop("2")
    TabularModelPredictions._keep_only_models_in_both_validation_and_test(pred_dict)

    # check that the intersection works as expected, model "2" should not be present in the test split as it was
    # absent from the validation split
    assert "2" not in pred_dict["d2"][1]["pred_proba_dict_test"]


def test_tabular_predictions():
    pred_dict = generate_artificial_dict(num_folds, models, dataset_shapes)

    # remove the second model from the second dataset and the second fold
    pred_dict["d2"][1]["pred_proba_dict_val"].pop("2")
    # pred_dict["d2"][1]['pred_proba_dict_test'].pop("2")

    with tempfile.TemporaryDirectory() as tmpdirname:
        preds = TabularPredictionsMemmap.from_dict(pred_dict, output_dir=tmpdirname)
        pred_expected = preds.predict_val(dataset="d2", fold=1, models=["3"], model_fallback="3")
        pred_obtained = preds.predict_val(dataset="d2", fold=1, models=["2"], model_fallback="3")
        assert np.allclose(pred_expected, pred_obtained)


def test_repository():
    zsc, _configs_full, pred_proba, zeroshot_gt = load_context_artificial()

    with tempfile.TemporaryDirectory() as tmpdirname:
        # remove NeuralNetFastAI_r1 from fold 1 of abalone...
        pred_dict = pred_proba.pred_dict
        pred_dict["abalone"][1]["pred_proba_dict_val"].pop("NeuralNetFastAI_r1")
        repo = EvaluationRepository(
            zeroshot_context=zsc,
            tabular_predictions=TabularPredictionsMemmap.from_dict(pred_dict, tmpdirname),
            ground_truth=zeroshot_gt,
        )

        # ... and make sure we retrieve the result of the fallback when querying results from missing NeuralNetFastAI_r1
        repo.set_config_fallback("NeuralNetFastAI_r2")
        pred_expected = repo.predict_val(dataset="abalone", fold=1, config="NeuralNetFastAI_r2")
        pred_obtained = repo.predict_val(dataset="abalone", fold=1, config="NeuralNetFastAI_r1")
        assert np.allclose(pred_expected, pred_obtained)
