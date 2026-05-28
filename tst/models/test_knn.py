from autogluon.tabular.testing import FitHelper
from tabarena.models.knn.model import KNNNewModel


def test_knn():
    toy_model_params = {}
    model_hyperparameters = toy_model_params
    model_cls = KNNNewModel
    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
