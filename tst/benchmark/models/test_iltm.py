from __future__ import annotations

import pytest


def test_iltm():
    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.models.iltm.model import ILTMModel

        model_cls = ILTMModel
        FitHelper.verify_model(
            model_cls=model_cls,
            model_hyperparameters={"finetuning_max_steps": 1, "n_ensemble": 1, "tree_n_estimators": 1},
        )
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... Ensure you have the proper dependencies installed to run this test:\n{err}"
        )
