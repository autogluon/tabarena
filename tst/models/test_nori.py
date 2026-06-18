from __future__ import annotations

import pytest


def test_nori():
    try:
        from autogluon.tabular.testing import FitHelper

        from tabarena.models.nori.model import NoriModel

        model_cls = NoriModel
        # Nori only supports regression.
        FitHelper.verify_model(
            model_cls=model_cls,
            model_hyperparameters={},
            problem_types=["regression"],
        )
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... Ensure you have the proper dependencies installed to run this test:\n{err}"
        )
