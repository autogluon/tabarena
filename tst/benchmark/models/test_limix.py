from __future__ import annotations

import pytest


def test_limix():
    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.benchmark.models.ag.limix.limix_model import LimiXModel

        FitHelper.verify_model(model_cls=LimiXModel, model_hyperparameters={})
    except ImportError as err:
        pytest.skip(
            "Import Error, skipping test... "
            "Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
