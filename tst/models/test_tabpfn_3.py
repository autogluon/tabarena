from __future__ import annotations

import pytest


def test_tabpfn_3():
    try:
        from autogluon.tabular.testing import FitHelper

        from tabarena.models.tabpfn_3.model import (
            TabPFN3Model,
        )

        model_cls = TabPFN3Model
        FitHelper.verify_model(
            model_cls=model_cls,
            model_hyperparameters={
                "n_estimators": 1,
                "device": "cpu",
            },
        )
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... Ensure you have the proper dependencies installed to run this test:\n{err}",
        )
