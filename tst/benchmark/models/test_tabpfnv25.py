from __future__ import annotations

import pytest


def test_tabpfnv25():
    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model import (
            RealTabPFNv25Model,
        )

        model_cls = RealTabPFNv25Model
        FitHelper.verify_model(
            model_cls=model_cls,
            model_hyperparameters={
                "n_estimators": 1,
            },
        )
        FitHelper.verify_model(
            model_cls=model_cls,
            model_hyperparameters={
                "n_estimators": 1,
                "finetuning_config/epochs": 1,
                "use_finetuning": True,
                "ag_args_ensemble": {"refit_folds": False},
            },
            use_larger_toy_datasets=True,
        )
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
