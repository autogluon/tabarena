from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_HAS_LOGGED_TABPFN_LICENSE: bool = False

from tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model import TabPFNModel
from tabarena.tabarena.tabarena.benchmark.models.prep_ag.prep_mixin import ModelAgnosticPrepMixin

from autogluon.features import ArithmeticFeatureGenerator
from autogluon.features import CategoricalInteractionFeatureGenerator
from autogluon.features import OOFTargetEncodingFeatureGenerator

class PrepTabPFNModel(ModelAgnosticPrepMixin, TabPFNModel):
    ag_key = "NOTSET"
    ag_name = "NOTSET"
    ag_priority = 105
    seed_name = "random_state"

    custom_model_dir: str | None = None
    default_classification_model: str | None = "NOTSET"
    default_regression_model: str | None = "NOTSET"

    # FIXME: Crashes during model download if bagging with parallel fit.
    #  Consider adopting same download logic as TabPFNMix which doesn't crash during model download.
    # FIXME: Maybe support child_oof somehow with using only one model and being smart about inference time?
    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        time_limit: float | None = None,
        **kwargs,
    ):
        time.time()

        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.model.loading import resolve_model_path
        from torch.cuda import is_available

        is_classification = self.problem_type in ["binary", "multiclass"]

        model_base = TabPFNClassifier if is_classification else TabPFNRegressor

        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not is_available()):
            # FIXME: warn instead and switch to CPU.
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )


        hps = self._get_model_params()
        hps["device"] = device
        hps["n_jobs"] = num_cpus
        hps["categorical_features_indices"] = self._cat_indices

        prep_params = hps.pop("prep_params", {})
        X = self.preprocess(X, y=y, is_train=True, prep_params=prep_params)


        # Resolve preprocessing
        if "preprocessing/scaling" in hps:
            hps["inference_config/PREPROCESS_TRANSFORMS"] = [
                {
                    "name": scaler,
                    "global_transformer_name": hps.pop("preprocessing/global", None),
                    "categorical_name": hps.pop(
                        "preprocessing/categoricals", "numeric"
                    ),
                    "append_original": hps.pop("preprocessing/append_original", True),
                }
                for scaler in hps["preprocessing/scaling"]
            ]
        for k in [
            "preprocessing/scaling",
            "preprocessing/categoricals",
            "preprocessing/append_original",
            "preprocessing/global",
        ]:
            hps.pop(k, None)

        # Remove task specific HPs
        if is_classification:
            hps.pop("inference_config/REGRESSION_Y_PREPROCESS_TRANSFORMS", None)
        else:
            hps.pop("balance_probabilities", None)

        # Resolve model_path
        if self.custom_model_dir is not None:
            model_dir = Path(self.custom_model_dir)
        else:
            _, model_dir, _, _ = resolve_model_path(
                model_path=None,
                which="classifier" if is_classification else "regressor",
            )
            model_dir = model_dir[0]
        clf_path, reg_path = hps.pop(
            "zip_model_path",
            [self.default_classification_model, self.default_regression_model],
        )
        model_path = clf_path if is_classification else reg_path
        if model_path is not None:
            hps["model_path"] = model_dir / model_path

        # Resolve inference_config
        inference_config = {
            _k: v
            for k, v in hps.items()
            if k.startswith("inference_config/") and (_k := k.split("/")[-1])
        }
        if inference_config:
            hps["inference_config"] = inference_config
        for k in list(hps.keys()):
            if k.startswith("inference_config/"):
                del hps[k]

        # Model and fit
        self.model = model_base(**hps)
        self.model = self.model.fit(
            X=X,
            y=y,
        )

class PrepRealTabPFNv25Model(PrepTabPFNModel):
    """RealTabPFN-v2.5 version: https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report.

    We name this model RealTabPFN-v2.5 as its default checkpoints were trained on
    real-world datasets, following the naming conventions of Prior Labs.
    The extra checkpoints include models trained on only synthetic datasets as well.
    """

    ag_key = "prep_REALTABPFN-V2.5"
    ag_name = "prep_RealTabPFN-v2.5"

    default_classification_model: str | None = (
        "tabpfn-v2.5-classifier-v2.5_default.ckpt"
    )
    default_regression_model: str | None = "tabpfn-v2.5-regressor-v2.5_default.ckpt"

    @staticmethod
    def extra_checkpoints_for_tuning(problem_type: str) -> list[str]:
        """The list of checkpoints to use for hyperparameter tuning."""
        if problem_type == "classification":
            return [
                "tabpfn-v2.5-classifier-v2.5_default-2.ckpt",
                "tabpfn-v2.5-classifier-v2.5_large-features-L.ckpt",
                "tabpfn-v2.5-classifier-v2.5_large-features-XL.ckpt",
                "tabpfn-v2.5-classifier-v2.5_large-samples.ckpt",
                "tabpfn-v2.5-classifier-v2.5_real-large-features.ckpt",
                "tabpfn-v2.5-classifier-v2.5_real-large-samples-and-features.ckpt",
                "tabpfn-v2.5-classifier-v2.5_real.ckpt",
                "tabpfn-v2.5-classifier-v2.5_variant.ckpt",
            ]

        return [
            "tabpfn-v2.5-regressor-v2.5_low-skew.ckpt",
            "tabpfn-v2.5-regressor-v2.5_quantiles.ckpt",
            "tabpfn-v2.5-regressor-v2.5_real-variant.ckpt",
            "tabpfn-v2.5-regressor-v2.5_real.ckpt",
            "tabpfn-v2.5-regressor-v2.5_small-samples.ckpt",
            "tabpfn-v2.5-regressor-v2.5_variant.ckpt",
        ]
