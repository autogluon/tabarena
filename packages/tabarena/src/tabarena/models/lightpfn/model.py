from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
from autogluon.core.constants import BINARY, MULTICLASS
from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tabarena.utils.wrapper_utils import import_many_class_classifier

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


class _LightPFNOutputCodeEstimator:
    """The base estimator of the ``ManyClassClassifier`` output coding: one ``LightPFNClassifier`` per code row.

    ``ManyClassClassifier`` validates the frame into a NumPy array, while LightPFN reads the category dtype of
    its columns. This estimator keeps the training frame's columns and dtypes and rebuilds the frame around
    every array it is given; clones share them.
    """

    def __init__(self, *, params: Mapping, columns: pd.Index, dtypes: pd.Series):
        self._params = dict(params)
        self._columns = columns
        self._dtypes = dtypes

    def __sklearn_clone__(self) -> _LightPFNOutputCodeEstimator:
        return _LightPFNOutputCodeEstimator(params=self._params, columns=self._columns, dtypes=self._dtypes)

    def _frame(self, X) -> pd.DataFrame:
        return pd.DataFrame(X, columns=self._columns).astype(self._dtypes)

    def fit(self, X, y) -> _LightPFNOutputCodeEstimator:
        from lightpfn import LightPFNClassifier

        self._estimator = LightPFNClassifier(**self._params).fit(self._frame(X), np.asarray(y))
        self.classes_ = self._estimator.classes_
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._estimator.predict_proba(self._frame(X))


class LightPFNModel(AbstractTorchModel):
    """LightPFN is a small tabular foundation model for classification (2 to 10 classes): an in-context
    learner with 4.6M parameters, pretrained only on synthetic data (structural causal graph and rule
    priors), designed to run on a CPU as well as on a GPU.

    Paper: A Sling Against Giants: LightPFN, a 4.6M-parameter tabular in-context classifier designed to stay small
    Authors: Giorgio Ottoboni
    Codebase: https://github.com/GioOtto/LightPFN
    License: Apache-2.0 (code and weights)

    Categorical columns arrive with the category dtype; the library encodes them as ordinal codes of the
    training categories, with unseen and missing values as NaN, which the model reads as missing. The
    classifier's fit has no training loop, early stopping or internal validation split, so ``X_val`` and
    ``time_limit`` are not used, as for the other in-context models.

    The checkpoint's classification head is ten classes wide. Above ``many_class_threshold`` (an
    ``ag_args_fit`` parameter, ten by default) the fit wraps a :class:`_LightPFNOutputCodeEstimator` in the
    ``ManyClassClassifier`` of tabpfn-extensions, which codes the labels over that many symbols and fits one
    LightPFN estimator per code row.
    """

    ag_key = "TA-LIGHTPFN"
    ag_name = "TA-LightPFN"
    ag_priority = 65
    seed_name = "random_state"
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "lightpfn",
        "lightpfn.sklearn",
        "lightpfn.checkpoint",
        "lightpfn.model.lightpfn",
        "safetensors.torch",
        "tabpfn_extensions.many_class",
    )
    _supported_problem_types = ["binary", "multiclass"]
    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 0  # the library runs on the CPU as well
    #: No ``max_classes`` cap: above ``many_class_threshold`` (the head's width) the fit output-codes the labels.
    _default_auxiliary_params_extra = {
        "valid_raw_types": ["int", "float", "category"],
        "max_classes": None,
        "many_class_threshold": 10,
    }
    _default_ag_args_ensemble_extra = {"fold_fitting_strategy": "sequential_local", "refit_folds": True}
    #: ``LightPFNClassifier.fit`` builds its network in ``_initialize_backend``: the released weights at the
    #: revision pinned in the package (or ``checkpoint`` / ``repo_id`` + ``revision``) and their folded copy for
    #: inference (``fold``). Fitting never writes into the network; a user-supplied torch ``model`` keeps its own.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="lightpfn.sklearn:LightPFNClassifier._initialize_backend",
        key=("checkpoint", "repo_id", "revision", "fold"),
        disabled_by=("model",),
    )
    #: Knobs that make the warm-up's dummy fit cheap without touching the network.
    cheap_hyperparameters: ClassVar[dict] = {"n_estimators": 1}
    _use_many_class: bool = False

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        import torch
        from lightpfn import LightPFNClassifier

        device = self._resolve_fit_device(num_gpus=num_gpus)
        self._inference_threads = num_cpus
        previous_threads = torch.get_num_threads()
        try:
            # the library's n_threads would set the process-wide thread count; set it here and restore it
            torch.set_num_threads(num_cpus)
            params = self._get_model_params()
            # Resource allocations take precedence over estimator hyperparameters.
            params.update(device=device, n_threads=None)
            X = self.preprocess(X)
            many_class_threshold = self.params_aux.get("many_class_threshold", 10)
            self._use_many_class = (
                self.problem_type in [BINARY, MULTICLASS]
                and self.num_classes is not None
                and self.num_classes > many_class_threshold
            )
            if self._use_many_class:
                ManyClassClassifier = import_many_class_classifier()

                logger.log(
                    20,
                    f"\tLightPFN: {self.num_classes} classes exceed the checkpoint's {many_class_threshold}-class "
                    "head, fitting ManyClassClassifier (output coding) around it.",
                )
                base = _LightPFNOutputCodeEstimator(params=params, columns=X.columns, dtypes=X.dtypes)
                self.model = ManyClassClassifier(
                    estimator=base, alphabet_size=many_class_threshold, random_state=params.get(self.seed_name, 0)
                ).fit(X, y)
            else:
                self.model = LightPFNClassifier(**params)
                self.model.fit(X, y)
        finally:
            torch.set_num_threads(previous_threads)

    def _predict_proba(self, X, **kwargs):
        import torch

        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self._inference_threads)
            return super()._predict_proba(X, **kwargs)
        finally:
            torch.set_num_threads(previous_threads)

    def _set_default_params(self):
        self._set_default_param_value("n_estimators", 4)

    def get_device(self) -> str:
        if self._use_many_class:
            # Output coding keeps no fitted network: every code row is fit at predict time on this device.
            return self.model.estimator._params["device"]
        return self.model.device_

    def _set_device(self, device: str):
        if self._use_many_class:
            self.model.estimator._params["device"] = device
        else:
            self.model.to(device)

    def _ag_params(self) -> set[str]:
        return super()._ag_params() | {"many_class_threshold"}

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}


def prefetch_weights() -> str:
    """Download the released weights at the commit pinned in the installed package and return the snapshot folder."""
    from huggingface_hub import snapshot_download
    from lightpfn.checkpoint import pretrained_spec

    spec = pretrained_spec()
    return snapshot_download(
        spec["repo_id"], revision=spec["revision"], allow_patterns=["config.json", "model.safetensors"]
    )
