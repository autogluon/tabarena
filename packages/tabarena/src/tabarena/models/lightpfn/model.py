from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd


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
    )
    _supported_problem_types = ["binary", "multiclass"]
    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 0  # the library runs on the CPU as well
    _default_auxiliary_params_extra = {"valid_raw_types": ["int", "float", "category"], "max_classes": 10}
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
            self.model = LightPFNClassifier(**params)
            self.model.fit(self.preprocess(X), y)
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
        return self.model.device_

    def _set_device(self, device: str):
        self.model.to(device)

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
