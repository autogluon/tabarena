from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

HF_REPO = "h2oai/tabh2o-v1.1"
#: Commit pinned so the checkpoint fetched here never silently changes if the repo's default
#: branch moves. Bump deliberately (with a note on what changed) when picking up a newer checkpoint.
HF_REVISION = "7542639fdcae6b76f9107dee27f5d119b6080e4e"


class TabH2OModel(AbstractTorchModel):
    """TabH2O is a tabular foundation model by H2O.ai for classification and regression. It predicts
    the test rows from the labeled training rows in one forward pass (in-context learning).

    Model card: https://huggingface.co/h2oai/tabh2o-v1.1
    Codebase: https://pypi.org/project/h2ofm/
    License: code Apache-2.0; weights H2O Model Non-Commercial Licence v1.0
    """

    ag_key = "TA-TABH2O"
    warmup_modules: ClassVar[tuple[str, ...]] = ("h2ofm",)
    ag_name = "TA-TabH2O"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 1
    _default_auxiliary_params_extra = {"valid_raw_types": ["int", "float", "category"]}
    _default_ag_args_ensemble_extra = {"fold_fitting_strategy": "sequential_local", "refit_folds": True}
    #: ``TabH2O.from_pretrained`` builds the network; one build per checkpoint and device per process.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="h2ofm:TabH2O.from_pretrained", key=("pretrained_model_name_or_path", "revision")
    )
    #: Knobs that make the warm-up's dummy fit cheap without touching the network.
    cheap_hyperparameters: ClassVar[dict] = {"n_estimators": 1}

    def _fit(self, X: pd.DataFrame, y: pd.Series, num_cpus: int = 1, num_gpus: int = 0, **kwargs):
        from h2ofm import TabH2O

        # In-context learning: no training loop, so no validation split or time limit to honor.
        device = self._resolve_fit_device(num_gpus=num_gpus)
        self.model = TabH2O.from_pretrained(HF_REPO, revision=HF_REVISION, device=device)
        self._X_train = self.preprocess(X, y=y)
        self._y_train = y
        self._inference_threads = num_cpus

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        import torch

        X = self.preprocess(X, **kwargs)
        params = self._get_model_params()
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self._inference_threads)
            if self.problem_type == "regression":
                params.pop("softmax_temperature", None)
                return self.model.regress(self._X_train, self._y_train, X, **params).mean
            result = self.model.classify(self._X_train, self._y_train, X, **params)
        finally:
            torch.set_num_threads(previous_threads)
        # A bagged fold can miss a class; its column stays zero.
        proba = np.zeros((len(X), self.num_classes), dtype=result.probabilities.dtype)
        proba[:, result.classes] = result.probabilities
        return self._convert_proba_to_unified_form(proba)

    def _set_default_params(self):
        self._set_default_param_value("n_estimators", 8)

    def get_device(self) -> str:
        return self.model.device.type

    def _set_device(self, device: str):
        self.model._model.to(self.to_torch_device(device))

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}


def prefetch_weights() -> str:
    """Download the pinned TabH2O checkpoint into the Hugging Face cache; return the snapshot folder."""
    from huggingface_hub import snapshot_download

    return snapshot_download(HF_REPO, revision=HF_REVISION, allow_patterns=["config.json", "model.safetensors"])
