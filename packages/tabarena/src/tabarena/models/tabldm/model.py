from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

_HF_REPO = "occams/Xiaomi-TabLDM"
_CLASSIFIER_CHECKPOINT = "checkpoints/clf_default.ckpt"
_REGRESSOR_CHECKPOINT = "checkpoints/reg_default.ckpt"
#: Hub revision the hosted results were produced with. The regression checkpoint was replaced on
#: the Hub on 2026-09-28; the Xiaomi-TabLDM+ system pins the newer revision.
_HF_REVISION = "0e10427273bc862fb60fe7df84c89e5298e16cc7"
#: Classifier defaults of the estimator the hosted results were produced with (``6773a30d``).
#: Upstream changed them later; the regressor's plain-path defaults are unchanged.
_CLASSIFIER_DEFAULTS = {
    "n_estimators": 8,
    "feat_shuffle_method": "latin",
    "cat_random_encode": False,
    "use_amp": "auto",
}


def _checkpoint_path(filename: str) -> Path:
    """Download ``filename`` at the pinned Hub revision (a no-op once cached) and return its path."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=_HF_REPO, filename=filename, revision=_HF_REVISION))


class TabLDMModel(AbstractTorchModel):
    """TabLDM: a tabular foundation model with a dual-stream column embedder and MoE backbone.

    Runs the plain inference path of TabLDM's sklearn estimators (``TabLDMClassifier`` /
    ``TabLDMRegressor``) with ``enhance_candidates=False``: the candidate enhancement fits its
    weights on a holdout it carves out of the training data, which a model may not do. That path
    is benchmarked as the Xiaomi-TabLDM+ system.

    Codebase: pip name ``Xiaomi-TabLDM``, import name ``tabldm``, installed from GitHub since
    it is not on PyPI.
    Checkpoints: https://huggingface.co/occams/Xiaomi-TabLDM
    License: Apache-2.0 (Copyright Xiaomi Corporation)
    """

    ag_key = "TA-XIAOMI-TABLDM"
    warmup_modules: ClassVar[tuple[str, ...]] = ("tabldm",)
    ag_name = "TA-Xiaomi-TabLDM"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1
    #: TabLDM builds its network inside ``_load_model``, which ``fit`` calls; one build per checkpoint
    #: and device per process. A key-value cache writes into the network, so that configuration
    #: builds its own.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader=("tabldm:TabLDMClassifier._load_model", "tabldm:TabLDMRegressor._load_model"),
        key=("checkpoint_version", "model_path"),
        disabled_by=("kv_cache",),
    )
    #: Knobs that make the warm-up's dummy fit cheap without touching the network.
    cheap_hyperparameters: ClassVar[dict] = {"n_estimators": 1}
    # Sequential fold fitting avoids contention on the shared HF checkpoint cache; refitting one
    # model on all data (like the other TFM wrappers: TabICL, TabSwift, LimiX, ...) gives faster
    # inference at similar quality to the bagged ensemble for an in-context-learning model.
    _default_ag_args_ensemble_extra = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
        "vary_seed_across_folds": False,  # pretrained: the fold models are the refit that predicts
    }

    def get_model_cls(self):
        if self.problem_type in ("binary", "multiclass"):
            from tabldm import TabLDMClassifier

            return TabLDMClassifier
        from tabldm import TabLDMRegressor

        return TabLDMRegressor

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        """Fit TabLDM.

        As an in-context-learning foundation model there is no training loop and no early
        stopping, so (like the other TFM wrappers) ``X_val``/``y_val`` and ``time_limit`` are
        intentionally ignored. Categorical columns and missing values need no upfront handling
        here: the estimator's own ``TransformToNumerical`` preprocessing ordinal-encodes
        categorical dtypes and mean-imputes numeric NaNs internally when given a DataFrame, and
        for regression it standardizes/inverse-transforms the target itself, so ``X``/``y`` are
        passed straight through.
        """
        import torch

        available_num_gpus = ResourceManager.get_gpu_count_torch(cuda_only=True)
        if num_gpus > available_num_gpus:
            raise AssertionError(
                f"Fit specified to use {num_gpus} GPU, but only {available_num_gpus} "
                "CUDA GPUs are available. Please activate CUDA or switch to CPU usage.",
            )
        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not torch.cuda.is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        model_cls = self.get_model_cls()
        is_classification = self.problem_type in ("binary", "multiclass")
        hps = {
            **(_CLASSIFIER_DEFAULTS if is_classification else {}),
            **self._get_model_params(),
            "enhance_candidates": False,
        }
        model_path = _checkpoint_path(_CLASSIFIER_CHECKPOINT if is_classification else _REGRESSOR_CHECKPOINT)

        X = self.preprocess(X, y=y, is_train=True)

        self.model = model_cls(device=device, n_jobs=num_cpus, model_path=model_path, **hps)
        self.model.fit(X, y)

    def get_device(self) -> str:
        return self.model.device_.type

    def _set_device(self, device: str):
        device = self.to_torch_device(device)
        self.model.device_ = device
        self.model.model_ = self.model.model_.to(device)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def prefetch_weights(cls) -> None:
        """Pre-download both checkpoints (classifier and regressor) at the pinned Hub revision."""
        for filename in (_CLASSIFIER_CHECKPOINT, _REGRESSOR_CHECKPOINT):
            _checkpoint_path(filename)
