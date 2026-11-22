from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

_HF_REPO_ID = "josifgrabocka/ftfm"
#: Commit pinned so the checkpoint fetched here never silently changes when the repository's default
#: branch moves (the library overwrites ``model.safetensors`` in place for new checkpoints). This is
#: FTFM-1, stage C at update 130,000. Bump deliberately when picking up a newer checkpoint.
_HF_REVISION = "a8362b12392a80d2e4618e91e6ce02b0bed2781d"

#: Bytes one fitted member context keeps per support row with the pinned checkpoint (fp32, dim 512,
#: 16 layers, equivariant head over 10 classes); ``ftfm.serving.estimate_context_bytes`` per row.
_CONTEXT_BYTES_PER_ROW = 67_624
_MAX_CLASSES = 10


class FTFMModel(AbstractTorchModel):
    """FTFM, a factorized tabular foundation model for in-context classification.

    Cell representations come from factorizing one contextual latent per row against one per column,
    instead of cell-to-cell attention. The support-side computation is cached once at ``fit`` and
    reused for every query. The default estimator predicts with a TabICLv2-style ensemble of eight
    views; a table with more than ten classes is fit through the library's error-correcting output
    codes, one context set per codeword.

    ``cache_context`` decides when the contexts are built. True (the default) builds them in ``fit``,
    so a predict only decodes. False keeps the support rows and builds the contexts inside every
    predict call, releasing them after it; the default config uses that for the bagged fold models,
    which predict once and are dropped, and True for the refit model.

    Authors: Josif Grabocka
    Codebase: https://github.com/machinelearningnuremberg/ftfm-official
    Weights: https://huggingface.co/josifgrabocka/ftfm
    License: code Apache-2.0, weights CC BY-NC 4.0 (non-commercial).
    """

    ag_key = "TA-FTFM"
    ag_name = "TA-FTFM"
    ag_priority = 65
    seed_name = "random_state"
    warmup_modules: ClassVar[tuple[str, ...]] = ("ftfm",)
    _supported_problem_types = ["binary", "multiclass"]
    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 1
    _default_ag_args_ensemble_extra = {"fold_fitting_strategy": "sequential_local", "refit_folds": True}
    #: The ensemble's member ``FTFMClassifier`` builds the network in ``_load_model``, which its first
    #: ``fit`` calls; one build per checkpoint and device per process. Fitted contexts are kept on the
    #: estimator and installed per decode, so the network carries no per-fit state.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="ftfm.estimator:FTFMClassifier._load_model",
        key=("repo_id", "checkpoint_version", "model_path"),
    )
    #: Knobs that make the warm-up's dummy fit cheap without touching the network.
    cheap_hyperparameters: ClassVar[dict] = {"n_estimators": 1}

    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_cpus=1, num_gpus=0, **kwargs):
        import torch

        from tabarena.models.ftfm._estimators import FTFMClassifier

        # In-context learning: no training loop, early stopping or validation split.
        device = self._resolve_fit_device(num_gpus=num_gpus)
        self._inference_threads = num_cpus
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(num_cpus)
            params = self._get_model_params()
            cache_context = params.pop("cache_context")
            self.model = FTFMClassifier(device=device, model_path=self.prefetch_weights(), **params)
            if cache_context:
                self.model.fit(self.preprocess(X), y)
            else:
                self.model.defer_fit(self.preprocess(X), y)
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
        self._set_default_param_value("n_estimators", 8)
        self._set_default_param_value("cache_context", True)

    @classmethod
    def prefetch_weights(cls) -> Path:
        """Return the folder holding the pinned checkpoint, downloading it into the Hub cache once."""
        from ftfm import download_checkpoint

        return download_checkpoint(repo_id=_HF_REPO_ID, revision=_HF_REVISION)

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        num_classes: int | None = None,
        **kwargs,
    ) -> int:
        """The fitted contexts dominate: one per ensemble member, and per codeword above ten classes.

        They stay in host memory (offloaded from the GPU between decodes) and are saved with the model,
        to AutoGluon's tensor file beside ``model.pkl``.
        """
        hyperparameters = hyperparameters or {}
        n_contexts = hyperparameters.get("n_estimators", 8)
        if num_classes is not None and num_classes > _MAX_CLASSES:
            redundancy = hyperparameters.get("ecoc_redundancy", 4)
            n_words = max(2, redundancy * math.ceil(math.log(num_classes) / math.log(_MAX_CLASSES)))
            n_contexts = n_words * (hyperparameters.get("ecoc_estimators") or n_contexts)
        context_mem = n_contexts * X.shape[0] * _CONTEXT_BYTES_PER_ROW
        return context_mem + 3 * get_approximate_df_mem_usage(X).sum() + 1e9

    def get_device(self) -> str:
        return str(self.model.member.device)

    def _set_device(self, device: str):
        member = self.model.member
        member.device = device
        member.model = member.model.to(device)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
