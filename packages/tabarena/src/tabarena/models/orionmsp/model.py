from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from autogluon.common.utils.pretrained_weights import (
    PretrainedWeightsUnavailableError,
    fetch_allowed,
    unavailable_message,
)
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tabarena.models._shared_weights_model import CheckpointSpec, SharedWeightsModelMixin, SharedWeightsSpec
from tabarena.models.prefetch import WeightsUnavailableError

if TYPE_CHECKING:
    from types import ModuleType

    import pandas as pd

    from tabarena.models._weights import WeightsKey


logger = logging.getLogger(__name__)

_HF_REPO_ID = "Lexsi/Orion-MSP"
_DEFAULT_CHECKPOINT_FILE = "OrionMSP-classifier-v1.5-202603.ckpt"
#: Commit pinned so checkpoints fetched here never silently change if the
#: repo's default branch moves. Bump deliberately (with a note on what
#: changed) when picking up newer checkpoints.
_HF_REVISION = "8b712f6ff699750f7ac5825f31e89e6f6f161577"
#: The checkpoint-relevant literals of ``_set_default_params``.
DEFAULT_PARAMS: dict[str, Any] = {
    "checkpoint_version": _DEFAULT_CHECKPOINT_FILE,
    "allow_auto_download": True,
}
#: ``inference_config`` switches that make the module-held ``InferenceManager`` cache training data.
_KV_CACHE_KEYS: tuple[str, ...] = ("enable_kv_cache", "cache_trainset_representation")
_ESTIMATORS_MODULE = "tabarena.models.orionmsp._estimators"


def _estimators() -> ModuleType:
    """The library-importing half of the wrapper, imported on first use."""
    return importlib.import_module(_ESTIMATORS_MODULE)


def uses_kv_cache(inference_config: Any) -> bool:
    """Whether an ``inference_config`` hyperparameter turns on training-data caching in the network.

    ``enable_kv_cache`` and ``cache_trainset_representation`` make the ``InferenceManager`` objects
    that live on the module keep KV pairs of the support set, which would leak one child's data into
    every other holder of a shared network. A dict form is inspected per sub-config; a prebuilt
    ``InferenceConfig`` object cannot be inspected without the library and is treated as caching.
    """
    if inference_config is None:
        return False
    if not isinstance(inference_config, dict):
        return True
    for sub_config in inference_config.values():
        if isinstance(sub_config, dict) and any(sub_config.get(key) for key in _KV_CACHE_KEYS):
            return True
    return False


def _clear_transient_state(network: Any) -> None:
    """Drop the hierarchical class tree a many-class forward leaves on ``icl_predictor``.

    Above the checkpoint's ``max_classes`` the library builds a ``ClassNode`` tree holding the
    support representations and labels on the module and reads it in the same forward; it is rebuilt
    before every use, so clearing it after a predict changes nothing and keeps one child's data off a
    module other children share.
    """
    icl = getattr(network, "icl_predictor", None)
    if icl is not None and getattr(icl, "root", None) is not None:
        icl.root = None


class OrionMSPModel(SharedWeightsModelMixin, AbstractTorchModel):
    """Orion-MSP v1.5: Multi-Scale Sparse Attention for Tabular In-Context Learning.

    We have to use the code from TabTune, as the standalone package does not support the newest
    checkpoints. The standalone package is hardcoded to 1.0 checkpoints.

    Codebase: https://github.com/Lexsi-Labs/Orion-MSP
    Hugging Face: https://huggingface.co/Lexsi/Orion-MSP
    TabTune (wrapper used here): https://github.com/Lexsi-Labs/TabTune
    Paper: Orion-MSP: Multi-Scale Sparse Attention for Tabular In-Context Learning
        (https://arxiv.org/abs/2511.02818)
    Authors: Mohamed Bouadi, Pratinav Seth, Aditya Tanna, Vinay Kumar Sankarapu
    License: MIT
    """

    ag_key = "TA-ORION-MSP"
    ag_name = "TA-OrionMSP"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1

    #: Modules the timed fit would otherwise import for the first time: the library classifier and
    #: the embedding module the class patch touches (both pulled in by ``_estimators``), the seam
    #: module itself and the Hub client the resolver uses.
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "tabtune.models.orionmsp_v15.sklearn.classifier",
        "tabtune.models.orionmsp_v15.model.embedding",
        _ESTIMATORS_MODULE,
        "huggingface_hub",
        "huggingface_hub.errors",
    )
    #: Cheapness knob for the warm-up dummy fit; the ensemble size never touches the network, so the
    #: primed key is unaffected.
    warmup_dummy_fit_hyperparameters: ClassVar[dict] = {"n_estimators": 1}

    shared_weights_spec: ClassVar[SharedWeightsSpec] = SharedWeightsSpec(
        library="orionmsp",
        checkpoint=CheckpointSpec(
            repo_id=_HF_REPO_ID,
            filename_param="checkpoint_version",
            revision=_HF_REVISION,
            default_filename=_DEFAULT_CHECKPOINT_FILE,
        ),
        # The module is built in fp32 whatever the file stores (``use_amp`` is an autocast at
        # forward time) and the architecture comes from the checkpoint, so there are no flags.
        variant="classifier",
        default_params=DEFAULT_PARAMS,
        # A user ``model_path`` selects custom weights and keeps the library loader; an
        # ``inference_config`` that caches training data would store one child's support set on
        # the shared module.
        disable_when=("model_path", lambda hps: uses_kv_cache(hps.get("inference_config"))),
        unshareable_examples=(
            {"inference_config": {"ICL_CONFIG": {"enable_kv_cache": True}}},
            {"inference_config": {"COL_CONFIG": {"cache_trainset_representation": True}}},
        ),
        download_param="allow_auto_download",
        network_attr="model_",
        seam="load_model",
        # The library captures ``device_`` into the three ``InferenceManager`` configs at fit time
        # and reads them on every forward, so a device change has to update all of them.
        device_attrs=(
            ("device_", "torch"),
            ("device", "str"),
            ("inference_config_.COL_CONFIG.device", "torch"),
            ("inference_config_.ROW_CONFIG.device", "torch"),
            ("inference_config_.ICL_CONFIG.device", "torch"),
        ),
    )
    # Refitting one model on all data gives faster inference at similar quality for an in-context model.
    _default_ag_args_ensemble_extra: ClassVar[dict] = {"refit_folds": True}

    @classmethod
    def _build_shared_weights(cls, key: WeightsKey) -> Any:
        """Build the network for ``key`` exactly as the library's ``_load_model`` does.

        The class-level positional-embedding patch is installed first so a network primed by the
        warm-up runs the same embedding code as one built inside a fit.
        """
        est = _estimators()
        est.patch_col_embedder_pos_emb()
        return est.build_network(key.checkpoint, key.device)

    def _checkpoint_path_for_fit(self, hyperparameters: dict) -> str:
        """Local path of the pinned checkpoint for a fit that loads its own network.

        Pre-resolving keeps the library off the network when the file is cached and pins the
        revision; a download needs both ``ag.fetch_pretrained_weights`` and the configuration's
        ``allow_auto_download``.
        """
        spec = self.shared_weights_spec
        hps = spec.overlay(type(self), hyperparameters)
        allow = fetch_allowed(self.aux_params.fetch_pretrained_weights, stage="fit") and bool(
            hps.get("allow_auto_download", True)
        )
        try:
            return spec.checkpoint.resolve(
                hyperparameters=hps, variant="classifier", allow_download=allow, cls=type(self)
            ).path
        except WeightsUnavailableError as exc:
            raise PretrainedWeightsUnavailableError(
                unavailable_message(model_name=self.name, stage="fit", location=str(exc))
            ) from exc

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
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

        est = _estimators()
        # See `_estimators.patch_col_embedder_pos_emb`: works around an upstream shape-discriminator
        # bug that breaks inference whenever H != T.
        est.patch_col_embedder_pos_emb()

        if self.problem_type not in ["binary", "multiclass"]:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")

        hps = self._get_model_params()

        # Needs up to 400GB VRAM for datasets with 1k features.
        # Adjust batch size as needed.
        if X.shape[1] > 500:
            hps["batch_size"] = 1  # avoid OOM for wide datasets; can be slow but is a fallback

        key, payload = self._acquire_shared_weights(device=device)
        if key is not None:
            hps["model_path"] = key.checkpoint
        elif hps.get("model_path") is None:
            hps["model_path"] = self._checkpoint_path_for_fit(hps)
        self.model = est.SharedOrionMSPv15Classifier(**hps, device=device)
        if key is not None:
            self.model.use_shared_weights(key, payload, type(self)._load_shared_weights)

        X = self.preprocess(X, y=y)
        self.model = self.model.fit(
            X=X,
            y=y,
        )

    def _predict_proba(self, X, **kwargs):
        try:
            return super()._predict_proba(X, **kwargs)
        finally:
            if self._shared_key is not None and self._network_attached():
                _clear_transient_state(self.model.model_)

    def _set_default_params(self):
        default_params = {
            **DEFAULT_PARAMS,
            # AMP introduces ~1e-4 fp16 jitter between single-row and batch
            # predict_proba calls, which breaks AutoGluon's determinism check.
            # "use_amp": False, # disabled for now due to VRAM issues
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
