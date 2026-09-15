from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tabarena.models import prefetch as _hub
from tabarena.models._shared_estimators import check_payload_device, detach_by_path
from tabarena.models._shared_weights_model import (
    CheckpointSpec,
    ResolvedCheckpoint,
    SharedWeightsModelMixin,
    SharedWeightsSpec,
)
from tabarena.models._weights import normalize_device

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import pandas as pd
    import torch


logger = logging.getLogger(__name__)

#: Hugging Face repository holding the ConTextTab checkpoints.
REPO_ID = "SAP/sap-rpt-1-oss"
#: Commit of :data:`REPO_ID` every resolution pins, so the checkpoint never silently changes when the
#: repository's default branch moves. Bump deliberately (with a note on what changed) when picking up
#: a newer checkpoint.
HF_REVISION = "bc2c99cc541eeac9d1e10ae79da192d38b9fb27b"
#: The library's default checkpoint file; the ``checkpoint`` hyperparameter selects another file of the repository.
DEFAULT_CHECKPOINT = "2025-11-04_sap-rpt-one-oss.pt"
#: Sentence-embedding model the library's tokenizer loads for column names and text cells
#: (``sap_rpt_oss.data.tokenizer.Tokenizer.sentence_embedding_model_name``).
EMBEDDER_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
#: Commit of :data:`EMBEDDER_REPO_ID` the shared path pins; the library resolves the branch ``main``,
#: which points at this commit. Bump deliberately.
EMBEDDER_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
#: The files ``AutoModel.from_pretrained`` and ``AutoTokenizer.from_pretrained`` read for the embedder;
#: the repository also ships ONNX, OpenVINO and TensorFlow exports that are never needed.
EMBEDDER_FILES: tuple[str, ...] = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


@dataclass(frozen=True)
class SharedSAPRPTWeights:
    """The immutable objects one SAP-RPT-OSS estimator takes from the shared-weights registry.

    Args:
        module: The ``RPT`` network in eval mode with gradients disabled, on ``device`` in ``dtype``.
        sentence_embedder: The library's ``SentenceEmbedder`` (MiniLM model plus its tokenizer) on
            ``device``; its output feeds the network's text and column-name embeddings.
        device: The device both objects live on (what the estimator stores as ``device``).
        dtype: Parameter dtype of ``module`` (what the estimator stores as ``dtype``).
        checkpoint_path: The resolved checkpoint file ``module`` was built from.
        embedder_dir: The resolved snapshot directory the embedder was loaded from.
    """

    module: torch.nn.Module
    sentence_embedder: Any
    device: torch.device
    dtype: torch.dtype
    checkpoint_path: str
    embedder_dir: str


def weights_dtype_name(device_type: str) -> str:
    """The parameter dtype the library casts the network to on ``device_type``, as a string.

    Mirrors ``SAP_RPT_OSS_Estimator.__init__``: ``float32`` on the CPU; on CUDA ``bfloat16`` for
    compute capability 8 and above, else ``float16``. Without a CUDA device to inspect the CUDA
    answer is ``bfloat16`` (every current data-center GPU); such a key names a device this host
    cannot serve and is never built here.
    """
    if normalize_device(device_type) != "cuda":
        return "float32"
    import torch

    if not torch.cuda.is_available():
        return "bfloat16"
    return "bfloat16" if torch.cuda.get_device_capability(0)[0] >= 8 else "float16"


def _checkpoint_is_not_a_repo_file(hyperparameters: Mapping[str, Any]) -> bool:
    """Every hyperparameter but ``checkpoint`` lives on the estimator; only a non-string ``checkpoint`` is unshareable."""
    return not isinstance(hyperparameters.get("checkpoint"), str)


def resolve_embedder_dir(*, allow_download: bool = True) -> str:
    """Local snapshot directory of the sentence embedder at :data:`EMBEDDER_REVISION`, the cache first.

    Only :data:`EMBEDDER_FILES` are fetched, and a cached snapshot counts only when all of them
    are present.

    Raises:
        tabarena.models.prefetch.WeightsUnavailableError: The snapshot is missing or incomplete
            and ``allow_download`` is False.
    """
    return _hub.resolve_hf_snapshot(
        EMBEDDER_REPO_ID,
        revision=EMBEDDER_REVISION,
        allow_patterns=list(EMBEDDER_FILES),
        required_files=EMBEDDER_FILES,
        allow_download=allow_download,
    )


def attach_shared_weights(estimator: Any, shared: SharedSAPRPTWeights) -> None:
    """Point ``estimator`` at the objects of ``shared`` (the post-conditions of the library constructor).

    Sets the network, the device and dtype the estimator moves its tokenized batches to, the
    checkpoint path, and the tokenizer's sentence embedder. Used by the shared estimator factory,
    after unpickling and when the device changes; the objects themselves are never moved.
    """
    estimator.model = shared.module
    estimator.device = shared.device
    estimator.dtype = shared.dtype
    estimator._checkpoint_path = shared.checkpoint_path
    tokenizer = getattr(estimator, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.sentence_embedder = shared.sentence_embedder


def _embedder_module(estimator: Any) -> torch.nn.Module | None:
    """The torch module of the estimator's sentence embedder, or ``None`` when it is detached."""
    embedder = getattr(getattr(estimator, "tokenizer", None), "sentence_embedder", None)
    return getattr(embedder, "model", None)


# FIXME: model is for some reason super slow for 200 features and 50k samples (363616)
class SAPRPTOSSModel(SharedWeightsModelMixin, AbstractTorchModel):
    """ConTextTab Model: https://github.com/SAP-samples/sap-rpt-1-oss.

    The library builds its network and loads its sentence embedder inside the estimator
    constructor, so one :class:`SharedSAPRPTWeights` (network plus embedder) per checkpoint, device
    type and dtype is shared through the weights registry (see
    :mod:`tabarena.models._shared_weights_model`) and reaches the estimator through the constructor
    replica in :mod:`tabarena.models.sap_rpt_oss._estimators`. The fit device follows the library
    (CUDA whenever it is available), so with ``num_gpus=0`` on a CUDA host the warm-up primes a CPU
    entry that the fit does not use; TabArena allocates a GPU to every fit of this model.
    """

    ag_key = "SAP-RPT-OSS"
    #: ``import sap_rpt_oss`` pulls torch, transformers, sklearn and the Hub client; the last entry is
    #: TabArena's registry-backed estimator module, imported lazily by the fit.
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "sap_rpt_oss",
        "huggingface_hub",
        "tabarena.models.sap_rpt_oss._estimators",
    )
    #: Cheapness knob for the warm-up dummy fit: one bag means one forward per predict; the number of
    #: bags never touches the network or the checkpoint.
    warmup_dummy_fit_hyperparameters: ClassVar[dict] = {"bagging": 1}
    ag_name = "SAP-RPT-OSS"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 0.5
    _default_ag_args_ensemble_extra: ClassVar[dict] = {"refit_folds": True}

    shared_weights_spec: ClassVar[SharedWeightsSpec] = SharedWeightsSpec(
        library="sap_rpt_oss",
        checkpoint=CheckpointSpec(repo_id=REPO_ID, filename_param="checkpoint", revision=HF_REVISION),
        variant="network",  # the classifier and the regressor run the same module and text embedder
        default_params={"checkpoint": DEFAULT_CHECKPOINT},
        dtype=lambda hyperparameters, device_type: weights_dtype_name(device_type),
        disable_when=(_checkpoint_is_not_a_repo_file,),
        unshareable_examples=({"checkpoint": None},),
        network_attr="model",
        device_attrs=(("device", "torch"), ("tokenizer.sentence_embedder.device", "torch")),
    )

    # --- fit ------------------------------------------------------------------------------------

    # TODO: Figure out if num_cpus could be used somewhere
    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        available_num_gpus = ResourceManager.get_gpu_count_torch(cuda_only=True)
        if num_gpus > available_num_gpus:
            raise AssertionError(
                f"Fit specified to use {num_gpus} GPU, but only {available_num_gpus} "
                "CUDA GPUs are available. Please activate CUDA or switch to CPU usage.",
            )
        if self.problem_type not in self._supported_problem_types:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")

        hps = self._get_model_params()
        random_state = hps.pop(self.seed_name, 42)

        key, payload = self._acquire_shared_weights(device=self._fit_device())
        if key is None:
            from sap_rpt_oss import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Regressor

            model_cls = (
                SAP_RPT_OSS_Classifier if self.problem_type in ["binary", "multiclass"] else SAP_RPT_OSS_Regressor
            )
            self.model = model_cls(**hps)
        else:
            from tabarena.models.sap_rpt_oss._estimators import shared_estimator_cls

            self.model = shared_estimator_cls(self.problem_type).from_shared(payload, **hps)
        # TODO: make code support this like a normal sklearn model
        self.model.seed = random_state

        X = self.preprocess(X, y=y)  # does nothing, as no preprocessing is defined
        self.model = self.model.fit(
            X=X,
            y=y,
        )

    def _set_default_params(self):
        # Default values from the current version of the code base
        default_params = {
            "checkpoint": DEFAULT_CHECKPOINT,
            "max_context_size": 8192,
            "bagging": 8,
            "test_chunk_size": 4000,  # TODO, optimize based on dataset/VRAM?
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @staticmethod
    def _fit_device() -> str:
        """The device type a fit runs on: the library uses CUDA whenever it is available."""
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    # --- shared weights ---------------------------------------------------------------------------

    @classmethod
    def _resolve_shared_checkpoint(
        cls,
        *,
        problem_type: str,
        variant: str,
        hyperparameters: Mapping[str, Any],
        allow_download: bool,
        stage: str = "fit",
    ) -> ResolvedCheckpoint | None:
        """The checkpoint file of the repository, plus the embedder snapshot under the same download policy.

        The payload holds two checkpoints; the embedder snapshot is resolved here so a forbidden
        download surfaces at key derivation, where the mixin reports it, and the build finds it cached.
        """
        resolved = super()._resolve_shared_checkpoint(
            problem_type=problem_type,
            variant=variant,
            hyperparameters=hyperparameters,
            allow_download=allow_download,
            stage=stage,
        )
        if resolved is not None:
            resolve_embedder_dir(allow_download=allow_download)
            resolved = ResolvedCheckpoint(
                path=resolved.path,
                source={**resolved.source, "embedder": {"repo_id": EMBEDDER_REPO_ID, "revision": EMBEDDER_REVISION}},
            )
        return resolved

    @classmethod
    def _build_shared_weights(cls, key) -> SharedSAPRPTWeights:
        from tabarena.models.sap_rpt_oss._estimators import load_shared_weights

        return load_shared_weights(key)

    def _attach_shared_weights(self, payload: SharedSAPRPTWeights, device: str) -> None:
        device_type = normalize_device(device)
        check_payload_device(payload, device_type)
        attach_shared_weights(self.model, payload)
        self._apply_device_bookkeeping(device_type)

    def _detach_for_pickle(self, estimator: Any) -> Any:
        """A shallow copy of the estimator (and its tokenizer) without the network and the sentence embedder."""
        return detach_by_path(detach_by_path(estimator, "model"), "tokenizer.sentence_embedder")

    def _move_owned_network(self, device: str) -> None:
        """An estimator that owns its objects moves its network and its sentence embedder in place."""
        torch_device = self.to_torch_device(normalize_device(device))
        self.model.model.to(torch_device)
        embedder_module = _embedder_module(self.model)
        if embedder_module is not None:
            embedder_module.to(torch_device)

    def _shared_modules(self) -> Iterable[torch.nn.Module]:
        """The network and, when attached, the sentence embedder's module."""
        modules = list(super()._shared_modules())
        embedder_module = _embedder_module(self.model)
        if embedder_module is not None:
            modules.append(embedder_module)
        return modules

    @classmethod
    def prefetch_weights(cls) -> list[str]:
        """Make the checkpoint and the sentence embedder present locally; returns the resolved paths.

        Local-first: a cached file is kept and only a missing one is downloaded, both at their
        pinned commits.
        """
        return [*super().prefetch_weights(), resolve_embedder_dir()]

    # --- AutoGluon plumbing -------------------------------------------------------------------------

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    # TODO: Configure the AutoGluon preprocessing to pass the raw data format to the model
    #  (without preprocessing dates or texts) and not remove it from the features.
    # def _get_default_auxiliary_params(self) -> dict:
    #     default_auxiliary_params = super()._get_default_auxiliary_params()
    #     extra_auxiliary_params = dict(
    #         get_features_kwargs=dict(
    #             valid_special_types=[S_TEXT],
    #         )
    #     )
    #     default_auxiliary_params.update(extra_auxiliary_params)
    #     return default_auxiliary_params
