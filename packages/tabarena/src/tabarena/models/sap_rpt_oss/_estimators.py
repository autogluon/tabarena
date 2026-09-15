"""SAP-RPT-OSS estimators that take their network and sentence embedder from the shared-weights registry.

``sap_rpt_oss.rpt.SAP_RPT_OSS_Estimator.__init__`` (commit a323a0af) downloads the checkpoint through
``hf_hub_download`` (an etag request per estimator), builds the ``RPT`` network, casts and loads it,
and constructs a ``Tokenizer`` whose ``SentenceEmbedder`` loads the MiniLM model and tokenizer
through ``transformers``. It offers neither constructor arguments nor hooks for existing objects, so
:meth:`_SharedEstimatorMixin.from_shared` reproduces that constructor body with the three loads
replaced by the payload of :mod:`tabarena.models._weights` (:func:`load_shared_weights` performs the
same library steps once per key). Attribute names, order and values are the library's, so a shared
estimator carries exactly the attributes a library estimator carries; :func:`check_library_signatures`
is the drift guard for a ``sap_rpt_oss`` bump, and the ``models``-marked drift-guard test in
``tests/tabarena/models/test_shared_weights_models.py`` compares the constructor defaults and the
attribute sets when the library is installed. Re-diff the three replicated bodies against the
library whenever that guard fails.

This module imports ``sap_rpt_oss`` (and with it torch and transformers) at import time, so
``model.py`` imports it lazily inside the methods that need it.
"""

from __future__ import annotations

import inspect
import os
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from sap_rpt_oss.constants import ModelSize, embedding_model_to_dimension_and_pooling
from sap_rpt_oss.data.sentence_embedder import SentenceEmbedder
from sap_rpt_oss.data.tokenizer import Tokenizer
from sap_rpt_oss.model.torch_model import RPT
from sap_rpt_oss.rpt import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Estimator, SAP_RPT_OSS_Regressor
from sap_rpt_oss.utils.lru_cache import LRU_Cache
from transformers import AutoModel, AutoTokenizer

from tabarena.models.sap_rpt_oss.model import (
    SharedSAPRPTWeights,
    attach_shared_weights,
    resolve_embedder_dir,
    weights_dtype_name,
)

if TYPE_CHECKING:
    from tabarena.models._weights import WeightsKey

__all__ = [
    "LocalSentenceEmbedder",
    "SharedSAPRPTClassifier",
    "SharedSAPRPTRegressor",
    "SharedTokenizer",
    "build_module",
    "check_library_signatures",
    "load_shared_weights",
    "shared_estimator_cls",
    "weights_dtype",
]

#: Parameter names of the three library constructors this module reproduces, at the pinned commit.
ESTIMATOR_INIT_PARAMS = (
    "self",
    "checkpoint",
    "bagging",
    "max_context_size",
    "drop_constant_columns",
    "test_chunk_size",
)
TOKENIZER_INIT_PARAMS = (
    "self",
    "regression_type",
    "classification_type",
    "num_regression_bins",
    "random_seed",
    "is_valid",
)
EMBEDDER_INIT_PARAMS = ("self", "sentence_embedding_model_name", "batch_size", "device")


@cache
def check_library_signatures() -> None:
    """Raise when a library constructor no longer has the signature the replicated bodies assume.

    Raises:
        RuntimeError: A signature drifted; the replicated body must be re-diffed against the library.
    """
    expected = {
        SAP_RPT_OSS_Estimator: ESTIMATOR_INIT_PARAMS,
        Tokenizer: TOKENIZER_INIT_PARAMS,
        SentenceEmbedder: EMBEDDER_INIT_PARAMS,
    }
    for cls, params in expected.items():
        actual = tuple(inspect.signature(cls.__init__).parameters)
        if actual != params:
            raise RuntimeError(
                f"{cls.__module__}.{cls.__name__}.__init__ has parameters {actual}, expected {params}; the "
                "shared-weights estimators in tabarena.models.sap_rpt_oss._estimators reproduce that constructor "
                "and must be re-diffed against the installed sap_rpt_oss."
            )


def weights_dtype(device: torch.device) -> torch.dtype:
    """The dtype the library casts the network to on ``device`` (see ``model.weights_dtype_name``)."""
    return getattr(torch, weights_dtype_name(device.type) or "float32")


def build_module(checkpoint_path: str | Path, device: torch.device) -> RPT:
    """Build one ``RPT`` network from a checkpoint exactly as ``SAP_RPT_OSS_Estimator.__init__`` does.

    The same steps in the same order: construct the base-size module with the library's fixed
    ``regression_type="l2"`` and ``classification_type="cross-entropy"``, cast it to the CUDA dtype
    (bfloat16 or float16 by compute capability; the CPU keeps float32), ``load_weights`` with
    ``map_location=device`` (which copies the checkpoint's last encoder layer to every layer and
    strips a ``module.`` prefix), then move to ``device`` and ``eval()``. Gradients are switched off
    because every holder only runs inference. The registry runs this under its random-state guard,
    so the discarded random initialization never advances a process generator.
    """
    module = RPT(ModelSize.base, regression_type="l2", classification_type="cross-entropy")
    if device.type == "cuda":
        module = module.to(dtype=weights_dtype(device))
    module.load_weights(Path(checkpoint_path), device)
    module.to(device).eval()
    module.requires_grad_(False)
    return module


class LocalSentenceEmbedder(SentenceEmbedder):
    """``SentenceEmbedder`` whose model and tokenizer load from a local snapshot directory.

    The body is ``sap_rpt_oss.data.sentence_embedder.SentenceEmbedder.__init__`` with
    ``from_pretrained(sentence_embedding_model_name)`` replaced by ``from_pretrained(local_dir)``;
    the dimension and pooling lookup still uses the model name, the device rule and the CUDA
    ``half()`` cast are unchanged; gradients are switched off afterwards because the embedder only
    ever runs under ``no_grad`` and is shared.
    """

    def __init__(self, sentence_embedding_model_name: str, local_dir: str, batch_size: int = 512, device=None):
        check_library_signatures()
        self.sentence_embedding_model_name = sentence_embedding_model_name
        self.model = AutoModel.from_pretrained(local_dir)
        self.embedding_dimension, self.pooling_method = embedding_model_to_dimension_and_pooling[
            sentence_embedding_model_name
        ]
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = self.model.to(self.device).eval()
        if torch.cuda.is_available():
            self.model = self.model.half()
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.model.requires_grad_(False)


class SharedTokenizer(Tokenizer):
    """``Tokenizer`` that receives its sentence embedder instead of loading one.

    The body is ``sap_rpt_oss.data.tokenizer.Tokenizer.__init__`` with the ``SentenceEmbedder``
    construction replaced by the ``sentence_embedder`` argument; the text cache is created the same
    way (``LRU_CACHE_SIZE`` environment variable, default 10 000).
    """

    def __init__(
        self,
        sentence_embedder: SentenceEmbedder,
        regression_type: str = "reg-as-classif",
        classification_type: str = "cross-entropy",
        num_regression_bins: int = 16,
        random_seed: int | None = None,
        is_valid: bool = False,
    ):
        check_library_signatures()
        self.regression_type = regression_type
        self.classification_type = classification_type
        self.random_seed = random_seed
        self.num_regression_bins = num_regression_bins
        self.is_valid = is_valid

        self.sentence_embedder = sentence_embedder
        self.cache = LRU_Cache(max_size=int(os.getenv("LRU_CACHE_SIZE", "10000")))  # the library default, 10_000


def load_shared_weights(key: WeightsKey, *, allow_download: bool = True) -> SharedSAPRPTWeights:
    """Registry loader: the network and sentence embedder for ``key`` (``key.checkpoint`` is the resolved file).

    The embedder snapshot is resolved local-first here (the wrapper's key derivation already
    resolved it under the fit's download policy, so this is a cache hit; ``allow_download`` decides
    whether a miss may reach the Hub), the network by :func:`build_module` on ``key.device``. The embedder is
    constructed with the library's device string (the type only, as the library passes it), so its
    ``device`` attribute equals what a library tokenizer stores.

    Raises:
        RuntimeError: ``key.dtype`` differs from the dtype the library would use on ``key.device``.
    """
    device = torch.device(key.device)
    dtype = weights_dtype(device)
    if str(dtype).removeprefix("torch.") != key.dtype:
        raise RuntimeError(f"Key {key.short()} names dtype {key.dtype} but the library casts to {dtype} on {device}.")
    embedder_dir = resolve_embedder_dir(allow_download=allow_download)
    module = build_module(key.checkpoint, device)
    sentence_embedder = LocalSentenceEmbedder(Tokenizer.sentence_embedding_model_name, embedder_dir, device=key.device)
    return SharedSAPRPTWeights(
        module=module,
        sentence_embedder=sentence_embedder,
        device=device,
        dtype=dtype,
        checkpoint_path=str(key.checkpoint),
        embedder_dir=embedder_dir,
    )


class _SharedEstimatorMixin:
    """Factory reproducing ``SAP_RPT_OSS_Estimator.__init__`` with the loads replaced by a registry payload.

    :meth:`from_shared` builds the instance without running the library constructor (which would
    download and build again) and sets the same attributes in the same order; ``model``, ``device``,
    ``dtype`` and ``_checkpoint_path`` come from the payload and the tokenizer is a
    :class:`SharedTokenizer` around the payload's embedder. The inherited ``__init__`` keeps the
    library signature, so sklearn's ``get_params`` and ``repr`` work on the result; calling it
    directly builds a self-owned estimator exactly like the library class.
    """

    @classmethod
    def from_shared(
        cls,
        shared: SharedSAPRPTWeights,
        checkpoint: str = "2025-11-04_sap-rpt-one-oss.pt",
        bagging: str | int = 8,
        max_context_size: int = 8192,
        drop_constant_columns: bool = True,
        test_chunk_size: int = 1000,
    ):
        """An unfitted estimator whose network and embedder are the objects of ``shared``.

        Args:
            shared: The registry payload for this checkpoint on the fit device.
            checkpoint: The checkpoint file name, stored as the library stores it (``shared`` was
                built from it).
            bagging: Number of bags, or ``"auto"``.
            max_context_size: Maximum number of context rows per bag.
            drop_constant_columns: Whether constant columns are dropped before tokenization.
            test_chunk_size: Number of query rows per forward.

        Raises:
            ValueError: ``bagging`` is neither an integer nor ``"auto"`` (the library's check).
        """
        check_library_signatures()
        self = cls.__new__(cls)
        self.model_size = ModelSize.base
        self.checkpoint = checkpoint
        self.regression_type = "l2"
        self.classification_type = "cross-entropy"
        self.test_chunk_size = test_chunk_size
        self.bagging = bagging
        if not isinstance(bagging, int) and bagging != "auto":
            raise ValueError('bagging must be an integer or "auto"')
        self.max_context_size = max_context_size
        self.num_regression_bins = 16
        attach_shared_weights(self, shared)  # model, device, dtype, _checkpoint_path
        self.seed = 42
        self.drop_constant_columns = drop_constant_columns
        self.tokenizer = SharedTokenizer(
            shared.sentence_embedder,
            regression_type=self.regression_type,
            classification_type=self.classification_type,
            random_seed=self.seed,
            num_regression_bins=self.num_regression_bins,
            is_valid=True,
        )
        return self


class SharedSAPRPTClassifier(_SharedEstimatorMixin, SAP_RPT_OSS_Classifier):
    """``SAP_RPT_OSS_Classifier`` whose network and embedder come from the shared-weights registry."""


class SharedSAPRPTRegressor(_SharedEstimatorMixin, SAP_RPT_OSS_Regressor):
    """``SAP_RPT_OSS_Regressor`` whose network and embedder come from the shared-weights registry."""


def shared_estimator_cls(problem_type: str) -> type[SharedSAPRPTClassifier | SharedSAPRPTRegressor]:
    """The registry-backed estimator class for ``problem_type`` (binary and multiclass share the classifier)."""
    return SharedSAPRPTClassifier if problem_type in ("binary", "multiclass") else SharedSAPRPTRegressor
