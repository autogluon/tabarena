"""TabDPT estimator subclasses whose constructor takes the network from the shared-weights registry.

``tabdpt.estimator.TabDPTEstimator.__init__`` (tabdpt 1.2.0) reads the safetensors checkpoint and
builds the network inside the constructor and offers neither a constructor argument nor a reload
hook for an existing network. The constructor body is therefore reproduced here with that block
replaced by the network :meth:`_SharedNetworkEstimator.from_shared` was handed; everything else
(attribute names, order, defaults, assertions, the normalizer match) is the library's code, so a
shared estimator carries exactly the attributes a library estimator carries. The signature is the
library's too, so sklearn's ``get_params`` keeps working through the task subclasses.
:func:`check_replica_signature` is the drift guard for a ``tabdpt`` bump (run by the
``models``-marked test in ``tests/tabarena/models/test_shared_weights_models.py``); re-diff
:meth:`_SharedNetworkEstimator.__init__` against ``TabDPTEstimator.__init__`` whenever it fails.

This module imports ``tabdpt`` (and with it torch, faiss, omegaconf and safetensors) at import
time, so ``model.py`` imports it lazily inside the methods that need it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from tabdpt.classifier import TabDPTClassifier
from tabdpt.estimator import TabDPTEstimator
from tabdpt.model import TabDPTModel
from tabdpt.regressor import TabDPTRegressor
from tabdpt.utils import Log1pScaler

from tabarena.models._shared_estimators import check_signature_matches

if TYPE_CHECKING:
    from tabarena.models._weights import WeightsKey

__all__ = [
    "SharedTabDPTClassifier",
    "SharedTabDPTRegressor",
    "check_replica_signature",
    "key_clip_sigma",
    "key_use_flash",
    "load_network",
    "shared_estimator_cls",
]


def key_use_flash(key: WeightsKey) -> bool:
    """The ``use_flash`` flag stored in ``key``."""
    return dict(key.flags)["use_flash"] == "True"


def key_clip_sigma(key: WeightsKey) -> float:
    """The ``clip_sigma`` value stored in ``key``."""
    return float(dict(key.flags)["clip_sigma"])


def load_network(key: WeightsKey) -> torch.nn.Module:
    """Build the network for ``key`` exactly as ``TabDPTEstimator.__init__`` does.

    The three library steps are kept verbatim: ``safe_open`` reads the tensors onto ``key.device``
    and the architecture config from the file metadata, ``TabDPTModel.load`` constructs the module
    (on the CPU, with a random initialization that ``load_state_dict`` then overwrites), moves it to
    the device and sets eval mode. ``use_flash`` and ``clip_sigma`` come from the key's flags because
    the module stores both and reads them in its forward pass. The registry runs this loader under
    its random-state guard, so the discarded random initialization never advances a process
    generator; the mixin puts the result in eval mode and freezes it.
    """
    with safe_open(key.checkpoint, framework="pt", device=key.device) as f:
        cfg = OmegaConf.create(json.loads(f.metadata()["cfg"]))
        model_state = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118  # safe_open handles are not iterable
    cfg.env.device = key.device
    network = TabDPTModel.load(
        model_state=model_state, config=cfg, use_flash=key_use_flash(key), clip_sigma=key_clip_sigma(key)
    )
    network.eval()
    return network


class _SharedNetworkEstimator(TabDPTEstimator):
    """``TabDPTEstimator`` whose constructor takes an already built network instead of reading the checkpoint.

    The body is tabdpt 1.2.0 ``estimator.py`` lines 89 to 152 with the checkpoint read and
    ``TabDPTModel.load`` replaced by the network :meth:`from_shared` stores on the instance before
    running the constructor. The signature (names, order, defaults, annotations) is identical to
    the library constructor so sklearn's ``get_params`` keeps working through the task subclasses.

    A shared module must never be mutated, so the constructor refuses a configuration whose
    effective ``compile`` flag is True: ``TabDPTEstimator.fit`` would call ``self.model.compile()``
    in place on the module every other child holds.
    """

    @classmethod
    def from_shared(cls, network: torch.nn.Module, **kwargs):
        """An unfitted estimator around ``network`` (the registry payload), constructed with the library's keywords.

        Raises:
            ValueError: ``network`` was built with another ``use_flash`` or ``clip_sigma`` than the
                constructor resolves from ``kwargs``, or the configuration compiles in place.
        """
        self = cls.__new__(cls)
        self._shared_network = network
        self.__init__(**kwargs)
        return self

    def __init__(
        self,
        mode: Literal["cls", "reg"],
        normalizer: Literal["standard", "minmax", "robust", "power", "quantile-uniform", "quantile-normal", "log1p"]
        | None = "standard",
        missing_indicators: bool = False,
        clip_sigma: float = 8.0,
        feature_reduction: Literal["pca", "subsample"] = "pca",
        context_reduction: Literal["retrieval", "subsample", "subsample-balanced"] = "subsample",
        faiss_metric: Literal["l2", "ip"] = "l2",
        device: str = None,  # noqa: RUF013  # the library annotates `str` with a None default
        use_flash: bool = True,
        compile: bool = True,  # the library's parameter name
        model_weight_path: str | None = None,
        verbose: bool = True,
    ):
        network = self.__dict__.pop("_shared_network", None)
        if network is None:
            raise TypeError(f"construct {type(self).__name__} through from_shared(network, ...)")
        self.mode = mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_flash = use_flash and self.device == "cuda"
        self.missing_indicators = missing_indicators

        # Library: resolve `path` (downloading when None), read the checkpoint, build the network.
        # Here: the path is recorded for reference and the network is the registry payload.
        if not model_weight_path:
            raise ValueError(f"{type(self).__name__} needs a resolved checkpoint path as `model_weight_path`.")
        self.path = model_weight_path
        if bool(network.use_flash) != self.use_flash or float(network.clip_sigma) != float(clip_sigma):
            raise ValueError(
                f"{type(self).__name__}: the shared network was built with use_flash={network.use_flash!r}, "
                f"clip_sigma={network.clip_sigma!r} but the constructor resolves use_flash={self.use_flash!r}, "
                f"clip_sigma={clip_sigma!r}; the registry key and the estimator configuration must agree."
            )
        self.model = network

        self.max_features = self.model.num_features
        self.max_num_classes = self.model.n_out
        self.compile = compile and self.device == "cuda"
        if self.compile:
            raise ValueError(
                f"{type(self).__name__} shares its network with other estimators and cannot compile it in place; "
                "construct the library estimator instead when `compile` is True on CUDA."
            )
        self.feature_reduction = feature_reduction
        self.context_reduction = context_reduction
        self.faiss_metric = faiss_metric
        self.faiss_knn = None

        assert self.mode in ["cls", "reg"], "mode must be 'cls' or 'reg'"
        assert self.feature_reduction in ["pca", "subsample"], "feature_reduction must be 'pca' or 'subsample'"
        assert self.context_reduction in ["retrieval", "subsample", "subsample-balanced"], (
            "context_reduction must be 'retrieval', 'subsample', or 'subsample-balanced'"
        )
        if self.mode == "reg" and self.context_reduction == "subsample-balanced":
            raise ValueError("context_reduction='subsample-balanced' is only supported for classification")
        assert self.faiss_metric in ["l2", "ip"], 'faiss_metric must be "l2" or "ip"'

        self.verbose = verbose

        self.normalizer = normalizer
        match normalizer:
            case "standard":
                self.scaler = StandardScaler()
            case "minmax":
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
            case "robust":
                self.scaler = RobustScaler()
            case "power":
                self.scaler = PowerTransformer()
            case "quantile-uniform":
                self.scaler = QuantileTransformer(output_distribution="uniform")
            case "quantile-normal":
                self.scaler = QuantileTransformer(output_distribution="normal")
            case "log1p":
                self.scaler = Log1pScaler()
            case None:
                self.scaler = None
            case _:
                raise ValueError(
                    "normalizer must be one of "
                    '["standard", "minmax", "robust", "power", "quantile-uniform", "quantile-normal", "log1p", None]'
                )

        self.V = None


class SharedTabDPTClassifier(TabDPTClassifier, _SharedNetworkEstimator):
    """``TabDPTClassifier`` whose network comes from the shared-weights registry.

    The MRO places :class:`_SharedNetworkEstimator` between the task class and
    ``TabDPTEstimator``, so ``TabDPTClassifier.__init__``'s ``super().__init__(mode="cls", ...)``
    lands in the replacement constructor.
    """


class SharedTabDPTRegressor(TabDPTRegressor, _SharedNetworkEstimator):
    """``TabDPTRegressor`` whose network comes from the shared-weights registry."""


_SHARED_BY_LIBRARY_CLS: dict[type, type] = {
    TabDPTClassifier: SharedTabDPTClassifier,
    TabDPTRegressor: SharedTabDPTRegressor,
}


def shared_estimator_cls(library_cls: type) -> type[SharedTabDPTClassifier | SharedTabDPTRegressor]:
    """The registry-backed subclass of a library estimator class.

    Raises:
        KeyError: ``library_cls`` is neither ``TabDPTClassifier`` nor ``TabDPTRegressor``.
    """
    return _SHARED_BY_LIBRARY_CLS[library_cls]


def check_replica_signature() -> None:
    """Raise ``TypeError`` when the installed ``TabDPTEstimator.__init__`` no longer matches the replica."""
    check_signature_matches(TabDPTEstimator.__init__, _SharedNetworkEstimator.__init__)
