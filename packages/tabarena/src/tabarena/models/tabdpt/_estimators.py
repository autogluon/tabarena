"""TabDPT's network load as a separable call, and a constructor that takes the network, so one network per process serves every fit.

Developer fix. ``tabdpt.estimator.TabDPTEstimator.__init__`` (tabdpt 1.2.0 and 1.3.0) reads the safetensors
checkpoint and builds the network inside the constructor and offers neither a constructor argument nor
a reload hook for an existing network. :func:`load_network` is the loading half of that constructor and
the wrapper's ``shared_weights`` loader; :class:`_SharedNetworkEstimator` reproduces the rest of the
constructor body around a network it is handed, so a shared estimator carries exactly the attributes a
library estimator carries, with the library's signature, so sklearn's ``get_params`` keeps working
through the task subclasses. :func:`make_estimator` is what the wrapper constructs. The library could
offer both itself, as a ``network=`` constructor argument or a ``_load_model`` method its ``fit`` calls.
That is proposed upstream in https://github.com/layer6ai-labs/TabDPT-inference/pull/79 (a separable
``TabDPTEstimator._load_model()`` the constructor calls once); once a release ships it, the wrapper
declares ``loader="tabdpt.estimator:TabDPTEstimator._load_model"`` like TabICL and this module goes.
Until then :func:`check_replica_signature` guards a ``tabdpt`` bump (run by the shared-weights
convention test) and the constructor body must be re-diffed against ``TabDPTEstimator.__init__``
whenever it fails.

Imports ``tabdpt`` (and with it torch, faiss, omegaconf and safetensors) at module level; the wrapper
imports it inside ``_fit``.
"""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any, Literal

import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from tabdpt.classifier import TabDPTClassifier
from tabdpt.estimator import TabDPTEstimator
from tabdpt.model import TabDPTModel
from tabdpt.regressor import TabDPTRegressor
from tabdpt.utils import Log1pScaler

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "SharedTabDPTClassifier",
    "SharedTabDPTRegressor",
    "check_replica_signature",
    "load_network",
    "make_estimator",
]


def load_network(model_weight_path: str | Path, device: str, *, use_flash: bool, clip_sigma: float) -> torch.nn.Module:
    """Build the network from ``model_weight_path`` exactly as ``TabDPTEstimator.__init__`` does.

    The library's steps are kept verbatim: ``safe_open`` reads the tensors onto ``device`` and the
    architecture config from the file metadata, ``TabDPTModel.load`` constructs the module (on the
    CPU, with a random initialization that ``load_state_dict`` then overwrites), moves it to the
    device and sets eval mode. ``use_flash`` and ``clip_sigma`` are constructor arguments the module
    stores and reads in its forward pass, so they key the network; ``use_flash`` is honored on CUDA
    only, as the constructor does. The registry runs this loader under its random-state guard, so the
    discarded random initialization never advances a process generator.
    """
    use_flash = bool(use_flash) and str(device).startswith("cuda")
    clip_sigma = float(clip_sigma)
    with safe_open(str(model_weight_path), framework="pt", device=device) as f:
        cfg = OmegaConf.create(json.loads(f.metadata()["cfg"]))
        model_state = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118  # safe_open handles are not iterable
    cfg.env.device = device
    network = TabDPTModel.load(model_state=model_state, config=cfg, use_flash=use_flash, clip_sigma=clip_sigma)
    network.eval()
    return network


class _SharedNetworkEstimator(TabDPTEstimator):
    """``TabDPTEstimator`` whose constructor takes an already built network instead of reading the checkpoint.

    The body is tabdpt 1.3.0 ``estimator.py`` lines 89 to 152 with the checkpoint read and
    ``TabDPTModel.load`` replaced by the network :meth:`from_shared` stores on the instance before
    running the constructor. 1.2.0 (the superseded TabDPT-Turbo pin) differs only in the name of
    the PCA projection attribute, so both names are initialised. The signature (names, order,
    defaults, annotations) is identical to the library constructor so sklearn's ``get_params``
    keeps working through the task subclasses.

    A shared module must never be mutated, so the constructor refuses a configuration whose
    effective ``compile`` flag is True: ``TabDPTEstimator.fit`` would call ``self.model.compile()``
    in place on the module every other child holds. The wrapper constructs the library's own class
    for such a configuration.
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

    def __init__(  # the library's positional signature, checked by check_replica_signature
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

        # tabdpt 1.3.0 names the PCA projection `projection`, 1.2.0 names it `V`; `fit` and `to` read
        # whichever the installed release uses.
        self.projection = None
        self.V = None


class SharedTabDPTClassifier(TabDPTClassifier, _SharedNetworkEstimator):
    """``TabDPTClassifier`` whose network comes from the shared-weights registry.

    The MRO places :class:`_SharedNetworkEstimator` between the task class and
    ``TabDPTEstimator``, so ``TabDPTClassifier.__init__``'s ``super().__init__(mode="cls", ...)``
    lands in the replacement constructor.
    """


class SharedTabDPTRegressor(TabDPTRegressor, _SharedNetworkEstimator):
    """``TabDPTRegressor`` whose network comes from the shared-weights registry."""


_SHARED_BY_MODE: dict[str, type] = {"cls": SharedTabDPTClassifier, "reg": SharedTabDPTRegressor}


def make_estimator(network: torch.nn.Module, *, mode: Literal["cls", "reg"], **kwargs: Any) -> Any:
    """An unfitted task estimator around ``network``: the library's classifier or regressor with the replica constructor.

    ``mode`` is the library's own name for the task and ``kwargs`` are ``TabDPTEstimator``'s
    constructor arguments (the same the library's class takes).
    """
    return _SHARED_BY_MODE[mode].from_shared(network, **kwargs)


def check_replica_signature() -> None:
    """Raise ``TypeError`` when the installed ``TabDPTEstimator.__init__`` no longer matches the replica."""
    library = inspect.signature(TabDPTEstimator.__init__)
    replica = inspect.signature(_SharedNetworkEstimator.__init__)
    if list(library.parameters) != list(replica.parameters) or any(
        library.parameters[name].default != replica.parameters[name].default for name in library.parameters
    ):
        raise TypeError(
            f"tabdpt's TabDPTEstimator.__init__ signature {library} no longer matches the replica {replica}; "
            "re-diff _SharedNetworkEstimator.__init__ against the installed library"
        )
