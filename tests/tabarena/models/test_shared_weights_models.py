"""Convention tests for every wrapper that shares pretrained weights through ``SharedWeightsModelMixin``.

One registry-driven parametrized file, like ``test_all_models.py``: every registered model class
that inherits the mixin and declares a ``shared_weights_spec`` is tested against the same generic
fakes (a tiny torch module as the payload, a faked checkpoint resolution, an estimator stand-in
built from the spec's attribute paths or from the real ``derive_shared_estimator`` factory). No
model library is imported; ``test_hooks_run_without_the_library`` blocks them outright. The two
``models``-marked tests at the end fit the real libraries and are skipped by the default ``pytest``.

Per-wrapper fakes live in the ``WRAPPER_FAKES`` blocks below, one block per wrapper, filled by the
agent that migrates that wrapper. A wrapper without a block gets the generic estimator for its seam.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import os
import pickle
import random
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from autogluon.common.utils.pretrained_weights import FETCH_ENV_VAR, PretrainedWeightsUnavailableError
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

import tabarena.models._shared_estimators as se
import tabarena.models._shared_weights_model as swm
import tabarena.models._weights as w
import tabarena.models.prefetch as hub
import tabarena.models.warmup as wu
from tabarena.models import get_model_registry
from tabarena.models._shared_weights_model import (
    PROBLEM_TYPES,
    ResolvedCheckpoint,
    SharedWeightsModelMixin,
    SharedWeightsSpec,
)

from .smoke_configs import smoke_for

torch = pytest.importorskip("torch")

# --- discovery ------------------------------------------------------------------------------------

_REGISTRY = get_model_registry()

#: Model classes that share through the mixin; a wrapper that is not migrated is simply absent.
MIXIN_USERS: list[type] = sorted(
    {
        info.model_cls
        for info in _REGISTRY.values()
        if isinstance(info.model_cls, type)
        and issubclass(info.model_cls, SharedWeightsModelMixin)
        and info.model_cls.shared_weights_spec is not None
    },
    key=lambda cls: cls.__qualname__,
)

#: Every class that migrated to the mixin. The agent migrating a wrapper adds its class names here,
#: so a wrapper that later drops the mixin (or its spec) fails ``test_replaced_wrappers_use_the_mixin``.
EXPECTED_MIXIN_USERS: frozenset[str] = frozenset(
    {
        "TabICLModel",
        "TabICLv2Model",
        "TabPFN3Model",
        "RealTabPFNv25Model",
        "TabPFNv26Model",
        "TabPFNWideModel",
        "CausiloModel",
        "OrionMSPModel",
        "TabLDMModel",
        "TabSwiftModel",
        "TabDPTTurboModel",
        "TabFMModel",
        "EXAONETabularModel",
        "NoriModel",
        "Nori30MModel",
        "LimiXModel",
        "SAPRPTOSSModel",
        "MitraV2Model",
        "TabSTARModel",
    }
)

assert MIXIN_USERS, "no registered model class inherits SharedWeightsModelMixin with a shared_weights_spec"

METHOD_BY_CLASS: dict[type, str] = {}
for _method, _info in sorted(_REGISTRY.items()):
    METHOD_BY_CLASS.setdefault(_info.model_cls, _method)


def _problem_types(cls: type) -> list[str]:
    """The first supported classification type and, when supported, regression."""
    supported = cls.supported_problem_types() or list(PROBLEM_TYPES)
    out = (
        [next(pt for pt in ("binary", "multiclass") if pt in supported)]
        if any(pt in supported for pt in ("binary", "multiclass"))
        else []
    )
    if "regression" in supported:
        out.append("regression")
    return out


CASES: list[tuple[type, str]] = [(cls, pt) for cls in MIXIN_USERS for pt in _problem_types(cls)]
CASE_IDS: list[str] = [f"{cls.__name__}-{pt}" for cls, pt in CASES]
MODULE_CASES = [case for case in CASES if case[0].shared_weights_spec.mode == "module"]
MODULE_IDS = [f"{cls.__name__}-{pt}" for cls, pt in MODULE_CASES]
STATE_DICT_CASES = [case for case in CASES if case[0].shared_weights_spec.mode == "state_dict"]
STATE_DICT_IDS = [f"{cls.__name__}-{pt}" for cls, pt in STATE_DICT_CASES]
CLASS_IDS = [cls.__name__ for cls in MIXIN_USERS]

#: Top-level import names blocked (``sys.modules[name] = None``) while the library-free hooks run.
BLOCKED_LIBRARIES: dict[str, tuple[str, ...]] = {
    "tabpfn": ("tabpfn",),
    "tabpfnwide": ("tabpfnwide", "tabpfn"),
    "tabicl": ("tabicl",),
    "tabdpt": ("tabdpt",),
    "causilo": ("causilo",),
    "tabldm": ("tabldm",),
    "mitra_v2": (),
    "tabfm": ("tabfm",),
    "tabswift": (),
    "exaone_tabular": ("exaonetabular",),
    "nori": ("synthefy_nori",),
    "limix": ("limix",),
    "orionmsp": ("tabtune",),
    "sap_rpt_oss": ("sap_rpt_oss",),
    "tabstar": ("tabstar",),
}


# --- generic fakes ----------------------------------------------------------------------------------


class _TinyNet(torch.nn.Module):
    """A deterministic Linear(3, 2) that counts forwards and records (and can refuse) ``.to()`` calls."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        with torch.no_grad():
            self.linear.weight.copy_(torch.arange(6.0).reshape(2, 3) / 10)
            self.linear.bias.zero_()
        self.forward_calls = 0
        self.to_calls: list[tuple] = []
        self.shared_guard = False
        self._fake_device: str | None = None

    def forward(self, x):
        self.forward_calls += 1
        return self.linear(x)

    def to(self, *args, **kwargs):
        self.to_calls.append(args)
        if self.shared_guard:
            raise AssertionError("a shared payload was moved with .to()")
        return super().to(*args, **kwargs)


class _DeviceTaggedTensor(torch.Tensor):
    """A tensor that reports ``device.type == "cuda"`` while ``_fake_cuda`` is set (no CUDA needed).

    ``detach().to("cpu")`` returns a fresh instance without the flag, so a moved tensor reads as CPU
    and a tensor left in place keeps the tag through pickling.
    """

    @property
    def device(self):
        if getattr(self, "_fake_cuda", False):
            return torch.device("cuda")
        return super().device


def _tagged_cuda_tensor() -> _DeviceTaggedTensor:
    tensor = _DeviceTaggedTensor(torch.zeros(2))
    tensor._fake_cuda = True
    return tensor


def walk_tensors(obj: Any, *, max_depth: int = 8) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(path, tensor)`` for every tensor a pickle of ``obj`` would hold.

    Walks dicts, sequences, dataclasses and modules; any other object is walked through its
    ``__getstate__()`` result, which is what pickle serializes.
    """
    seen: set[int] = set()

    def walk(node: Any, path: str, depth: int) -> Iterator[tuple[str, torch.Tensor]]:
        if depth < 0 or node is None or id(node) in seen:
            return
        if isinstance(node, torch.Tensor):
            yield path, node
            return
        if isinstance(node, (str, bytes, int, float, bool, type)):
            return
        seen.add(id(node))
        if isinstance(node, torch.nn.Module):
            for name, tensor in (*node.named_parameters(), *node.named_buffers()):
                yield f"{path}.{name}", tensor
            return
        if isinstance(node, dict):
            for key, value in node.items():
                yield from walk(value, f"{path}[{key!r}]", depth - 1)
        elif isinstance(node, (list, tuple, set)):
            for index, value in enumerate(node):
                yield from walk(value, f"{path}[{index}]", depth - 1)
        elif dataclasses.is_dataclass(node):
            for field in dataclasses.fields(node):
                yield from walk(getattr(node, field.name, None), f"{path}.{field.name}", depth - 1)
        elif hasattr(node, "__getstate__"):
            # What pickle serializes: an estimator that drops its network in __getstate__ shows none here.
            state = node.__getstate__()
            if isinstance(state, dict):
                for name, value in state.items():
                    yield from walk(value, f"{path}.{name}", depth - 1)
            elif state is not None:
                yield from walk(state, f"{path}.__getstate__()", depth - 1)

    yield from walk(obj, "state", max_depth)


def _set_path_creating(obj: Any, path: str, value: Any) -> None:
    """``set_by_path`` that creates missing intermediate objects as ``SimpleNamespace``."""
    parts = path.split(".")
    parent = obj
    for part in parts[:-1]:
        child = getattr(parent, part, None)
        if child is None:
            child = SimpleNamespace()
            setattr(parent, part, child)
        parent = child
    setattr(parent, parts[-1], value)


def _ensure_spec_paths(estimator: Any, spec: SharedWeightsSpec, device: str) -> None:
    """Create every ``device_attrs`` and ``owned_tensor_attrs`` path of ``spec`` on ``estimator``."""
    for path, kind in spec.device_attrs:
        _set_path_creating(estimator, path, torch.device(device) if kind == "torch" else device)
    for path in spec.owned_tensor_attrs:
        _set_path_creating(estimator, path, torch.zeros(2))
    for name, paths in spec.flag_attrs.items():
        for path in paths:
            _set_path_creating(estimator, path, None)


def _net_of(payload_or_network: Any) -> _TinyNet:
    """The tiny net inside a payload or an attached network (a module, a container or a list)."""
    modules = se.payload_modules(payload_or_network)
    assert modules, f"no module in {payload_or_network!r}"
    return modules[0]


class _PathEstimator:
    """Generic stand-in for ``seam="attr"``: the payload sits at ``network_attr``, every spec path exists."""

    def __init__(self, spec: SharedWeightsSpec, payload: Any, device: str) -> None:
        self._network_attr = spec.network_attr
        _ensure_spec_paths(self, spec, device)
        _set_path_creating(self, spec.network_attr, payload)
        self.n_features_in_ = 3

    def to(self, device):
        """The library estimator's own move (``estimator_move="to"``): moves the network it owns."""
        for module in se.payload_modules(se.get_by_path(self, self._network_attr)):
            module.to(device)
        return self

    def predict_proba(self, X):
        return _net_of(se.get_by_path(self, self._network_attr))(torch.as_tensor(np.asarray(X), dtype=torch.float32))


class _TinyLibraryEstimator:
    """Stand-in for an sklearn-style library estimator whose ``_load_model`` builds the network (keeps it in the pickle)."""

    def __init__(self, spec: SharedWeightsSpec | None = None, device: str = "cpu", kv_cache: bool = False) -> None:
        self.device = device
        self.kv_cache = kv_cache
        self.device_ = None
        self.model_ = None
        self.model_path_ = None
        self.calls: list[str] = []
        if spec is not None:
            _ensure_spec_paths(self, spec, device)

    def _resolve_device(self) -> None:
        self.device_ = torch.device(self.device)

    def _load_model(self) -> None:
        self.calls.append("library_load")
        self.model_ = _TinyNet()
        self.model_path_ = "library"

    def _build_inference_config(self) -> None:
        self.calls.append("_build_inference_config")
        self.inference_config_ = SimpleNamespace(
            COL_CONFIG=SimpleNamespace(device=self.device_),
            ROW_CONFIG=SimpleNamespace(device=self.device_),
            ICL_CONFIG=SimpleNamespace(device=self.device_),
        )

    def _move_cache_to_device(self) -> None:
        self.calls.append("_move_cache_to_device")

    def fit(self):
        self._resolve_device()
        self._build_inference_config()
        self._load_model()
        self.n_features_in_ = 3
        return self

    def predict_proba(self, X):
        return self.model_(torch.as_tensor(np.asarray(X), dtype=torch.float32))


class _TinyWeightlessLibraryEstimator(_TinyLibraryEstimator):
    """Like tabicl: pickles without the network and its device fields, re-runs ``_load_model`` on unpickle."""

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for name in ("model_", "device_", "inference_config_", "model_path_"):
            state.pop(name, None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._resolve_device()
        self._load_model()
        self._build_inference_config()


_DerivedTinyEstimator = se.derive_shared_estimator(_TinyLibraryEstimator, module=__name__, name="_DerivedTinyEstimator")
_DerivedWeightlessTinyEstimator = se.derive_shared_estimator(
    _TinyWeightlessLibraryEstimator, module=__name__, name="_DerivedWeightlessTinyEstimator"
)

#: ``(spec, payload, key, device, loader) -> estimator``; the estimator ends up fitted and attached.
EstimatorFactory = Callable[[SharedWeightsSpec, Any, w.WeightsKey, str, Callable[..., Any]], Any]


def _attr_seam_estimator(spec, payload, key, device, loader) -> _PathEstimator:
    return _PathEstimator(spec, payload, device)


def _load_model_seam_estimator(spec, payload, key, device, loader):
    derived = _DerivedWeightlessTinyEstimator if spec.detach_attr == "library" else _DerivedTinyEstimator
    estimator = derived(spec, device=device)
    if key is not None:
        estimator.use_shared_weights(key, payload, loader)
    return estimator.fit()


@dataclasses.dataclass
class WrapperFake:
    """Per-wrapper fakes for hooks the generic stand-ins cannot express.

    Args:
        payload: Wraps the tiny net into the payload shape the wrapper's seam expects (default: the net).
        estimator: Builds the fitted, attached estimator stand-in (default: chosen by ``spec.seam``).
        library_free_patches: ``(dotted target, replacement)`` monkeypatches applied in every generic
            test, so the wrapper's hooks run through them with and without the libraries blocked (an
            accessor in the wrapper's ``_estimators.py`` returning a fake).
        generic_hyperparameters: Hyperparameters the generic CPU tests overlay on the smoke
            configuration (a wrapper that shares on the CPU only when asked to); the
            ``models``-marked real fits do not use them.
    """

    payload: Callable[[_TinyNet, w.WeightsKey], Any] = lambda net, key: net
    estimator: EstimatorFactory | None = None
    library_free_patches: tuple[tuple[str, Any], ...] = ()
    generic_hyperparameters: dict = dataclasses.field(default_factory=dict)


WRAPPER_FAKES: dict[str, WrapperFake] = {}

#: Library drift guards (``models`` marker), one callable per wrapper that relies on a constructor
#: replica or on a library seam (``_load_model`` called by ``fit``, an engine that loads only when
#: empty); each guard skips when its library is not installed.
REPLICA_DRIFT_GUARDS: dict[str, Callable[[], None]] = {}


# --- per-wrapper fake table (one block per wrapper; filled by the agent migrating that wrapper) -------

# tabpfn_3 (TabPFN3Model), tabpfnv2_5 (RealTabPFNv25Model, TabPFNv26Model), tabpfnwide (TabPFNWideModel): the
# TabPFN preset attaches through ``models_``, the engine's ``_set_models`` and the estimator's ``to``, so the
# stand-in mirrors tabpfn's per-device model cache: a shared module is found under its exact device key and
# never moved, an owned module is moved by the library's own ``to``.


@dataclasses.dataclass(eq=False)
class _FakeModelSpecs:
    """Stand-in for a tabpfn ``ModelSpecs`` container, the payload the TabPFN preset hands to ``model_path``."""

    model: torch.nn.Module

    def __deepcopy__(self, memo):
        return self


class _FakeTabPFNEngine:
    """tabpfn's inference engine as far as the preset touches it: ``_set_models`` and a per-device ``to``."""

    def __init__(self, models: list, *, owned: bool) -> None:
        self.owned = owned
        self._set_models(models)

    def _set_models(self, models: list) -> None:
        self.model_caches = [{se.payload_device(module): module} for module in models]

    def to(self, devices, *args) -> None:
        for cache in self.model_caches:
            for device in devices:
                device_type = w.normalize_device(device)
                if self.owned:
                    cache[device_type] = next(iter(cache.values())).to(device_type)
                elif device_type not in cache:
                    raise AssertionError(f"the shared module is not resident on {device_type}")


class _FakeTabPFN:
    """Stand-in for a fitted tabpfn estimator: ``models_``, ``executor_``, ``to`` and the device fields the preset reads."""

    def __init__(self, payload: Any, device: str, *, owned: bool) -> None:
        self.model_path = "ckpt.bin"
        self.models_ = list(se.payload_modules(payload))
        self.executor_ = _FakeTabPFNEngine(self.models_, owned=owned)
        self.to(device)
        self.n_features_in_ = 3

    def to(self, device) -> None:
        self.device = device
        self.devices_ = (torch.device(w.normalize_device(device)),)
        self.executor_.to(self.devices_)

    def predict_proba(self, X):
        return self.models_[0](torch.as_tensor(np.asarray(X), dtype=torch.float32))


class _FakeManyClass:
    """Stand-in for ``ManyClassClassifier``: an unfitted base estimator holding the specs, plus optional fitted rows."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator
        self.alphabet_size = 10
        self.estimators_ = None


def _tabpfn_estimator(spec, payload, key, device, loader) -> _FakeTabPFN:
    return _FakeTabPFN(payload, device, owned=key is None)


_TABPFN_FAKE = WrapperFake(payload=lambda net, key: _FakeModelSpecs(net), estimator=_tabpfn_estimator)
WRAPPER_FAKES.update(
    dict.fromkeys(("TabPFN3Model", "RealTabPFNv25Model", "TabPFNv26Model", "TabPFNWideModel"), _TABPFN_FAKE)
)


def _tabpfn_seam_guard() -> None:
    """Tabpfn facts the preset relies on, checked against the installed library without a checkpoint.

    The shared specs survive ``copy.deepcopy`` and ``sklearn.base.clone`` as the same object (the
    many-class wrapper clones its base estimator per row), ``initialize_tabpfn_model`` unpacks a
    specs object handed in as ``model_path``, the engine's per-device cache reuses a module already
    resident on the requested device, and the loader call in ``build_shared_model_specs`` names
    parameters the library still has.
    """
    pytest.importorskip("tabpfn")
    import copy
    import inspect

    from sklearn.base import clone
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.base import initialize_tabpfn_model
    from tabpfn.inference import InferenceEngine, _PerDeviceModelCache
    from tabpfn.model_loading import load_model_criterion_config

    from tabarena.models.tabpfnv2_5._estimators import SharedClassifierModelSpecs, SharedRegressorModelSpecs

    net = _TinyNet()
    specs = SharedClassifierModelSpecs(net, SimpleNamespace(), SimpleNamespace())
    assert copy.deepcopy(specs) is specs
    assert clone(TabPFNClassifier(model_path=specs, device="cpu")).model_path is specs
    models, configs, criterion, inference_config = initialize_tabpfn_model(specs, "classifier", "fit_preprocessors")
    assert models == [net] and configs[0] is specs.architecture_config and criterion is None
    assert inference_config is specs.inference_config

    bardist = torch.nn.Identity()
    regressor_specs = SharedRegressorModelSpecs(net, SimpleNamespace(), SimpleNamespace(), bardist)
    assert copy.deepcopy(regressor_specs) is regressor_specs
    assert clone(TabPFNRegressor(model_path=regressor_specs, device="cpu")).model_path is regressor_specs
    models, _configs, criterion, _inference_config = initialize_tabpfn_model(
        regressor_specs, "regressor", "fit_preprocessors"
    )
    assert models == [net] and criterion is bardist

    cache = _PerDeviceModelCache(net)
    net.to_calls.clear()
    cache.to((torch.device("cpu"),))
    assert cache.get(torch.device("cpu")) is net and not net.to_calls
    assert callable(getattr(InferenceEngine, "_set_models", None))
    assert all(callable(getattr(estimator_cls, "to", None)) for estimator_cls in (TabPFNClassifier, TabPFNRegressor))
    parameters = set(inspect.signature(load_model_criterion_config).parameters)
    assert {
        "model_path",
        "check_bar_distribution_criterion",
        "cache_trainset_representation",
        "estimator_type",
        "version",
        "download_if_not_exists",
    } <= parameters


REPLICA_DRIFT_GUARDS["tabpfn"] = _tabpfn_seam_guard

# tabicl (TabICLModel, TabICLv2Model): seam="load_model", detach_attr="library"; the generic derived
# weightless estimator covers the library seam, so no custom fake is needed.

# tabdpt (TabDPTTurboModel): constructor replica, owned_tensor_attrs=("V",), flag_attrs use_flash; the generic
# attr-seam estimator expresses every spec path, so no custom fake is needed. TabDPTModel (v1.1) pins no
# ``compile`` flag and stays off the mixin.


def _tabdpt_replica_guard() -> None:
    """The replica constructor still matches tabdpt's, and builds the library's attribute set around a network."""
    pytest.importorskip("tabdpt")
    from tabdpt.estimator import TabDPTEstimator

    est = importlib.import_module("tabarena.models.tabdpt._estimators")
    est.check_replica_signature()
    network = SimpleNamespace(num_features=4, n_out=3, use_flash=False, clip_sigma=8.0)
    shared = est.SharedTabDPTClassifier.from_shared(network, device="cpu", compile=False, model_weight_path="ckpt")
    assert shared.model is network
    assert shared.path == "ckpt" and shared.compile is False
    with pytest.raises(ValueError, match="clip_sigma"):
        est.SharedTabDPTClassifier.from_shared(
            network, device="cpu", compile=False, model_weight_path="ckpt", clip_sigma=4
        )
    try:
        checkpoint = hub.resolve_hf_file("Layer6/TabDPT", "tabdpt1_2.safetensors", allow_download=False)
    except hub.WeightsUnavailableError:
        pytest.skip("the TabDPT v1.2 checkpoint is not cached; attribute parity needs the library constructor")
    library = TabDPTEstimator(mode="cls", device="cpu", compile=False, model_weight_path=checkpoint)
    assert set(vars(library)) == set(vars(shared)), "the replica sets other attributes than the library constructor"


REPLICA_DRIFT_GUARDS["tabdpt"] = _tabdpt_replica_guard

# causilo (CausiloModel): the network sits on a pre-built engine (``_engine.model``) that the estimator's
# ``attach_network`` swaps; the library pickle is weightless (``detach_attr="library"``).


class _CausiloEstimator:
    """Stand-in for the shared Causilo estimators: an engine holds the network, ``attach_network`` swaps it."""

    def __init__(self, payload: Any, device: str) -> None:
        self.device = device
        self._engine = SimpleNamespace(
            model=None, device=torch.device(device), device_request=device, task="classification", state=None
        )
        self.n_features_in_ = 3
        if payload is not None:
            self.attach_network(payload)

    def attach_network(self, network: Any, *, device: torch.device | None = None) -> None:
        if device is not None and device.type != self._engine.device.type:
            self._engine.device = device
            self._engine.device_request = str(device)
            self.device = str(device)
        self._engine.model = network

    def network_detached(self) -> bool:
        return self._engine.model is None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_engine"] = SimpleNamespace(**{**vars(self._engine), "model": None})
        return state

    def predict_proba(self, X):
        return self._engine.model(torch.as_tensor(np.asarray(X), dtype=torch.float32))


def _causilo_estimator(spec, payload, key, device, loader) -> _CausiloEstimator:
    return _CausiloEstimator(payload, device)


def _causilo_seam_guard() -> None:
    """``Engine.fit`` loads only into an empty engine, ``fit_adapter`` reuses a matching engine, the pickle is weightless."""
    engine = pytest.importorskip("causilo.engine")
    estimators = pytest.importorskip("causilo.estimators")
    serialization = pytest.importorskip("causilo.serialization")
    assert "if self.model is None:" in inspect.getsource(engine.Engine.fit)
    assert "engine.device_request != estimator.device" in inspect.getsource(estimators.fit_adapter)
    for estimator in (estimators.CausiloClassifier, estimators.CausiloRegressor):
        assert estimator.__getstate__ is serialization.export_estimator
        assert estimator.__setstate__ is serialization.import_estimator


WRAPPER_FAKES["CausiloModel"] = WrapperFake(estimator=_causilo_estimator)
REPLICA_DRIFT_GUARDS["causilo"] = _causilo_seam_guard

# tabldm (TabLDMModel): seam="load_model", detach_attr="library", freeze_parameters=False; the generic
# derived weightless estimator covers the library seam.


def _tabldm_seam_guard() -> None:
    """``fit`` and ``__setstate__`` build the network in ``_load_model`` after ``_resolve_device``; the pickle drops it."""
    tabldm = pytest.importorskip("tabldm")
    for estimator in (tabldm.TabLDMEnhancedClassifier, tabldm.TabLDMEnhancedRegressor):
        fit_source = inspect.getsource(estimator.fit)
        assert (
            fit_source.index("self._resolve_device()")
            < fit_source.index("self._load_model()")
            < fit_source.index("self.model_.to(self.device_)")
        )
        setstate_source = inspect.getsource(estimator.__setstate__)
        assert setstate_source.index("self._resolve_device()") < setstate_source.index("self._load_model()")
        assert "self._build_inference_config()" in setstate_source and "self._move_cache_to_device()" in setstate_source
        getstate_source = inspect.getsource(estimator.__getstate__)
        for dropped in ("model_", "device_", "inference_config_", "model_path_"):
            assert f'state.pop("{dropped}", None)' in getstate_source, dropped


REPLICA_DRIFT_GUARDS["tabldm"] = _tabldm_seam_guard

# mitra_v2 (MitraV2Model): state_dict mode (the default safetensors reader is the build hook); the
# estimator copies the cached dict into each trainer's meta-built Tab2D, so no estimator fake is needed.

# tabfm (TabFMModel): constructor keyword model=, device_from_params rule. A CPU fit shares only when the
# configuration asks for the CPU, which the generic CPU tests do here.

WRAPPER_FAKES["TabFMModel"] = WrapperFake(generic_hyperparameters={"device": "cpu"})

# tabswift (TabSwiftModel): seam="load_model" on the vendored estimator, user_path_param="model_path",
# owned_tensor_attrs for the PCA basis; the generic derived estimator covers the seam.


def _tabswift_seam_guard() -> None:
    """The vendored ``fit`` resolves ``device_``, calls ``_load_model`` and moves ``model_``; the tree lives on ``icl_predictor.root``."""
    from tabarena.models.tabswift._vendor.classifier import TabSwiftClassifier
    from tabarena.models.tabswift._vendor.model.learning import ICLearning
    from tabarena.models.tabswift._vendor.model.tabswift import TabSwift
    from tabarena.models.tabswift._vendor.regressor import TabSwiftRegressor

    for estimator in (TabSwiftClassifier, TabSwiftRegressor):
        fit_source = inspect.getsource(estimator.fit)
        assert (
            fit_source.index("self.device_ =")
            < fit_source.index("self._load_model()")
            < fit_source.index("self.model_.to(self.device_)")
        )
    assert "self.icl_predictor = ICLearning(" in inspect.getsource(TabSwift.__init__)
    assert "self.root = ClassNode(" in inspect.getsource(ICLearning)


REPLICA_DRIFT_GUARDS["tabswift"] = _tabswift_seam_guard

# exaone_tabular (EXAONETabularModel): constructor keyword model=, library manifest checkpoint hook (faked
# by ``_install_fakes`` like every resolver); the generic attr-seam estimator covers the rest.

# nori (NoriModel, Nori30MModel): the predictor (not just the module) is rebuilt per device through
# ``_estimators.predictor_cls()``, replaced here by a stand-in that records what it was built from.


class _FakeNoriPredictor:
    """The ``NoriPredictor`` surface the wrapper touches: built from ``device``, ``model_path`` and ``model``."""

    def __init__(self, device, model_path=None, model=None):
        self.device = device
        self.model_path = model_path
        self.model = model


class _FakeNoriRegressor:
    """A fitted ``NoriRegressor`` stand-in: a device field, a predictor, sklearn's ``get_params``."""

    def __init__(self, device: str, predictor: _FakeNoriPredictor) -> None:
        self.device = torch.device(device)
        self._predictor = predictor

    def get_params(self, deep: bool = False) -> dict:
        return {}

    def _get_predictor(self) -> _FakeNoriPredictor:
        return self._predictor

    def predict(self, X):
        return self._predictor.model(torch.as_tensor(np.asarray(X), dtype=torch.float32))


def _nori_estimator(spec, payload, key, device, loader) -> _FakeNoriRegressor:
    checkpoint = key.checkpoint if key is not None else "library"
    return _FakeNoriRegressor(device, _FakeNoriPredictor(torch.device(device), model_path=checkpoint, model=payload))


def _nori_predictor_kwargs_guard() -> None:
    """``_estimators.predictor_kwargs`` derives exactly the arguments ``NoriRegressor._get_predictor`` passes."""
    pytest.importorskip("synthefy_nori")
    import synthefy_nori.inference.predictor as predictor_module
    from synthefy_nori import NoriRegressor

    from tabarena.models.nori._estimators import predictor_kwargs

    recorded: dict[str, Any] = {}

    class _Recorder:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

    estimator = NoriRegressor(model_path="/nowhere/nori.pt", device="cpu")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(predictor_module, "NoriPredictor", _Recorder)
        estimator._get_predictor()
    derived = predictor_kwargs(estimator)
    assert set(recorded) - {"device", "model_path"} == set(derived)
    assert {name: recorded[name] for name in derived} == derived


WRAPPER_FAKES["NoriModel"] = WrapperFake(
    estimator=_nori_estimator,
    library_free_patches=(("tabarena.models.nori._estimators.predictor_cls", lambda: _FakeNoriPredictor),),
)
WRAPPER_FAKES["Nori30MModel"] = WRAPPER_FAKES["NoriModel"]
REPLICA_DRIFT_GUARDS["nori"] = _nori_predictor_kwargs_guard

# limix (LimiXModel): the predictor and every ``InferenceAttentionMap`` retrieval step hold the network;
# the stand-in carries one such step so attach and detach cover the pipelines.


class _FakeAttentionStep:
    """An ``InferenceAttentionMap`` stand-in: a retrieval step holding its own reference to the network."""

    def __init__(self, model) -> None:
        self.calculate_sample_attention = True
        self.model = model


class _FakeLimiXPredictor(_PathEstimator):
    def __init__(self, spec: SharedWeightsSpec, payload: Any, device: str) -> None:
        super().__init__(spec, payload, device)
        self.preprocess_pipelines = [[SimpleNamespace(name="shuffle"), _FakeAttentionStep(payload)]]


WRAPPER_FAKES["LimiXModel"] = WrapperFake(
    estimator=lambda spec, payload, key, device, loader: _FakeLimiXPredictor(spec, payload, device)
)

# orionmsp (OrionMSPModel): seam="load_model" like tabicl, uses_kv_cache predicate; the generic derived
# estimator covers the seam.


def _orionmsp_seam_guard() -> None:
    """``fit`` builds the network in ``_load_model`` then moves it; the patched embedding hook exists; two builds agree."""
    classifier = pytest.importorskip("tabtune.models.orionmsp_v15.sklearn.classifier")
    embedding = pytest.importorskip("tabtune.models.orionmsp_v15.model.embedding")
    from tabarena.models.orionmsp.model import OrionMSPModel

    fit_source = inspect.getsource(classifier.OrionMSPv15Classifier.fit)
    assert fit_source.index("self._load_model()") < fit_source.index("self.model_.to(self.device_)")
    assert callable(getattr(embedding.ColEmbedding, "_add_feature_pos_emb", None))
    try:
        key = OrionMSPModel.shared_weights_key(
            problem_type="binary", hyperparameters={}, device="cpu", allow_download=False
        )
    except hub.WeightsUnavailableError:
        pytest.skip("the OrionMSP checkpoint is not cached; prefetch it to run the two-build check")
    a, b = OrionMSPModel._load_shared_weights(key), OrionMSPModel._load_shared_weights(key)
    assert not a.training and not any(p.requires_grad for p in a.parameters())
    assert a.state_dict().keys() == b.state_dict().keys()
    assert all(torch.equal(a.state_dict()[name], b.state_dict()[name]) for name in a.state_dict())


REPLICA_DRIFT_GUARDS["orionmsp"] = _orionmsp_seam_guard

# sap_rpt_oss (SAPRPTOSSModel): the payload is a dataclass holding the network and the sentence embedder;
# the estimator stand-in carries the library's ``tokenizer.sentence_embedder`` surface.


def _sap_payload(net: _TinyNet, key: w.WeightsKey):
    from tabarena.models.sap_rpt_oss.model import SharedSAPRPTWeights

    return SharedSAPRPTWeights(
        module=net,
        sentence_embedder=SimpleNamespace(model=_TinyNet(), device=torch.device(key.device)),
        device=torch.device(key.device),
        dtype=getattr(torch, key.dtype),
        checkpoint_path=key.checkpoint,
        embedder_dir="embedder",
    )


def _sap_estimator(spec, payload, key, device, loader) -> _PathEstimator:
    from tabarena.models.sap_rpt_oss.model import SharedSAPRPTWeights, attach_shared_weights

    estimator = _PathEstimator(spec, None, device)
    if isinstance(payload, SharedSAPRPTWeights):
        attach_shared_weights(estimator, payload)
    else:  # an estimator that owns its objects (the library constructor's result)
        estimator.model = payload
        estimator.device = torch.device(device)
        estimator.tokenizer.sentence_embedder = SimpleNamespace(model=_TinyNet(), device=torch.device(device))
    return estimator


def _sap_rpt_oss_replica_guard() -> None:
    """The replicated constructors still match the library's, and ``from_shared`` sets the library's attribute set."""
    pytest.importorskip("sap_rpt_oss")
    from sap_rpt_oss.data.tokenizer import Tokenizer
    from sap_rpt_oss.rpt import SAP_RPT_OSS_Estimator

    est = importlib.import_module("tabarena.models.sap_rpt_oss._estimators")
    est.check_library_signatures()
    se.check_signature_matches(SAP_RPT_OSS_Estimator.__init__, est._SharedEstimatorMixin.from_shared, skip=("shared",))
    key = w.make_key("sap_rpt_oss", "/nowhere/ckpt.pt", "network", "cpu")
    shared = _sap_payload(_TinyNet(), key)
    estimator = est.SharedSAPRPTClassifier.from_shared(shared, bagging=2, max_context_size=64, test_chunk_size=10)
    assert set(vars(estimator)) == {
        "model_size",
        "checkpoint",
        "regression_type",
        "classification_type",
        "test_chunk_size",
        "_checkpoint_path",
        "bagging",
        "max_context_size",
        "num_regression_bins",
        "model",
        "device",
        "dtype",
        "seed",
        "drop_constant_columns",
        "tokenizer",
    }
    assert estimator.model is shared.module
    assert estimator.tokenizer.sentence_embedder is shared.sentence_embedder
    assert isinstance(estimator.tokenizer, Tokenizer)
    assert estimator.tokenizer.random_seed == 42 and estimator.tokenizer.is_valid is True
    assert estimator.get_params() == {
        "bagging": 2,
        "checkpoint": "2025-11-04_sap-rpt-one-oss.pt",
        "drop_constant_columns": True,
        "max_context_size": 64,
        "test_chunk_size": 10,
    }
    with pytest.raises(ValueError, match="bagging"):
        est.SharedSAPRPTClassifier.from_shared(shared, bagging="most")


WRAPPER_FAKES["SAPRPTOSSModel"] = WrapperFake(payload=_sap_payload, estimator=_sap_estimator)
REPLICA_DRIFT_GUARDS["sap_rpt_oss"] = _sap_rpt_oss_replica_guard

# tabstar (TabSTARModel): state_dict mode on AbstractModel (no device API); no estimator fake is needed.
# The replicas in tabstar/_estimators.py (SharedTabStarTrainer.__init__ mirrors TabStarTrainer.__init__,
# build_base_model mirrors TabStarModel.from_pretrained) are guarded by the models-marked drift test.


def _tabstar_replicas_match_the_library() -> None:
    """Trainer constructor parity, and a state-dict build equal to the library's file build tensor for tensor.

    Resolves the base checkpoint and the text encoder from the Hub when they are not cached, then
    builds the base model both ways on the CPU: the shared build leaves torch's generator alone,
    matches ``from_pretrained`` in every tensor, dtype and config field, and owns its storage (the
    library's ``from_pretrained(state_dict=...)`` assigns tensors, so a CPU build must copy them).
    """
    pytest.importorskip("tabstar")
    from tabstar.training.trainer import TabStarTrainer
    from tabstar.training.utils import fix_seed

    from tabarena.models.tabstar import model as tm

    est = importlib.import_module("tabarena.models.tabstar._estimators")
    se.check_signature_matches(TabStarTrainer.__init__, est.SharedTabStarTrainer.__init__, skip=("base_state_dict",))

    base_dir = tm.resolve_base_model_dir(allow_download=True)
    est.use_local_text_encoder(tm.resolve_text_encoder_dir(allow_download=True))
    key = tm.TabSTARModel.shared_weights_key(problem_type="binary", hyperparameters={}, device="cpu")
    state_dict = tm.TabSTARModel._load_shared_weights(key)
    ids = {name: id(tensor) for name, tensor in state_dict.items()}

    fix_seed(0)
    before = torch.get_rng_state().clone()
    stock = est.TabStarModel.from_pretrained(base_dir, local_files_only=True)
    assert torch.equal(torch.get_rng_state(), before)
    shared = est.build_base_model(base_dir, state_dict, device="cpu")
    assert torch.equal(torch.get_rng_state(), before)
    assert {name: id(tensor) for name, tensor in state_dict.items()} == ids, "the registry's dict is not mutated"
    sd_stock, sd_shared = stock.state_dict(), shared.state_dict()
    assert set(sd_stock) == set(sd_shared) == set(state_dict)
    assert all(torch.equal(sd_stock[n], sd_shared[n]) and sd_stock[n].dtype == sd_shared[n].dtype for n in sd_stock)
    assert stock.config.to_dict() == shared.config.to_dict()
    assert shared.training is stock.training is False
    ptrs_cache = {t.untyped_storage().data_ptr() for t in state_dict.values()}
    assert not any(p.untyped_storage().data_ptr() in ptrs_cache for p in shared.parameters()), (
        "a CPU build owns its weights"
    )


REPLICA_DRIFT_GUARDS["tabstar"] = _tabstar_replicas_match_the_library

# iltm (ILTMModel): opts out of the mixin; the library shares through its own class-level cache.
# No block, no spec, not discovered here.


def _block(cls: type) -> WrapperFake:
    return WRAPPER_FAKES.get(cls.__name__, WrapperFake())


# --- fixtures and helpers ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Every test starts and ends with an empty registry, zeroed stats, the default capacity and empty memos."""
    w.release()
    w.reset_stats()
    w.set_capacity(w.DEFAULT_CAPACITY)
    for cls in MIXIN_USERS:
        cls._shared_weights_memo.clear()
    yield
    w.release()
    w.reset_stats()
    w.set_capacity(w.DEFAULT_CAPACITY)


@pytest.fixture(autouse=True)
def _no_cuda(request, monkeypatch):
    """The generic tests run as on a CUDA-less host; the ``models``-marked tests see the real hardware."""
    if request.node.get_closest_marker("models") is None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@dataclasses.dataclass
class _Fakes:
    ckpt: Path
    builds: list[w.WeightsKey] = dataclasses.field(default_factory=list)
    resolves: list[tuple[str, bool, str]] = dataclasses.field(default_factory=list)


def _install_fakes(monkeypatch, cls: type, tmp_path: Path) -> _Fakes:
    """Fake checkpoint resolution and the build hook so no library and no Hub is touched."""
    ckpt = tmp_path / "ckpt.bin"
    ckpt.write_bytes(b"0")
    fakes = _Fakes(ckpt=ckpt)
    spec = cls.shared_weights_spec
    block = _block(cls)

    def fake_resolve(klass, *, problem_type, variant, hyperparameters, allow_download, stage="fit"):
        fakes.resolves.append((variant, allow_download, stage))
        return ResolvedCheckpoint(str(ckpt), source={"repo_id": "fake", "filename": "ckpt.bin", "revision": "r"})

    def fake_build(klass, key):
        fakes.builds.append(key)
        if spec.mode == "state_dict":
            return {"w": torch.zeros(2)}
        net = _TinyNet()
        net.shared_guard = True
        return block.payload(net, key)

    monkeypatch.setattr(cls, "_resolve_shared_checkpoint", classmethod(fake_resolve))
    monkeypatch.setattr(cls, "_build_shared_weights", classmethod(fake_build))
    for target, replacement in block.library_free_patches:
        module_name, attr = target.rsplit(".", 1)
        monkeypatch.setattr(importlib.import_module(module_name), attr, replacement)
    return fakes


def _smoke_hps(cls: type, *, real: bool = False) -> dict:
    """The smoke hyperparameters of ``cls``; the generic CPU tests (``real=False``) add the wrapper's overlay."""
    hps = dict(smoke_for(METHOD_BY_CLASS[cls]).hyperparameters)
    hps.update(getattr(cls, "warmup_dummy_fit_hyperparameters", None) or {})
    if not real:
        hps.update(_block(cls).generic_hyperparameters)
    hps["ag_args_fit"] = {"max_memory_usage_ratio": None}
    return hps


def _model(cls: type, problem_type: str, hyperparameters: dict, tmp_path: Path, name: str = "model"):
    return cls(
        path=str(tmp_path / name) + os.sep,
        name=name,
        problem_type=problem_type,
        eval_metric=None,
        hyperparameters=dict(hyperparameters),
    )


def _fake_fit(model, fakes: _Fakes, device: str = "cpu") -> tuple[w.WeightsKey, Any]:
    """What a wrapper's ``_fit`` does around the library call, with the estimator stand-in instead of the library."""
    cls = type(model)
    spec = cls.shared_weights_spec
    model.initialize()
    key, payload = model._acquire_shared_weights(device=device)
    assert key is not None, f"{cls.__name__} did not share under its smoke hyperparameters"
    make = _block(cls).estimator or (_load_model_seam_estimator if spec.seam == "load_model" else _attr_seam_estimator)
    model.model = make(spec, payload, key, device, cls._load_shared_weights)
    model.device = device
    model.device_train = device
    return key, payload


def _attached_network(model) -> Any:
    return se.get_by_path(model.model, type(model).shared_weights_spec.network_attr)


def _detach(model) -> None:
    se.set_by_path(model.model, type(model).shared_weights_spec.network_attr, None)
    assert not model._network_attached()


def _patch_fake_cuda(monkeypatch) -> None:
    """Let a payload or tensor tagged ``cuda`` pass the device checks and tag instead of moving owned tensors."""
    real_payload_device = se.payload_device

    def fake_payload_device(payload):
        for module in se.payload_modules(payload):
            tag = getattr(module, "_fake_device", None)
            if tag:
                return tag
        return real_payload_device(payload)

    def fake_move(obj, paths, device, *, copy_holder):
        for path in paths:
            tensor = se.get_by_path(obj, path)
            if isinstance(tensor, torch.Tensor):
                moved = _DeviceTaggedTensor(tensor.detach().cpu())
                moved._fake_cuda = w.normalize_device(device) == "cuda"
                se.set_by_path(obj, path, moved)
        return obj

    monkeypatch.setattr(se, "payload_device", fake_payload_device)
    monkeypatch.setattr(swm, "payload_device", fake_payload_device)
    monkeypatch.setattr(swm, "move_tensor_attrs", fake_move)


# --- class-level conventions --------------------------------------------------------------------------


def test_replaced_wrappers_use_the_mixin():
    names = {cls.__name__ for cls in MIXIN_USERS}
    missing = EXPECTED_MIXIN_USERS - names
    assert not missing, f"migrated wrappers dropped the mixin or their spec: {sorted(missing)}"
    owners = [cls._class_settings_owner() for cls in MIXIN_USERS]
    assert len(set(owners)) == len(owners), "two registered classes share one class_settings owner"


@pytest.mark.parametrize("cls", MIXIN_USERS, ids=CLASS_IDS)
def test_spec_is_valid_and_settings_are_per_registered_class(cls):
    spec = cls.shared_weights_spec
    assert isinstance(spec, SharedWeightsSpec)
    assert spec.mode in ("module", "state_dict")
    assert issubclass(cls.__dict__.get("class_settings_cls"), w.SharedWeightsClassSettings)
    assert wu.collect_warmup_modules(cls), "declare warmup_modules"
    for problem_type in _problem_types(cls):
        assert (
            cls._get_default_ag_args_ensemble(problem_type=problem_type)["fold_fitting_strategy"] == "sequential_local"
        )

    class _OwnSettings(w.SharedWeightsClassSettings):
        pass

    declared = type("_Declared", (cls,), {"ag_key": "_DECLARED", "class_settings_cls": _OwnSettings})
    assert declared.class_settings_cls is _OwnSettings

    class _Passthrough(SharedWeightsModelMixin, AbstractTorchModel):
        pass

    assert SharedWeightsModelMixin._class_tags() == {}
    assert _Passthrough._get_class_tags() == AbstractTorchModel._get_class_tags()
    assert _Passthrough.shared_weights_key(problem_type="binary", hyperparameters={}, device="cpu") is None
    assert _Passthrough.prefetch_weights() == []


@pytest.mark.parametrize(("cls", "problem_type"), CASES, ids=CASE_IDS)
def test_class_tags(cls, problem_type):
    tags = cls._get_class_tags()
    if cls.shared_weights_spec.mode == "module":
        assert tags["set_device_on_save_to"] is None
        assert tags["can_set_device"] is True
        assert tags["set_device_on_load"] is True
    else:
        base = next(
            b for b in cls.__mro__[1:] if not issubclass(b, SharedWeightsModelMixin) and hasattr(b, "_get_class_tags")
        )
        assert tags == base._get_class_tags()


# --- key derivation and sharing decisions -------------------------------------------------------------


@pytest.mark.parametrize(("cls", "problem_type"), CASES, ids=CASE_IDS)
def test_key_is_identical_for_warmup_and_fit_paths(cls, problem_type, tmp_path, monkeypatch):
    fakes = _install_fakes(monkeypatch, cls, tmp_path)
    hps = _smoke_hps(cls)
    report = wu.WarmupReport()
    wu.warmup_shared_weights(cls, problem_type=problem_type, num_gpus=0, hyperparameters=hps, report=report)
    assert report.weights_preloaded and not report.failed_steps, report.steps
    assert len(fakes.builds) == 1
    key = cls.shared_weights_key(problem_type=problem_type, hyperparameters=wu.strip_ag_args(hps), device="cpu")
    entry = w.peek(key)
    assert entry is not None and entry.loaded_by == "warmup"
    assert entry.source == {**cls.shared_weights_source(key), "stage": "warmup"}
    assert entry.source["repo_id"] == "fake"

    model = _model(cls, problem_type, hps, tmp_path)
    model.initialize()
    fit_key, payload = model._acquire_shared_weights(device="cpu")
    assert fit_key == key
    assert payload is entry.value
    assert len(fakes.builds) == 1
    assert entry.hits == 1
    assert model._present_before_fit is True
    assert model.get_info()["shared_weights"]["loaded_by"] == "warmup"


@pytest.mark.parametrize(("cls", "problem_type"), CASES, ids=CASE_IDS)
def test_two_children_share_one_loader_call(cls, problem_type, tmp_path, monkeypatch):
    fakes = _install_fakes(monkeypatch, cls, tmp_path)
    hps = _smoke_hps(cls)
    wu.warmup_shared_weights(cls, problem_type=problem_type, num_gpus=0, hyperparameters=hps, report=wu.WarmupReport())
    payloads = []
    for name in ("child_a", "child_b"):
        model = _model(cls, problem_type, hps, tmp_path, name=name)
        model.initialize()
        key, payload = model._acquire_shared_weights(device="cpu")
        payloads.append(payload)
    assert len(fakes.builds) == 1
    assert payloads[0] is payloads[1]
    assert w.peek(key).hits == 2


@pytest.mark.parametrize(("cls", "problem_type"), CASES, ids=CASE_IDS)
def test_disable_conditions_yield_no_key_and_no_sharing(cls, problem_type, tmp_path, monkeypatch):
    _install_fakes(monkeypatch, cls, tmp_path)
    spec = cls.shared_weights_spec
    base = _smoke_hps(cls)
    examples = [dict(example) for example in spec.unshareable_examples]
    examples.extend({name: True} for name in spec.disable_when if isinstance(name, str))
    for example in examples:
        hps = {**base, **example}
        assert (
            cls.shared_weights_key(problem_type=problem_type, hyperparameters=wu.strip_ag_args(hps), device="cpu")
            is None
        )
        model = _model(cls, problem_type, hps, tmp_path)
        model.initialize()
        assert not model._shares_network(), example

    saving = {**base, "ag_args_fit": {**base["ag_args_fit"], "save_pretrained_weights": True}}
    model = _model(cls, problem_type, saving, tmp_path)
    model.initialize()
    assert not model._shares_network()

    cls.set_class_settings(share_weights=False)
    try:
        model = _model(cls, problem_type, base, tmp_path)
        model.initialize()
        assert not model._shares_network()
    finally:
        cls.set_class_settings(share_weights=True)
    model = _model(cls, problem_type, base, tmp_path)
    model.initialize()
    assert model._shares_network()

    supported = cls.supported_problem_types() or list(PROBLEM_TYPES)
    for other in PROBLEM_TYPES:
        if other not in supported:
            assert (
                cls.shared_weights_key(problem_type=other, hyperparameters=wu.strip_ag_args(base), device="cpu") is None
            )


# --- pickling, devices and inference ------------------------------------------------------------------


@pytest.mark.parametrize(("cls", "problem_type"), MODULE_CASES, ids=MODULE_IDS)
def test_weightless_pickle_round_trip_reattaches_the_same_object(cls, problem_type, tmp_path, monkeypatch):
    fakes = _install_fakes(monkeypatch, cls, tmp_path)
    spec = cls.shared_weights_spec
    model = _model(cls, problem_type, _smoke_hps(cls), tmp_path)
    key, payload = _fake_fit(model, fakes)
    net = _net_of(payload)
    for path in spec.owned_tensor_attrs:
        _set_path_creating(model.model, path, _tagged_cuda_tensor())
    x = torch.ones(1, 3)
    before = net(x).detach().clone()

    state = model.__getstate__()
    payload_tensors = {id(t) for t in (*net.parameters(), *net.buffers())}
    for path, tensor in walk_tensors(state):
        assert tensor.device.type == "cpu", f"{path} pickled on {tensor.device}"
        assert id(tensor) not in payload_tensors, f"{path} is a shared payload tensor"
    if spec.detach_attr != "library":
        assert not se.get_by_path(state["model"], spec.network_attr), "the pickled copy still holds the network"
    assert _net_of(_attached_network(model)) is net, "the live estimator lost its network"
    for path in spec.owned_tensor_attrs:
        assert getattr(se.get_by_path(model.model, path), "_fake_cuda", False), "the live estimator's tensor was moved"

    loaded = pickle.loads(pickle.dumps(model))
    loaded._ensure_network()
    assert _net_of(_attached_network(loaded)) is net
    assert len(fakes.builds) == 1
    for path in spec.owned_tensor_attrs:
        assert se.get_by_path(loaded.model, path).device.type == "cpu"
    assert torch.equal(net(x), before)
    assert loaded.get_info()["shared_weights"]["device"] == "cpu"


@pytest.mark.parametrize(("cls", "problem_type"), MODULE_CASES, ids=MODULE_IDS)
def test_set_device_swaps_registry_entries_and_never_moves_the_shared_module(cls, problem_type, tmp_path, monkeypatch):
    fakes = _install_fakes(monkeypatch, cls, tmp_path)
    spec = cls.shared_weights_spec
    model = _model(cls, problem_type, _smoke_hps(cls), tmp_path)
    key, payload = _fake_fit(model, fakes)
    net = _net_of(payload)
    for path in spec.owned_tensor_attrs:
        _set_path_creating(model.model, path, torch.zeros(2))
    net.to_calls.clear()

    model._set_device("cpu")
    assert len(fakes.builds) == 1
    assert not net.to_calls
    assert model._shared_key == key

    _patch_fake_cuda(monkeypatch)
    cuda_key = model._key_for_device("cuda")
    assert cuda_key.device == "cuda"
    if callable(spec.dtype):
        assert cuda_key.dtype == spec.dtype(model._shared_hps, "cuda")
    cuda_net = _TinyNet()
    cuda_net._fake_device = "cuda"
    cuda_net.shared_guard = True
    cuda_payload = _block(cls).payload(cuda_net, cuda_key)
    w.prime(cuda_key, lambda: cuda_payload)

    model._set_device("cuda")
    assert _net_of(_attached_network(model)) is cuda_net
    assert model._shared_key == cuda_key
    assert not net.to_calls and not cuda_net.to_calls
    assert len(fakes.builds) == 1
    for path, kind in spec.device_attrs:
        value = se.get_by_path(model.model, path)
        if kind == "torch":
            assert isinstance(value, torch.device) and value.type == "cuda", path
        else:
            assert value == "cuda", path
    expected_flags = spec.flags_for(model._shared_hps, "cuda")
    for name, paths in spec.flag_attrs.items():
        for path in paths:
            assert se.get_by_path(model.model, path) == expected_flags[name], path
    for path in spec.owned_tensor_attrs:
        assert se.get_by_path(model.model, path).device.type == "cuda", path
    assert model.get_device() == "cuda"

    owned = _model(cls, problem_type, _smoke_hps(cls), tmp_path, name="owned")
    owned.initialize()
    own_net = _TinyNet()
    make = _block(cls).estimator or (_load_model_seam_estimator if spec.seam == "load_model" else _attr_seam_estimator)
    owned.model = make(spec, own_net, None, "cpu", cls._load_shared_weights)
    owned_net = _net_of(_attached_network(owned))
    owned_net.to_calls.clear()
    owned._set_device("cpu")
    assert owned_net.to_calls, "an estimator that owns its network is moved in place"
    assert len(fakes.builds) == 1


@pytest.mark.parametrize(("cls", "problem_type"), MODULE_CASES, ids=MODULE_IDS)
def test_prepare_for_inference_is_idempotent_and_model_only(cls, problem_type, tmp_path, monkeypatch):
    fakes = _install_fakes(monkeypatch, cls, tmp_path)
    model = _model(cls, problem_type, _smoke_hps(cls), tmp_path)
    key, payload = _fake_fit(model, fakes)
    net = _net_of(payload)
    _detach(model)
    net.train()
    builds = len(fakes.builds)

    model.prepare_for_inference()
    assert model._network_attached()
    assert _net_of(_attached_network(model)) is net
    assert len(fakes.builds) == builds
    assert all(not module.training for module in model._shared_modules())

    model.prepare_for_inference()
    assert len(fakes.builds) == builds
    assert net.forward_calls == 0


@pytest.mark.parametrize(("cls", "problem_type"), CASES, ids=CASE_IDS)
def test_get_info_shared_weights_block(cls, problem_type, tmp_path, monkeypatch):
    _install_fakes(monkeypatch, cls, tmp_path)
    model = _model(cls, problem_type, _smoke_hps(cls), tmp_path)
    model.initialize()
    key, _payload = model._acquire_shared_weights(device="cpu")
    block = model.get_info()["shared_weights"]
    assert set(block) == {
        "library",
        "checkpoint",
        "variant",
        "device",
        "dtype",
        "flags",
        "loaded_by",
        "present_before_fit",
        "checkpoint_source",
    }
    assert block["checkpoint"] == Path(key.checkpoint).name
    assert block["loaded_by"] == "fit"
    assert block["present_before_fit"] is False
    assert str(tmp_path) not in json.dumps(block), "an absolute host path leaked into the metadata"

    unshared = _model(cls, problem_type, _smoke_hps(cls), tmp_path, name="unshared")
    unshared.initialize()
    assert unshared.get_info()["shared_weights"] is None


@pytest.mark.parametrize(("cls", "problem_type"), MODULE_CASES, ids=MODULE_IDS)
def test_memory_size_counts_the_shared_network_from_its_tensors(cls, problem_type, tmp_path, monkeypatch):
    fakes = _install_fakes(monkeypatch, cls, tmp_path)
    model = _model(cls, problem_type, _smoke_hps(cls), tmp_path)
    key, payload = _fake_fit(model, fakes)
    net = _net_of(payload)
    modules = list(model._shared_modules())
    assert net in modules
    assert model._get_memory_size() == model._get_pickled_size() + w.tensor_bytes(modules)
    state = model.__getstate__()
    assert sys.getsizeof(pickle.dumps(state)) < sys.getsizeof(pickle.dumps({**state, "network": net}))


@pytest.mark.parametrize(("cls", "problem_type"), STATE_DICT_CASES, ids=STATE_DICT_IDS)
def test_state_dict_mode_payload_is_a_cpu_state_dict_that_is_never_pickled(cls, problem_type, tmp_path, monkeypatch):
    _install_fakes(monkeypatch, cls, tmp_path)
    spec = cls.shared_weights_spec
    model = _model(cls, problem_type, _smoke_hps(cls), tmp_path)
    model.initialize()
    key, payload = model._acquire_shared_weights(device="cuda")
    assert key.device == spec.cache_device == "cpu"
    assert isinstance(payload, dict) and all(t.device.type == "cpu" for t in payload.values())

    class _Estimator(se.SharedStateDictEstimatorMixin):
        pass

    estimator = _Estimator()
    estimator.configure_shared_weights(payload)
    assert "_shared_state_dict" not in estimator.__getstate__()

    class _Holder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.empty(2))

    built = se.build_from_state_dict(_Holder, payload, "cpu")
    assert torch.equal(built.w.detach(), payload["w"])
    assert built.w.data_ptr() != payload["w"].data_ptr()


@pytest.mark.parametrize(("cls", "problem_type"), CASES, ids=CASE_IDS)
def test_loader_honours_fetch_policy_and_leaves_rng_untouched(cls, problem_type, tmp_path, monkeypatch):
    fakes = _install_fakes(monkeypatch, cls, tmp_path)
    hps = wu.strip_ag_args(_smoke_hps(cls))
    key = cls.shared_weights_key(problem_type=problem_type, hyperparameters=hps, device="cpu")
    fakes.ckpt.unlink()
    with pytest.raises(PretrainedWeightsUnavailableError):
        cls._load_shared_weights(key, allow_download=False)
    assert fakes.resolves[-1][1] is False
    monkeypatch.setenv(FETCH_ENV_VAR, "False")
    with pytest.raises(PretrainedWeightsUnavailableError):
        cls._load_shared_weights(key)
    assert fakes.resolves[-1][1] is False
    monkeypatch.setenv(FETCH_ENV_VAR, "True")
    with pytest.raises(PretrainedWeightsUnavailableError):
        cls._load_shared_weights(key)
    assert fakes.resolves[-1][1] is True
    monkeypatch.delenv(FETCH_ENV_VAR)

    fakes.ckpt.write_bytes(b"0")
    py_state = random.getstate()
    np_state = np.random.get_state()[1].copy()
    torch_state = torch.get_rng_state().clone()
    w.get_or_load(key, partial(cls._load_shared_weights, key))
    assert len(fakes.builds) == 1
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state)
    assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.parametrize("cls", MIXIN_USERS, ids=CLASS_IDS)
def test_old_pickles_without_shared_fields_load(cls):
    model = cls.__new__(cls)
    model.__setstate__({"model": None, "name": "legacy"})
    assert model._shared_key is None
    assert model._present_before_fit is None
    assert model._checkpoint_source is None
    assert model._shared_hps is None
    assert model.name == "legacy"


def test_mixin_module_is_torch_free():
    code = """
        import sys
        import tabarena.models._shared_weights_model, tabarena.models._shared_estimators
        assert "torch" not in sys.modules, "the shared-weights modules import torch at import time"
        """
    subprocess.run([sys.executable, "-P", "-c", textwrap.dedent(code)], check=True, timeout=300)  # noqa: S603


@pytest.mark.parametrize(("cls", "problem_type"), MODULE_CASES, ids=MODULE_IDS)
def test_hooks_run_without_the_library(cls, problem_type, tmp_path, monkeypatch):
    fakes = _install_fakes(monkeypatch, cls, tmp_path)
    spec = cls.shared_weights_spec
    model = _model(cls, problem_type, _smoke_hps(cls), tmp_path)
    key, payload = _fake_fit(model, fakes)
    net = _net_of(payload)
    for name in BLOCKED_LIBRARIES.get(spec.library, (spec.library,)):
        monkeypatch.setitem(sys.modules, name, None)
    for target, replacement in _block(cls).library_free_patches:
        module_name, attr = target.rsplit(".", 1)
        monkeypatch.setattr(importlib.import_module(module_name), attr, replacement)

    pickle.dumps(model)
    assert model.get_device() == "cpu"
    assert model.get_info()["shared_weights"]["library"] == spec.library
    model.prepare_for_inference()
    _detach(model)
    model._ensure_network()
    assert _net_of(_attached_network(model)) is net
    model._set_device("cpu")
    assert len(fakes.builds) == 1


@pytest.mark.parametrize(("cls", "problem_type"), MODULE_CASES, ids=MODULE_IDS)
def test_autogluon_save_load_reattaches(cls, problem_type, tmp_path, monkeypatch):
    fakes = _install_fakes(monkeypatch, cls, tmp_path)
    spec = cls.shared_weights_spec
    model = _model(cls, problem_type, _smoke_hps(cls), tmp_path)
    key, payload = _fake_fit(model, fakes)
    net = _net_of(payload)
    model.save(verbose=False)
    loaded = cls.load(model.path, verbose=False)
    if not loaded._network_attached():
        loaded._ensure_network()
    assert _net_of(_attached_network(loaded)) is net
    assert len(fakes.builds) == 1
    for path, kind in spec.device_attrs:
        value = se.get_by_path(loaded.model, path)
        assert (value.type if kind == "torch" else value) == "cpu", path

    model._shared_key = model._key_for_device("cuda")
    model.device = "cuda"
    model.device_train = "cuda"
    if spec.seam == "load_model":
        model.model._tabarena_key = model._shared_key
        model.model.device = "cuda"
    model.save(verbose=False)
    reloaded = cls.load(model.path, verbose=False)
    if not reloaded._network_attached():
        reloaded._ensure_network()
    assert reloaded._shared_key.device == "cpu"
    assert _net_of(_attached_network(reloaded)) is net
    assert len(fakes.builds) == 1


@pytest.mark.parametrize("cls", MIXIN_USERS, ids=CLASS_IDS)
def test_prefetch_weights_resolves_every_declared_checkpoint(cls, tmp_path, monkeypatch):
    spec = cls.shared_weights_spec
    if spec.checkpoint is None:
        pytest.skip(f"{cls.__name__} resolves its checkpoints through the library (covered by the models-marked tests)")
    calls: list[tuple] = []

    def fake_file(repo_id, filename, *, revision=None, subfolder=None, allow_download=True, token=None):
        calls.append(("file", repo_id, filename, allow_download))
        path = tmp_path / filename.replace("/", "_")
        path.write_bytes(b"0")
        return str(path)

    def fake_snapshot(
        repo_id, *, revision=None, allow_patterns=None, required_files=None, allow_download=True, token=None
    ):
        calls.append(("snapshot", repo_id, tuple(required_files or ()), allow_download))
        root = tmp_path / "snapshot"
        for name in required_files or ():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(b"0")
        root.mkdir(exist_ok=True)
        return str(root)

    monkeypatch.setattr(hub, "resolve_hf_file", fake_file)
    monkeypatch.setattr(hub, "resolve_hf_snapshot", fake_snapshot)
    paths = cls.prefetch_weights()
    assert paths and all(Path(path).exists() for path in paths)
    assert calls and all(call[-1] is True for call in calls)
    filenames = {call[2] for call in calls if call[0] == "file"}
    for variant in spec.variants():
        checkpoint = spec.checkpoint_for(variant)
        if checkpoint is None or checkpoint.kind != "file" or checkpoint.filename_param is None:
            continue
        for choice in spec.checkpoint_choices.get(checkpoint.filename_param, ()):
            expected = checkpoint._pick(choice, variant)
            if expected is not None:
                assert expected in filenames, f"{variant}: {expected} was not prefetched"


_TABPFN_MANY_CLASS_USERS = [cls for cls in MIXIN_USERS if cls.__name__ in ("RealTabPFNv25Model", "TabPFNv26Model")]


@pytest.mark.parametrize("cls", _TABPFN_MANY_CLASS_USERS, ids=[cls.__name__ for cls in _TABPFN_MANY_CLASS_USERS])
def test_tabpfn_many_class_wrapper_shares_through_the_base_estimator(cls, tmp_path, monkeypatch):
    """With more than ten classes the specs sit on the wrapper's base estimator; the pickle and the reattach follow it."""
    fakes = _install_fakes(monkeypatch, cls, tmp_path)
    model = _model(cls, "multiclass", _smoke_hps(cls), tmp_path)
    model.initialize()
    key, payload = model._acquire_shared_weights(device="cpu")
    model._model_path_arg = "ckpt.bin"
    model.model = _FakeManyClass(SimpleNamespace(model_path=payload, device="cpu"))
    model.model.estimators_ = [_FakeTabPFN(payload, "cpu", owned=False)]
    model.device = model.device_train = "cpu"
    net = _net_of(payload)

    assert model._network_attached()
    assert list(model._shared_modules()) == [net]
    assert model.get_device() == "cpu"
    assert model._get_memory_size() == model._get_pickled_size() + w.tensor_bytes([net])

    state = model.__getstate__()
    assert state["model"].estimator.model_path == "ckpt.bin"
    assert state["model"].estimators_[0].models_ is None
    assert not list(walk_tensors(state)), "the many-class pickle holds tensors"
    assert model.model.estimator.model_path is payload, "the live base estimator lost the specs"
    assert model.model.estimators_[0].models_ == [net]

    loaded = pickle.loads(pickle.dumps(model))
    assert not loaded._network_attached()
    loaded._ensure_network()
    assert loaded.model.estimator.model_path is payload
    assert loaded.model.estimator.device == "cpu"
    assert loaded.model.estimators_[0].models_ == [net]
    assert len(fakes.builds) == 1

    _patch_fake_cuda(monkeypatch)
    cuda_net = _TinyNet()
    cuda_net._fake_device = "cuda"
    cuda_net.shared_guard = True
    cuda_payload = _FakeModelSpecs(cuda_net)
    w.prime(model._key_for_device("cuda"), lambda: cuda_payload)
    model._set_device("cuda")
    assert model.model.estimator.model_path is cuda_payload
    assert model.model.estimators_[0].models_ == [cuda_net]
    assert model.get_device() == "cuda"
    assert not net.to_calls and not cuda_net.to_calls


def test_derived_estimator_classes_are_pickle_addressable():
    models_dir = Path(swm.__file__).parent
    found = []
    for estimators_file in sorted(models_dir.glob("*/_estimators.py")):
        module_name = f"tabarena.models.{estimators_file.parent.name}._estimators"
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for value in vars(module).values():
            if isinstance(value, type) and se.DERIVED_MARKER in value.__dict__:
                found.append(value)
                assert value.__module__ == module_name
                assert getattr(sys.modules[value.__module__], value.__qualname__) is value
                assert pickle.loads(pickle.dumps(value)) is value
    if not found:
        pytest.skip("no derived estimator class importable (its library is not installed)")


# --- real libraries (models marker) ------------------------------------------------------------------


@pytest.mark.models
@pytest.mark.parametrize(("cls", "problem_type"), CASES, ids=CASE_IDS)
def test_real_shared_fit_equals_unshared_fit_pickles_weightless_and_reloads(cls, problem_type, tmp_path):
    from tabarena.utils.synthetic_data import make_synthetic_frames

    info = _REGISTRY[METHOD_BY_CLASS[cls]]
    gpu = info.method_metadata.compute == "gpu"
    if gpu and not torch.cuda.is_available():
        pytest.skip(f"{cls.__name__}: compute='gpu' and no CUDA device is available")
    spec = cls.shared_weights_spec
    hps = _smoke_hps(cls, real=True)
    X, y, X_test = make_synthetic_frames(problem_type, n_rows=64, n_features=4)

    def fit(name: str, share: bool):
        cls.set_class_settings(share_weights=share)
        try:
            model = _model(cls, problem_type, hps, tmp_path, name=name)
            try:
                model.fit(X=X, y=y, num_cpus=2, num_gpus=1 if gpu else 0)
            except ImportError as exc:
                pytest.skip(f"{cls.__name__}: optional dependency not installed ({exc})")
            return model
        finally:
            cls.set_class_settings(share_weights=True)

    shared = fit("shared", share=True)
    plain = fit("plain", share=False)
    assert shared._shared_key is not None and plain._shared_key is None
    np.testing.assert_allclose(shared.predict_proba(X_test), plain.predict_proba(X_test), rtol=1e-5, atol=1e-6)
    if spec.mode == "module":
        assert not list(walk_tensors(shared.__getstate__())), "the shared child's pickle holds tensors"
        network = _attached_network(shared)
        reloaded = pickle.loads(pickle.dumps(shared))
        if not reloaded._network_attached():
            reloaded._ensure_network()
        assert se.payload_modules(_attached_network(reloaded))[0] is se.payload_modules(network)[0]
        np.testing.assert_allclose(reloaded.predict_proba(X_test), shared.predict_proba(X_test))


@pytest.mark.models
@pytest.mark.parametrize("wrapper", sorted(REPLICA_DRIFT_GUARDS))
def test_library_replica_drift_guards(wrapper):
    REPLICA_DRIFT_GUARDS[wrapper]()
