"""Convention checks for every registered wrapper that declares ``shared_weights``.

The wrappers are discovered from the model registry, so a new declaring wrapper is covered without
registering anything here. The default tests read the declaration alone: well-formed, the loader
reference resolves when the library is installed, the keyed inputs are the loader's parameters (a
typo in ``key`` would silently key every fit on ``None``), the named disablers switch sharing off,
and the class carries the ``share_weights`` class setting. The ``models``-marked test fits the real
library: two fits share one network, the pickle is weightless and a reload predicts the same, and a
fit with sharing switched off predicts the same as a shared one.
"""

from __future__ import annotations

import importlib.util
import inspect
import pickle
from typing import Any

import numpy as np
import pytest

from tabarena.models import _weights as registry
from tabarena.utils.synthetic_data import make_synthetic_frames

from .smoke_configs import registry_or_fail, smoke_for

_REGISTRY = registry_or_fail()
_DECLARING = sorted(
    method for method, info in _REGISTRY.items() if getattr(info.model_cls, "shared_weights", None) is not None
)


def _sw():
    from autogluon.core.models.abstract import shared_weights

    return shared_weights


def _cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available()


def _loader_callables(spec) -> list[tuple[str, Any, str]]:
    """``(reference, owner, kind)`` per loader, importing the library; skips the test when it is absent."""
    sw = _sw()
    out = []
    for text in spec.loaders():
        ref = sw.LoaderRef.parse(text)
        try:
            owner, name = ref.owner_and_name()
        except ImportError as err:
            pytest.skip(f"{text}: library not installed ({err})")
        assert hasattr(owner, name), f"{text}: {owner!r} has no attribute {name!r}"
        out.append((text, owner, ref.kind()))
    return out


def test_at_least_one_wrapper_declares_sharing():
    assert _DECLARING, "no registered model declares shared_weights"


@pytest.mark.parametrize("method", _DECLARING, ids=str)
def test_declaration_is_well_formed(method: str):
    sw = _sw()
    cls = _REGISTRY[method].model_cls
    spec = cls.shared_weights
    sw.validate_declaration(spec, cls)
    assert spec.loaders(), f"{method}: empty loader"
    for text in spec.loaders():
        ref = sw.LoaderRef.parse(text)
        assert ref.library and ref.qualname
    assert all(isinstance(name, str) and name for name in spec.key), f"{method}: key names must be strings"
    for rule in spec.disabled_by:
        assert isinstance(rule, str) or callable(rule)
    if spec.copy_per_fit:
        assert cls._class_tags().get("pickles_pretrained_weights", True), (
            f"{method}: a fine-tuning model pickles its own copy of the network"
        )


@pytest.mark.parametrize("method", _DECLARING, ids=str)
def test_loader_resolves_and_the_key_names_its_inputs(method: str):
    """The loader is a real attribute of the library and every ``key`` entry is one of its inputs."""
    sw = _sw()
    spec = _REGISTRY[method].model_cls.shared_weights
    for text, owner, kind in _loader_callables(spec):
        ref = sw.LoaderRef.parse(text)
        _, name = ref.owner_and_name()
        if kind == "method":
            assert inspect.isclass(owner), f"{text}: a method loader names a class"
            parameters = set(inspect.signature(owner.__init__).parameters)
        else:
            raw = inspect.getattr_static(owner, name)
            function = raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else getattr(owner, name)
            signature = inspect.signature(function)
            parameters = {
                p for p, parameter in signature.parameters.items() if parameter.kind is not parameter.VAR_KEYWORD
            }
            if isinstance(raw, classmethod) and parameters:
                parameters.discard(next(iter(signature.parameters)))
        missing = [k for k in spec.key if k not in parameters]
        assert not missing, f"{text}: key names {missing} are not inputs of the loader ({sorted(parameters)})"


@pytest.mark.parametrize("method", _DECLARING, ids=str)
def test_named_disablers_switch_sharing_off(method: str):
    spec = _REGISTRY[method].model_cls.shared_weights
    for example in spec.disabled_examples():
        assert not spec.allows(example), f"{method}: {example} should disable sharing"
    assert spec.allows({}), f"{method}: the default configuration shares"


@pytest.mark.parametrize("method", _DECLARING, ids=str)
def test_class_settings_carry_the_share_weights_switch(method: str):
    cls = _REGISTRY[method].model_cls
    assert cls.class_settings_cls is not None
    assert issubclass(cls.class_settings_cls, registry.SharedWeightsClassSettings)
    assert cls.get_class_settings().share_weights is True


def _problem_type(cls) -> str:
    supported = getattr(cls, "_supported_problem_types", None) or ["binary", "multiclass", "regression"]
    return "binary" if "binary" in supported else supported[0]


def _fit(cls, tmp_path, name: str, X, y, hyperparameters: dict, num_gpus: int):
    model = cls(
        path=str(tmp_path / name) + "/",
        name=name,
        problem_type=_problem_type(cls),
        hyperparameters=dict(hyperparameters),
    )
    model.fit(X=X, y=y, num_cpus=1, num_gpus=num_gpus)
    return model


def _predict(model, X) -> np.ndarray:
    return model.predict_proba(X) if model.problem_type != "regression" else model.predict(X)


@pytest.mark.models
@pytest.mark.parametrize("method", _DECLARING, ids=str)
def test_fit_shares_one_network_and_pickles_without_it(method: str, tmp_path):
    """Two fits build one network; the pickle is weightless and reloads to the same predictions; an unshared fit agrees."""
    info = _REGISTRY[method]
    cls = info.model_cls
    if info.method_metadata.compute == "gpu" and not _cuda_available():
        pytest.skip(f"{method}: requires a GPU and no CUDA device is available")
    if info.superseded:
        pytest.skip(f"{method}: superseded")
    sw = _sw()
    spec = cls.shared_weights
    _loader_callables(spec)  # skips when the library is absent
    num_gpus = 1 if info.method_metadata.compute == "gpu" else 0
    hps = dict(smoke_for(method).hyperparameters)
    X, y, X_predict = make_synthetic_frames(_problem_type(cls), n_rows=64, n_features=5, n_categorical=1, seed=0)

    registry.release()
    registry.reset_stats()
    try:
        first = _fit(cls, tmp_path, "first", X, y, hps, num_gpus)
    except ImportError as err:
        pytest.skip(f"{method}: optional dependency not installed ({err})")
    assert first._shared_state is not None, f"{method}: the fit took nothing from the registry"
    stats = registry.report()["stats"]
    assert stats["misses"] >= 1
    second = _fit(cls, tmp_path, "second", X, y, hps, num_gpus)
    assert registry.report()["stats"]["hits"] > stats["hits"], f"{method}: the second fit did not hit the registry"
    assert second.get_info()["shared_weights"]["present_before_fit"] is True
    expected = _predict(first, X_predict)
    np.testing.assert_allclose(_predict(second, X_predict), expected, rtol=1e-4, atol=1e-5)

    blob = pickle.dumps(first)
    if spec.copy_per_fit:
        assert sw.owns_network(first)
    else:
        loaded = pickle.loads(blob)
        assert not sw.attached(loaded), f"{method}: the pickle carries the shared network"
        loaded.prepare_for_inference()
        np.testing.assert_allclose(_predict(loaded, X_predict), expected, rtol=1e-4, atol=1e-5)
        assert all(a is b for a, b in zip(sw.components(loaded), sw.components(first), strict=True)), (
            "the reload references the registry's network"
        )
        registry.release()
        rebuilt = pickle.loads(blob)
        rebuilt.prepare_for_inference()
        np.testing.assert_allclose(_predict(rebuilt, X_predict), expected, rtol=1e-4, atol=1e-5)

    unshared = _fit(
        cls, tmp_path, "unshared", X, y, {**hps, "ag_args_fit": {"share_pretrained_weights": False}}, num_gpus
    )
    assert unshared._shared_state is None, f"{method}: share_pretrained_weights=False still shared"
    assert unshared.get_info()["shared_weights"] is None
    np.testing.assert_allclose(_predict(unshared, X_predict), expected, rtol=1e-4, atol=1e-5)
