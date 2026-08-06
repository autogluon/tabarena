from __future__ import annotations

import types

import pytest

from tabarena.models._method_metadata import MethodMetadata
from tabarena.systems import _registry
from tabarena.systems._registry import discover_systems, get_system_registry
from tabarena.systems._system_info import SystemInfo


class _DummySystem:
    pass


def _make_info(method: str, **metadata_kwargs) -> SystemInfo:
    return SystemInfo(
        system_cls=_DummySystem,
        config_generator=object(),
        method_metadata=MethodMetadata.system(method=method, **metadata_kwargs),
    )


def _info_module(**attrs) -> types.ModuleType:
    mod = types.ModuleType("fake_info")
    for name, value in attrs.items():
        setattr(mod, name, value)
    return mod


@pytest.fixture
def fresh_registry(monkeypatch):
    """Reset the cached registry so each test exercises a fresh discovery."""
    monkeypatch.setattr(_registry, "_REGISTRY", None)


@pytest.fixture
def patched_discovery(monkeypatch, fresh_registry):
    """Stub the package walk + info-import calls inside `discover_systems`.

    Tests populate `state["submodules"]` with `(name, is_pkg)` tuples and
    `state["info_modules"]` with `name -> module-or-exception` to control exactly what the
    discovery walk sees. Mirrors the models registry's fixture.
    """
    state = {"submodules": [], "info_modules": {}}

    def fake_iter_modules(_path):
        for name, is_pkg in state["submodules"]:
            yield (None, name, is_pkg)

    def fake_import_module(name):
        prefix, suffix = "tabarena.systems.", ".info"
        assert name.startswith(prefix) and name.endswith(suffix), name
        key = name[len(prefix) : -len(suffix)]
        result = state["info_modules"].get(key)
        if isinstance(result, Exception):
            raise result
        if result is None:
            raise ImportError(f"no fake info module registered for {name!r}")
        return result

    monkeypatch.setattr(_registry, "pkgutil", types.SimpleNamespace(iter_modules=fake_iter_modules))
    monkeypatch.setattr(_registry, "importlib", types.SimpleNamespace(import_module=fake_import_module))
    return state


def test_discover_systems_collects_systeminfo_keyed_by_method(patched_discovery):
    info_a = _make_info("SystemA")
    info_b = _make_info("SystemB")
    patched_discovery["submodules"] = [("a", True), ("b", True)]
    patched_discovery["info_modules"] = {
        "a": _info_module(a_info=info_a),
        "b": _info_module(b_info=info_b),
    }

    assert discover_systems() == {"SystemA": info_a, "SystemB": info_b}


def test_discover_systems_skips_packages_whose_info_fails_to_import(patched_discovery, caplog):
    """One system with a broken optional dependency must not take the registry down."""
    info_ok = _make_info("Works")
    patched_discovery["submodules"] = [("broken", True), ("ok", True)]
    patched_discovery["info_modules"] = {
        "broken": ImportError("no module named 'someautoml'"),
        "ok": _info_module(ok_info=info_ok),
    }

    assert discover_systems() == {"Works": info_ok}
    assert "tabarena.systems.broken" in caplog.text


def test_discover_systems_rejects_duplicate_method_keys(patched_discovery):
    patched_discovery["submodules"] = [("a", True), ("b", True)]
    patched_discovery["info_modules"] = {
        "a": _info_module(a_info=_make_info("Same")),
        "b": _info_module(b_info=_make_info("Same")),
    }

    with pytest.raises(RuntimeError, match="Duplicate SystemInfo key"):
        discover_systems()


def test_system_info_requires_a_system_method_class():
    """A SystemInfo built on model metadata would classify wrong everywhere downstream."""
    with pytest.raises(AssertionError, match="requires method_class='system'"):
        SystemInfo(
            system_cls=_DummySystem,
            config_generator=object(),
            method_metadata=MethodMetadata.baseline(method="NotASystem"),
        )


def test_registry_is_cached(patched_discovery):
    patched_discovery["submodules"] = [("a", True)]
    patched_discovery["info_modules"] = {"a": _info_module(a_info=_make_info("A"))}

    assert get_system_registry() is discover_systems()


# -- the real registry ------------------------------------------------------------------------


def test_real_registry_discovers_the_shipped_systems():
    # Keyed on each `SystemInfo`'s method, which for AutoGluon is whichever dated run is the
    # latest, so match the wrapper class instead of pinning that run's id.
    registry = discover_systems()
    assert "TabFM+" in registry
    assert any(info.system_cls.__name__ == "AutoGluonSystemModel" for info in registry.values())


def test_every_discovered_system_declares_method_class_system():
    for method, info in discover_systems().items():
        assert info.method_metadata.is_system, method
        # A system's results are recorded as a baseline, never as a tunable config.
        assert info.method_metadata.method_type != "config", method
