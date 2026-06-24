from __future__ import annotations

import types

import pytest

from tabarena.models import _registry
from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models._registry import (
    discover_models,
    get_model_registry,
    register_model_info,
)


class _DummyModel:
    pass


def _make_info(method: str, suite: str | None = None) -> ModelInfo:
    return ModelInfo(
        model_cls=_DummyModel,
        search_space=lambda: None,
        method_metadata=MethodMetadata(method=method, suite=suite),
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
    """Stub the package walk + info-import calls inside `discover_models`.

    Tests populate `state["submodules"]` with `(name, is_pkg)` tuples and
    `state["info_modules"]` with `name -> module-or-exception` to control
    exactly what the discovery walk sees.
    """
    state = {"submodules": [], "info_modules": {}}

    def fake_iter_modules(_path):
        for name, is_pkg in state["submodules"]:
            yield (None, name, is_pkg)

    def fake_import_module(name):
        prefix, suffix = "tabarena.models.", ".info"
        assert name.startswith(prefix) and name.endswith(suffix), name
        key = name[len(prefix) : -len(suffix)]
        result = state["info_modules"].get(key)
        if isinstance(result, Exception):
            raise result
        if result is None:
            raise ImportError(f"no fake info module registered for {name!r}")
        return result

    monkeypatch.setattr(
        _registry,
        "pkgutil",
        types.SimpleNamespace(iter_modules=fake_iter_modules),
    )
    monkeypatch.setattr(
        _registry,
        "importlib",
        types.SimpleNamespace(import_module=fake_import_module),
    )
    return state


def test_discover_models_collects_modelinfo_keyed_by_method(patched_discovery):
    info_a = _make_info("MethodA")
    info_b = _make_info("MethodB")
    patched_discovery["submodules"] = [("a", True), ("b", True)]
    patched_discovery["info_modules"] = {
        "a": _info_module(a_info=info_a),
        "b": _info_module(b_info=info_b),
    }

    assert discover_models() == {"MethodA": info_a, "MethodB": info_b}


def test_discover_models_caches_result(patched_discovery):
    patched_discovery["submodules"] = [("a", True)]
    patched_discovery["info_modules"] = {"a": _info_module(a_info=_make_info("A"))}

    first = discover_models()
    # Mutate the underlying state — a fresh walk would now produce a different result.
    patched_discovery["submodules"] = []
    patched_discovery["info_modules"] = {}
    second = discover_models()

    assert first is second


def test_get_model_registry_returns_cached_registry(patched_discovery):
    patched_discovery["submodules"] = [("a", True)]
    patched_discovery["info_modules"] = {"a": _info_module(a_info=_make_info("A"))}

    first = get_model_registry()
    second = get_model_registry()

    assert first is second
    assert "A" in first


def test_discover_models_skips_modules_and_underscore_packages(patched_discovery):
    info = _make_info("Real")
    patched_discovery["submodules"] = [
        ("real", True),
        ("_private", True),  # leading underscore -> skipped
        ("not_a_pkg", False),  # module, not package -> skipped
    ]
    patched_discovery["info_modules"] = {
        "real": _info_module(real_info=info),
        "_private": _info_module(should_not_appear=_make_info("Private")),
        "not_a_pkg": _info_module(also_skipped=_make_info("NotPkg")),
    }

    assert discover_models() == {"Real": info}


def test_discover_models_skips_packages_without_info(patched_discovery, caplog):
    info = _make_info("Real")
    patched_discovery["submodules"] = [("real", True), ("legacy", True)]
    patched_discovery["info_modules"] = {
        "real": _info_module(real_info=info),
        "legacy": ImportError("no info module"),
    }

    with caplog.at_level("WARNING", logger="tabarena.models._registry"):
        assert discover_models() == {"Real": info}

    # Failure is logged rather than silently swallowed (the silent-skip
    # behaviour previously masked a real CatBoost discovery regression).
    assert any("legacy" in record.message and "no info module" in record.message for record in caplog.records), (
        f"expected a warning mentioning 'legacy' and the import error, got: {[r.message for r in caplog.records]}"
    )


def test_discover_models_ignores_underscore_and_non_modelinfo_attrs(patched_discovery):
    real_info = _make_info("Real")
    patched_discovery["submodules"] = [("real", True)]
    patched_discovery["info_modules"] = {
        "real": _info_module(
            real_info=real_info,
            _hidden_info=_make_info("Hidden"),  # underscore attr -> skipped
            some_constant=42,  # non-ModelInfo -> skipped
            helper_fn=lambda: None,  # non-ModelInfo -> skipped
        ),
    }

    assert discover_models() == {"Real": real_info}


def test_discover_models_raises_on_duplicate_method_key(patched_discovery):
    patched_discovery["submodules"] = [("a", True), ("b", True)]
    patched_discovery["info_modules"] = {
        "a": _info_module(a_info=_make_info("SameMethod")),
        "b": _info_module(b_info=_make_info("SameMethod")),
    }

    with pytest.raises(RuntimeError, match="Duplicate ModelInfo key"):
        discover_models()


def test_register_model_info_adds_new_entry(patched_discovery):
    info = _make_info("NewMethod", suite="art-1")

    register_model_info(info)

    assert get_model_registry()["NewMethod"] is info


def test_register_model_info_is_idempotent_for_same_object(patched_discovery):
    info = _make_info("SameObject")

    register_model_info(info)
    register_model_info(info)

    assert get_model_registry()["SameObject"] is info


def test_register_model_info_disambiguates_duplicate_with_suite(patched_discovery):
    core_info = _make_info("Linear", suite="tabarena-core")
    ext_info = _make_info("Linear", suite="extension-rerun")
    patched_discovery["submodules"] = [("core", True)]
    patched_discovery["info_modules"] = {"core": _info_module(core=core_info)}

    register_model_info(ext_info)
    registry = get_model_registry()

    assert registry["Linear"] is core_info
    assert registry["Linear@extension-rerun"] is ext_info


def test_register_model_info_raises_on_conflicting_composite_key(patched_discovery):
    core_info = _make_info("Linear", suite="tabarena-core")
    ext_info_v1 = _make_info("Linear", suite="rerun")
    ext_info_v2 = _make_info("Linear", suite="rerun")
    patched_discovery["submodules"] = [("core", True)]
    patched_discovery["info_modules"] = {"core": _info_module(core=core_info)}

    register_model_info(ext_info_v1)
    with pytest.raises(RuntimeError, match="Duplicate ModelInfo composite key"):
        register_model_info(ext_info_v2)
