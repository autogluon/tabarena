from __future__ import annotations

import types
from importlib.machinery import ModuleSpec

import pytest

from tabarena.models import _registry
from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models._registry import (
    assert_autogluon_resolves,
    discover_models,
    get_model_registry,
    register_model_info,
)

# Info modules that import only from ``autogluon.tabular.models``, so they resolve on the base
# install (no ``[benchmark]`` extra) and lock the "registry is populated" precondition.
_CORE_MODELS = frozenset(
    {
        "CatBoost",
        "ExplainableBM",
        "ExtraTrees",
        "KNeighbors",
        "LightGBM",
        "LinearModel",
        "NeuralNetFastAI",
        "NeuralNetTorch",
        "RandomForest",
        "XGBoost",
    }
)


# The models TabArena defines whose bag gives each fold model a seed of its own: the models trained
# on the task's data, and Mitra-v2, which is fine-tuned per fold and not refit (its bag predicts).
# Every other model TabArena defines is a pretrained (foundation) model, whose bag keeps one seed so
# that the fold models are the refit that predicts.
_VARY_SEED_ACROSS_FOLDS_MODELS = frozenset(
    {
        "CTBoost",
        "ChimeraBoost",
        "KNeighbors",
        "MLP-A",
        "Mitra-v2",
        "ModernNCA",
        "ModernNCA_GPU",
        "PerpetualBooster",
        "PrismBoost",
        "RealMLP",
        "RealMLP_GPU",
        "TabM",
        "TabM-A",
        "TabPack",
        "aplr",
        "xRFM_GPU",
    }
)


def _regular_spec(name: str = "autogluon.tabular") -> ModuleSpec:
    return ModuleSpec(name, None, origin="/fake/autogluon/tabular/__init__.py")


def _namespace_spec() -> ModuleSpec:
    spec = ModuleSpec("autogluon.tabular", None, is_package=True)
    spec.submodule_search_locations = ["/somewhere/autogluon/tabular"]
    return spec


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
    exactly what the discovery walk sees. `find_spec` is stubbed to a regular
    package so the walk tests do not depend on the invoking interpreter's cwd.
    """
    state = {"submodules": [], "info_modules": {}}
    monkeypatch.setattr(_registry, "find_spec", _regular_spec)

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


def test_discover_models_raises_when_every_package_is_skipped(patched_discovery, caplog):
    patched_discovery["submodules"] = [("a", True), ("b", True)]
    patched_discovery["info_modules"] = {"a": ImportError("x"), "b": ImportError("y")}

    with (
        caplog.at_level("WARNING", logger="tabarena.models._registry"),
        pytest.raises(RuntimeError, match="registry is empty") as excinfo,
    ):
        discover_models()

    assert "a, b" in str(excinfo.value)
    assert _registry._REGISTRY is None  # nothing is cached on failure
    # An empty walk (no packages at all) is not an error.
    patched_discovery["submodules"] = []
    assert discover_models() == {}


def test_assert_autogluon_resolves_rejects_a_namespace_package(monkeypatch):
    monkeypatch.setattr(_registry, "find_spec", lambda name: _namespace_spec())
    with pytest.raises(RuntimeError, match="namespace package") as excinfo:
        assert_autogluon_resolves()
    assert "python -P" in str(excinfo.value)


@pytest.mark.parametrize(
    "fake_find_spec",
    [
        _regular_spec,
        lambda name: None,
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError("No module named 'autogluon'")),
        lambda name: (_ for _ in ()).throw(ValueError(f"{name}.__spec__ is None")),
    ],
    ids=["regular", "child-missing", "not-installed", "spec-less"],
)
def test_assert_autogluon_resolves_accepts_everything_but_shadowing(monkeypatch, fake_find_spec):
    monkeypatch.setattr(_registry, "find_spec", fake_find_spec)
    assert_autogluon_resolves()


def test_real_registry_discovers_core_models():
    missing = _CORE_MODELS - set(discover_models())
    assert not missing, f"core models missing from the registry: {sorted(missing)}"


def test_tabarena_models_declare_vary_seed_across_folds():
    """Every model class TabArena defines chooses whether a bag varies the seed across its folds, so the
    choice is the model's and a new model has to make it. AutoGluon's own classes declare it there.
    """
    checked = []
    for name, info in discover_models().items():
        model_cls = info.model_cls
        if not model_cls.__module__.startswith("tabarena."):
            continue
        vary_seed_across_folds = model_cls._get_default_ag_args_ensemble().get("vary_seed_across_folds")
        assert isinstance(vary_seed_across_folds, bool), (
            f"{name} ({model_cls.__name__}) must declare `vary_seed_across_folds` in "
            f"`_default_ag_args_ensemble_extra`: True for a model trained (or fine-tuned without a refit) "
            f"on the data, False for a pretrained model."
        )
        assert vary_seed_across_folds is (name in _VARY_SEED_ACROSS_FOLDS_MODELS), name
        checked.append(name)
    assert "KNeighbors" in checked  # resolves on the base install


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
