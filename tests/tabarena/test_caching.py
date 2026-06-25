from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from tabarena.caching import CacheConfig


@pytest.fixture(autouse=True)
def _isolate_cache_state():
    """Snapshot + restore the global cache state each test mutates.

    ``apply()`` writes process-global state (``HF_HOME`` env vars, the OpenML root cache, the
    TabArena cache-root holder), so we snapshot all of it up front and restore it afterwards to
    keep tests independent.
    """
    import openml

    from tabarena.loaders import set_tabarena_cache_root

    hf_keys = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")
    saved_env = {k: os.environ.get(k) for k in hf_keys}
    saved_openml_root = openml.config._root_cache_directory
    constants_present = "huggingface_hub.constants" in sys.modules
    saved_constants = sys.modules.get("huggingface_hub.constants")
    try:
        yield
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        openml.config.set_root_cache_directory(str(saved_openml_root))
        set_tabarena_cache_root(None)
        if constants_present:
            sys.modules["huggingface_hub.constants"] = saved_constants
        else:
            sys.modules.pop("huggingface_hub.constants", None)


def test_apply_sets_openml_root(tmp_path):
    import openml

    CacheConfig(openml=tmp_path / "openml").apply()
    assert Path(openml.config._root_cache_directory) == tmp_path / "openml"


def test_apply_sets_tabarena_root(tmp_path):
    from tabarena.loaders import get_tabarena_cache_root

    CacheConfig(tabarena=tmp_path / "tab").apply()
    assert get_tabarena_cache_root() == tmp_path / "tab"


def test_apply_sets_hf_home_and_clears_more_specific_vars(tmp_path):
    os.environ["HF_HUB_CACHE"] = "/stale/hub"  # a stale value must not shadow the new HF_HOME
    CacheConfig(huggingface=tmp_path / "hf").apply()
    assert os.environ["HF_HOME"] == str(tmp_path / "hf")
    assert "HF_HUB_CACHE" not in os.environ
    assert "HUGGINGFACE_HUB_CACHE" not in os.environ


def test_apply_patches_already_imported_hf_constants(tmp_path):
    # When huggingface_hub.constants is already imported its paths are frozen, so apply() must
    # repoint them in place (otherwise the env var alone would be ignored).
    fake = types.SimpleNamespace(HF_HOME="/old", HF_HUB_CACHE="/old/hub", HUGGINGFACE_HUB_CACHE="/old/hub")
    sys.modules["huggingface_hub.constants"] = fake
    CacheConfig(huggingface=tmp_path / "hf").apply()
    expected_home = str(tmp_path / "hf")
    expected_hub = str(tmp_path / "hf" / "hub")
    # `expected == fake.ATTR` ordering keeps ruff's SIM300 happy (all-caps attrs read as constants).
    assert expected_home == fake.HF_HOME
    assert expected_hub == fake.HF_HUB_CACHE
    assert expected_hub == fake.HUGGINGFACE_HUB_CACHE


def test_none_fields_are_noops(tmp_path):
    import openml

    before_openml = openml.config._root_cache_directory
    os.environ.pop("HF_HOME", None)
    CacheConfig().apply()
    assert openml.config._root_cache_directory == before_openml
    assert "HF_HOME" not in os.environ


def test_results_field_is_not_applied_globally(tmp_path):
    # `results` is the run_jobs default expname, not global state — apply() must ignore it.
    import openml

    before_openml = openml.config._root_cache_directory
    os.environ.pop("HF_HOME", None)
    CacheConfig(results=tmp_path / "res").apply()
    assert openml.config._root_cache_directory == before_openml
    assert "HF_HOME" not in os.environ


def test_from_root_lays_out_subdirs(tmp_path):
    cfg = CacheConfig.from_root(tmp_path)
    assert cfg.openml == tmp_path / "openml"
    assert cfg.huggingface == tmp_path / "huggingface"
    assert cfg.tabarena == tmp_path / "tabarena"
    assert cfg.results == tmp_path / "results"


def test_from_root_override_pins_field(tmp_path):
    cfg = CacheConfig.from_root(tmp_path, results=None)
    assert cfg.results is None
    assert cfg.openml == tmp_path / "openml"  # the rest still derive from root


def test_scoped_openml_restores_openml_root_but_keeps_other_caches(tmp_path):
    import openml

    from tabarena.loaders import get_tabarena_cache_root

    openml.config.set_root_cache_directory(str(tmp_path / "ambient"))
    cfg = CacheConfig(openml=tmp_path / "x", tabarena=tmp_path / "tab", huggingface=tmp_path / "hf")
    with cfg.scoped_openml():
        # Inside the block everything points at the configured caches.
        assert Path(openml.config._root_cache_directory) == tmp_path / "x"
        assert get_tabarena_cache_root() == tmp_path / "tab"
        assert os.environ["HF_HOME"] == str(tmp_path / "hf")
    # On exit the OpenML root is restored to the ambient location; HF + TabArena stay applied.
    assert Path(openml.config._root_cache_directory) == tmp_path / "ambient"
    assert get_tabarena_cache_root() == tmp_path / "tab"
    assert os.environ["HF_HOME"] == str(tmp_path / "hf")


def test_apply_openml_false_skips_openml(tmp_path):
    import openml

    from tabarena.loaders import get_tabarena_cache_root

    openml.config.set_root_cache_directory(str(tmp_path / "ambient"))
    CacheConfig(openml=tmp_path / "x", tabarena=tmp_path / "tab").apply(openml=False)
    assert Path(openml.config._root_cache_directory) == tmp_path / "ambient"  # untouched
    assert get_tabarena_cache_root() == tmp_path / "tab"  # the rest is still applied


def test_to_dict_from_dict_round_trips_str_config():
    cfg = CacheConfig(
        openml="/o", huggingface="/hf", tabarena="/t", results="/r", apply_on_run=False, scope_openml=True
    )
    assert CacheConfig.from_dict(cfg.to_dict()) == cfg


def test_to_dict_renders_paths_as_str_and_is_value_stable(tmp_path):
    d = CacheConfig.from_root(tmp_path).to_dict()
    assert d["openml"] == str(tmp_path / "openml")
    assert isinstance(d["openml"], str)
    # re-serializing the reconstructed config yields the same dict (value-stable round-trip)
    assert CacheConfig.from_dict(d).to_dict() == d


def test_from_dict_ignores_unknown_keys():
    cfg = CacheConfig.from_dict({"openml": "/o", "unknown": "x"})
    assert cfg.openml == "/o"


def test_apply_is_idempotent(tmp_path):
    import openml

    from tabarena.loaders import get_tabarena_cache_root

    cfg = CacheConfig(openml=tmp_path / "o", huggingface=tmp_path / "hf", tabarena=tmp_path / "t")
    cfg.apply()
    cfg.apply()
    assert Path(openml.config._root_cache_directory) == tmp_path / "o"
    assert os.environ["HF_HOME"] == str(tmp_path / "hf")
    assert get_tabarena_cache_root() == tmp_path / "t"
