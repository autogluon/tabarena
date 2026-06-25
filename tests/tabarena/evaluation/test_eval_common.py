"""Tests for the shared evaluation building blocks in ``tabarena.evaluation._eval_common``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tabarena.evaluation._eval_common import init_aux_metric_env, init_caches, subset_label


def test_subset_label_full_and_sorted():
    assert subset_label([]) == "full"
    assert subset_label(["regression"]) == "regression"
    assert subset_label(["b", "a"]) == "a_b"  # sorted + joined


@pytest.fixture
def _isolate_caches():
    """Snapshot + restore the cache state init_caches mutates (HF env, OpenML root, holder)."""
    import openml

    from tabarena.loaders import set_tabarena_cache_root

    saved_hf = os.environ.get("HF_HOME")
    saved_openml_root = openml.config._root_cache_directory
    try:
        yield
    finally:
        if saved_hf is None:
            os.environ.pop("HF_HOME", None)
        else:
            os.environ["HF_HOME"] = saved_hf
        openml.config.set_root_cache_directory(str(saved_openml_root))
        set_tabarena_cache_root(None)


@pytest.mark.usefixtures("_isolate_caches")
def test_init_caches_delegates_to_cache_config(tmp_path):
    import openml

    from tabarena.loaders import get_tabarena_cache_root

    os.environ.pop("HF_HOME", None)
    init_caches(
        tabarena_cache_path=str(tmp_path / "tab"),
        openml_cache_path=str(tmp_path / "openml"),
        huggingface_cache_path=str(tmp_path / "hf"),
    )
    assert get_tabarena_cache_root() == tmp_path / "tab"
    assert Path(openml.config._root_cache_directory) == tmp_path / "openml"
    assert os.environ["HF_HOME"] == str(tmp_path / "hf")


@pytest.mark.usefixtures("_isolate_caches")
def test_init_caches_two_arg_call_leaves_huggingface_untouched(tmp_path):
    # Backward compatibility: the historical two-positional-arg call must still work and not
    # introduce an HF_HOME (huggingface_cache_path defaults to None).
    os.environ.pop("HF_HOME", None)
    init_caches(str(tmp_path / "tab"), str(tmp_path / "openml"))
    assert "HF_HOME" not in os.environ


def test_init_aux_metric_env_sets_and_clears():
    from tabarena.utils.aux_metric import AUX_METRIC_ENV_VAR

    original = os.environ.get(AUX_METRIC_ENV_VAR)
    try:
        init_aux_metric_env({"binary": "balanced_accuracy", "regression": "r2"})
        assert AUX_METRIC_ENV_VAR in os.environ
        assert "balanced_accuracy" in os.environ[AUX_METRIC_ENV_VAR]

        init_aux_metric_env(None)
        assert AUX_METRIC_ENV_VAR not in os.environ
    finally:
        if original is None:
            os.environ.pop(AUX_METRIC_ENV_VAR, None)
        else:
            os.environ[AUX_METRIC_ENV_VAR] = original
