from __future__ import annotations

from dataclasses import fields

import pandas as pd
import pytest
import yaml

from tabarena.models._method_metadata import MethodMetadata


def test_location_args_are_first_fields():
    # The location-determining args live together at the front of the init signature.
    assert [f.name for f in fields(MethodMetadata)][:4] == ["method", "suite", "artifact_dir", "cache_root"]


def test_default_layout_uses_suite_and_method_segments():
    # Without an artifact_dir override the path is <cache>/artifacts/<suite>/methods/<method>.
    # Assert the trailing segments rather than the absolute root so the test is independent of
    # wherever the global TabArena cache happens to resolve.
    mm = MethodMetadata.config(method="Foo", suite="tabarena-2026-05-13")
    assert mm.artifact_dir is None
    assert mm.path.parts[-4:] == ("artifacts", "tabarena-2026-05-13", "methods", "Foo")


def test_artifact_dir_is_returned_as_the_path_verbatim(tmp_path):
    # When artifact_dir is set it *is* the artifact root; suite/method do not contribute.
    mm = MethodMetadata.config(method="Foo", suite="some-suite", artifact_dir=tmp_path)
    assert mm.path == tmp_path
    assert mm.path_results == tmp_path / "results"
    assert mm.path_metadata == tmp_path / "metadata.yaml"


def test_artifact_dir_excluded_from_info_dict(tmp_path):
    mm = MethodMetadata.config(method="Foo", artifact_dir=tmp_path)
    assert "artifact_dir" not in mm.to_info_dict()


def _write_committed_method(method_dir, *, method="Foo", suite="s1"):
    # Written through `to_yaml`, the same writer production uses, so the committed file this
    # exercises is byte-for-byte what a real cached method has.
    method_dir.mkdir(parents=True, exist_ok=True)
    MethodMetadata.config(method=method, suite=suite).to_yaml(method_dir / "metadata.yaml")


def test_from_yaml_path_resolves_artifacts_next_to_metadata(tmp_path):
    # Loading from an explicit metadata.yaml path points artifact_dir at the dir holding it, so
    # results resolve next to it -- regardless of how the dir is named.
    method_dir = tmp_path / "committed" / "renamed-folder"
    _write_committed_method(method_dir, method="Foo", suite="s1")

    mm = MethodMetadata.from_yaml(path=method_dir / "metadata.yaml")
    assert mm.method == "Foo"
    assert mm.artifact_dir == method_dir
    assert mm.path == method_dir
    assert mm.path_results == method_dir / "results"


def test_from_yaml_path_accepts_dir_or_yaml_file(tmp_path):
    # `path` may be the artifact directory or the metadata.yaml inside it; both are equivalent.
    # Use a dir name containing a dot to confirm dirs aren't misread as files.
    method_dir = tmp_path / "TA-TabPFN-2.6"
    _write_committed_method(method_dir, method="TA-TabPFN-2.6", suite="s2")

    from_dir = MethodMetadata.from_yaml(path=method_dir)
    from_file = MethodMetadata.from_yaml(path=method_dir / "metadata.yaml")

    assert from_dir.artifact_dir == method_dir == from_file.artifact_dir
    assert from_dir.path == method_dir == from_file.path
    assert from_dir.method == from_file.method == "TA-TabPFN-2.6"


def test_cache_root_resolves_layered_path_under_the_override(tmp_path):
    # cache_root keeps the standard <root>/artifacts/<suite>/methods/<method> layout, just under a
    # chosen root instead of the global cache.
    mm = MethodMetadata.config(method="Foo", suite="s1", cache_root=tmp_path)
    assert mm.path_cache_root == tmp_path
    assert mm.path == tmp_path / "artifacts" / "s1" / "methods" / "Foo"
    assert mm.path_results == tmp_path / "artifacts" / "s1" / "methods" / "Foo" / "results"


def test_cache_root_excluded_from_info_dict(tmp_path):
    mm = MethodMetadata.config(method="Foo", cache_root=tmp_path)
    assert "cache_root" not in mm.to_info_dict()


def test_artifact_dir_and_cache_root_are_mutually_exclusive(tmp_path):
    with pytest.raises(AssertionError, match="at most one of"):
        MethodMetadata.config(method="Foo", artifact_dir=tmp_path, cache_root=tmp_path)


def _write_layered_cache(cache_root, *, method, suite):
    # Same reasoning as _write_committed_method: write through the production writer.
    method_dir = cache_root / "artifacts" / suite / "methods" / method
    method_dir.mkdir(parents=True, exist_ok=True)
    MethodMetadata.config(method=method, suite=suite).to_yaml(method_dir / "metadata.yaml")


def test_from_yaml_cache_root_lookup_stamps_the_override(tmp_path):
    cache = tmp_path / "alice" / ".cache" / "tabarena"
    _write_layered_cache(cache, method="Foo", suite="s1")

    mm = MethodMetadata.from_yaml(method="Foo", suite="s1", cache_root=cache)
    assert mm.cache_root == cache
    assert mm.path == cache / "artifacts" / "s1" / "methods" / "Foo"


def test_from_yaml_path_and_cache_root_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        MethodMetadata.from_yaml(path=tmp_path / "metadata.yaml", cache_root=tmp_path)


def test_methods_from_different_cache_roots_stay_independent(tmp_path):
    # The shared-drive scenario: two caches loaded in one process must not interfere -- each
    # instance retains its own cache_root for deferred path resolution.
    alice = tmp_path / "alice"
    bob = tmp_path / "bob"
    _write_layered_cache(alice, method="Foo", suite="s1")
    _write_layered_cache(bob, method="Foo", suite="s1")

    mm_alice = MethodMetadata.from_yaml(method="Foo", suite="s1", cache_root=alice)
    mm_bob = MethodMetadata.from_yaml(method="Foo", suite="s1", cache_root=bob)

    assert mm_alice.path == alice / "artifacts" / "s1" / "methods" / "Foo"
    assert mm_bob.path == bob / "artifacts" / "s1" / "methods" / "Foo"
    assert mm_alice.path != mm_bob.path


def _config_result_df(*, model_type, ag_key, frameworks, name_prefix="Model"):
    """Minimal per-result frame mirroring the columns ``from_raw`` infers a config method from."""
    return pd.DataFrame(
        {
            "method_type": "config",
            "model_type": model_type,
            "ag_key": ag_key,
            "num_gpus": 0,
            "is_bag": True,
            "name_prefix": name_prefix,
            "framework": frameworks,
        }
    )


@pytest.mark.parametrize(
    ("model_type", "ag_key"),
    [
        ("GBM", "GBM"),  # un-re-keyed: model_type == ag_key (no-op regression guard)
        ("PREFIX_GBM", "GBM"),  # re-keyed family on the same model-class backbone
    ],
)
def test_from_raw_config_keys_off_model_type_not_ag_key(model_type, ag_key):
    """``model_key`` / ``config_type`` track the per-result ``model_type`` (the simulation /
    comparison family key the repo groups by), while ``ag_key`` (the model-class key) is preserved
    as-is. They may legitimately differ -- e.g. a re-keyed family on the same backbone -- and
    config_type must follow ``model_type``, not ``ag_key``.
    """
    df = _config_result_df(
        model_type=model_type,
        ag_key=ag_key,
        frameworks=["Model_c1_BAG_L1", "Model_c2_BAG_L1"],
    )
    mm = MethodMetadata._from_raw_config(result_df=df)
    assert mm.model_key == model_type
    assert mm.config_type == model_type
    assert mm.ag_key == ag_key


def test_from_raw_config_artifact_dir_pins_artifact_location(tmp_path):
    """``artifact_dir`` threads through ``from_raw`` so an inferred method's artifacts resolve
    directly under it (instead of the ``{suite}/methods/{method}`` layout under the cache root).
    """
    df = _config_result_df(
        model_type="GBM",
        ag_key="GBM",
        frameworks=["Model_c1_BAG_L1", "Model_c2_BAG_L1"],
    )
    mm = MethodMetadata._from_raw_config(result_df=df, method="M", suite="s", artifact_dir=tmp_path)
    assert mm.artifact_dir == tmp_path
    assert mm.path == tmp_path
    assert mm.path_results == tmp_path / "results"
    assert mm.path_processed == tmp_path / "processed"


# -- method_class / tags ----------------------------------------------------------------------


def test_system_constructor_sets_class_and_baseline_result_shape():
    """``MethodMetadata.system`` marks the entrant class without changing the result shape:
    a system's results are recorded as a baseline, so ``method_type`` stays ``"baseline"``.
    """
    mm = MethodMetadata.system(method="TabFM+", suite="s", tags=("with-llm",))
    assert mm.method_class == "system"
    assert mm.method_type == "baseline"
    assert mm.is_system
    assert mm.uses_llm
    assert not mm.is_closed_api


def test_method_class_defaults_to_model():
    # Every pre-existing method (and every metadata.yaml written before the field existed)
    # reads back as a model.
    assert MethodMetadata(method="M", suite="s").method_class == "model"
    assert MethodMetadata(method="M", suite="s").tags == ()
    assert not MethodMetadata(method="M", suite="s").is_system


def test_tags_are_sorted_and_deduped():
    # Two metadata objects listing the same tags in a different order must serialize identically.
    mm = MethodMetadata.system(method="M", suite="s", tags=("with-llm", "closed-source-api", "with-llm"))
    assert mm.tags == ("closed-source-api", "with-llm")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tags": ("gpu-only",)}, "Unknown tag"),
        ({"method_class": "pipeline"}, "Unknown method_class"),
        ({"method_type": "config", "method_class": "system"}, "incompatible"),
    ],
)
def test_invalid_method_class_or_tags_are_rejected(kwargs, match):
    with pytest.raises(AssertionError, match=match):
        MethodMetadata(method="M", suite="s", **kwargs)


def test_info_dict_keeps_tags_hashable_for_dataframes():
    """``to_info_dict`` feeds ``MethodMetadataCollection.info()`` straight into a DataFrame, where
    ``website_format.strict_merge`` runs ``drop_duplicates`` / ``.eq`` over the cells. A list
    would be unhashable there, so tags must stay a tuple.
    """
    info = MethodMetadata.system(method="M", suite="s", tags=("with-llm",)).to_info_dict()
    assert info["tags"] == ("with-llm",)
    assert isinstance(info["tags"], tuple)
    pd.DataFrame([info]).drop_duplicates(["method"])  # would raise on a list cell


def test_yaml_roundtrip_of_a_tagged_system():
    # yaml.safe_dump raises on a tuple, so the YAML writers flatten to a list; loading it back
    # re-normalizes to a tuple.
    mm = MethodMetadata.system(method="M", suite="s", tags=("closed-source-api", "with-llm"))
    assert isinstance(mm._to_yaml_dict()["tags"], list)
    loaded = MethodMetadata(**yaml.safe_load(mm.to_yaml_fileobj()))
    assert loaded.method_class == "system"
    assert loaded.tags == ("closed-source-api", "with-llm")


def test_legacy_yaml_without_the_new_fields_still_loads(tmp_path):
    # metadata.yaml written before method_class/tags existed picks up the defaults.
    path = tmp_path / "metadata.yaml"
    path.write_text("method: Old\nsuite: s\nmethod_type: baseline\n")
    loaded = MethodMetadata.from_yaml(path=path)
    assert loaded.method_class == "model"
    assert loaded.tags == ()
