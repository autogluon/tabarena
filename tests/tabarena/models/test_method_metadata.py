from __future__ import annotations

from dataclasses import fields

import yaml

from tabarena.models._method_metadata import MethodMetadata


def test_location_args_are_first_three_fields():
    # method / suite / artifact_dir are the location-determining identity args and live together
    # at the front of the init signature.
    assert [f.name for f in fields(MethodMetadata)][:3] == ["method", "suite", "artifact_dir"]


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
    method_dir.mkdir(parents=True, exist_ok=True)
    info = MethodMetadata.config(method=method, suite=suite).to_info_dict()
    with open(method_dir / "metadata.yaml", "w") as f:
        yaml.dump(info, f)


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
