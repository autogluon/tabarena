"""Tests for the standardized foundation-model weight prefetch dispatcher.

The actual downloads are never invoked: model resolution is monkeypatched so each fake model
carries a ``prefetch_weights`` callable (as real models declare via ``ModelInfo.prefetch_weights``).
These exercise the dispatch / dedup / skip / error-handling logic only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import tabarena.models.prefetch as pf


def _fake_info(method: str, prefetch=None) -> SimpleNamespace:
    return SimpleNamespace(method_metadata=SimpleNamespace(method=method), prefetch_weights=prefetch)


def _patch_registry(monkeypatch, infos: dict[str, SimpleNamespace]) -> None:
    """Make ``get_model_info_from_name`` resolve from ``infos`` (KeyError -> ValueError)."""

    def fake_get(name: str):
        if name not in infos:
            raise ValueError(f"unknown {name}")
        return infos[name]

    monkeypatch.setattr("tabarena.models.utils.get_model_info_from_name", fake_get)


def test_declared_prefetcher_called_once(monkeypatch):
    calls: list[str] = []
    _patch_registry(monkeypatch, {"M": _fake_info("M_GPU", lambda: calls.append("M"))})
    pf.prefetch_weights(["M"])
    assert calls == ["M"]


def test_shared_prefetcher_deduplicated(monkeypatch):
    calls: list[str] = []
    shared = lambda: calls.append("tabpfn")
    _patch_registry(
        monkeypatch,
        {"TabPFN-3": _fake_info("TabPFN-3", shared), "TabPFN-2.6": _fake_info("TabPFN-v2.6", shared)},
    )
    pf.prefetch_weights(["TabPFN-3", "TabPFN-2.6"])
    assert calls == ["tabpfn"]  # warmed once despite two variants


def test_non_foundation_model_skipped(monkeypatch):
    _patch_registry(monkeypatch, {"RandomForest": _fake_info("RandomForest", None)})
    pf.prefetch_weights(["RandomForest"])  # prefetch_weights is None -> skipped, no error


def test_unknown_model_skipped_unless_raising(monkeypatch):
    _patch_registry(monkeypatch, {})
    pf.prefetch_weights(["Nonexistent"])  # ValueError swallowed
    with pytest.raises(ValueError, match="unknown"):
        pf.prefetch_weights(["Nonexistent"], raise_on_error=True)


def test_prefetch_error_logged_then_raised_when_requested(monkeypatch):
    def boom():
        raise RuntimeError("download failed")

    _patch_registry(monkeypatch, {"M": _fake_info("M_GPU", boom)})
    pf.prefetch_weights(["M"])  # swallowed
    with pytest.raises(RuntimeError, match="download failed"):
        pf.prefetch_weights(["M"], raise_on_error=True)


def test_missing_optional_dependency_always_skipped(monkeypatch):
    def missing():
        raise ImportError("no module named 'tabpfn'")

    _patch_registry(monkeypatch, {"M": _fake_info("M_GPU", missing)})
    pf.prefetch_weights(["M"])  # ImportError swallowed
    pf.prefetch_weights(["M"], raise_on_error=True)  # even when raising


# --- structured report --------------------------------------------------------------------------


def test_prefetch_weights_returns_one_result_per_name(monkeypatch, tmp_path):
    existing = tmp_path / "weights.bin"
    existing.write_bytes(b"x")

    def missing_dep():
        raise ImportError("no module named 'tabpfn'")

    def boom():
        raise RuntimeError("download failed")

    shared = lambda: None
    _patch_registry(
        monkeypatch,
        {
            "Ok": _fake_info("Ok_GPU", lambda: None),
            "OkPath": _fake_info("OkPath_GPU", lambda: str(existing)),
            "OkPaths": _fake_info("OkPaths_GPU", lambda: [existing, str(existing)]),
            "Empty": _fake_info("Empty_GPU", list),
            "Gone": _fake_info("Gone_GPU", lambda: [str(tmp_path / "nope.bin")]),
            "Tree": _fake_info("Tree", None),
            "NoDep": _fake_info("NoDep_GPU", missing_dep),
            "Boom": _fake_info("Boom_GPU", boom),
            "A": _fake_info("A_GPU", shared),
            "B": _fake_info("B_GPU", shared),
        },
    )
    report = pf.prefetch_weights(["Ok", "OkPath", "OkPaths", "Empty", "Gone", "Tree", "NoDep", "Boom", "A", "B", "Zzz"])
    by_name = {result.name: result for result in report.results}
    assert [result.name for result in report.results] == [
        "Ok", "OkPath", "OkPaths", "Empty", "Gone", "Tree", "NoDep", "Boom", "A", "B", "Zzz",
    ]  # fmt: skip
    assert by_name["Ok"].status == "ok"
    assert by_name["Ok"].method == "Ok_GPU"
    assert by_name["OkPath"].status == "ok"
    assert by_name["OkPaths"].status == "ok"
    assert by_name["Empty"].status == "partial"
    assert "no paths" in by_name["Empty"].detail
    assert by_name["Gone"].status == "partial"
    assert "nope.bin" in by_name["Gone"].detail
    assert by_name["Tree"].status == "nothing"
    assert by_name["NoDep"].status == "missing_dependency"
    assert "tabpfn" in by_name["NoDep"].detail
    assert by_name["Boom"].status == "failed"
    assert by_name["Boom"].detail == "RuntimeError: download failed"
    assert by_name["A"].status == "ok"
    assert by_name["B"].status == "ok"  # alias of the same prefetcher records the first outcome
    assert by_name["Zzz"].status == "unknown"
    assert by_name["Zzz"].method is None
    assert not report.complete
    assert [result.name for result in report.with_status("partial", "failed")] == ["Empty", "Gone", "Boom"]
    summary = report.summary()
    assert "failed: Boom" in summary
    assert "partial: Empty, Gone" in summary
    assert summary.endswith("prefetch incomplete (11 model(s))")


def test_mapping_return_checks_the_values_not_the_keys(monkeypatch, tmp_path):
    """A ``{task: local_path}`` return (the EXAONE-Tabular convention) is judged by its paths."""
    existing = tmp_path / "clf.bin"
    existing.write_bytes(b"x")
    _patch_registry(
        monkeypatch,
        {
            "AllThere": _fake_info("AllThere_GPU", lambda: {"classification": str(existing), "regression": existing}),
            "OneGone": _fake_info(
                "OneGone_GPU", lambda: {"classification": str(existing), "regression": tmp_path / "reg.bin"}
            ),
            "EmptyMap": _fake_info("EmptyMap_GPU", dict),
        },
    )
    report = pf.prefetch_weights(["AllThere", "OneGone", "EmptyMap"])
    by_name = {result.name: result for result in report.results}
    assert by_name["AllThere"].status == "ok"
    assert by_name["AllThere"].detail == ""
    assert by_name["OneGone"].status == "partial"
    assert "reg.bin" in by_name["OneGone"].detail
    assert "classification" not in by_name["OneGone"].detail  # keys are never mistaken for paths
    assert by_name["EmptyMap"].status == "partial"


def test_non_path_return_is_ok_and_the_loop_continues(monkeypatch):
    """A prefetcher returning a flag or a count is not checked and never aborts the remaining names."""
    calls: list[str] = []

    def flag():
        calls.append("flag")
        return True

    def count():
        calls.append("count")
        return 3

    def after():
        calls.append("after")

    _patch_registry(
        monkeypatch,
        {
            "Flag": _fake_info("Flag_GPU", flag),
            "Count": _fake_info("Count_GPU", count),
            "After": _fake_info("After_GPU", after),
        },
    )
    report = pf.prefetch_weights(["Flag", "Count", "After"], raise_on_error=True)
    assert calls == ["flag", "count", "after"]
    assert [result.status for result in report.results] == ["ok", "ok", "ok"]
    assert "not checked" in report.results[0].detail
    assert "not checked" in report.results[1].detail
    assert report.results[2].detail == ""
    assert report.complete


def test_alias_of_failed_prefetcher_records_the_failure(monkeypatch):
    def boom():
        raise RuntimeError("download failed")

    _patch_registry(monkeypatch, {"A": _fake_info("A_GPU", boom), "B": _fake_info("B_GPU", boom)})
    report = pf.prefetch_weights(["A", "B"])
    assert [result.status for result in report.results] == ["failed", "failed"]
    assert report.results[1].detail == report.results[0].detail


def test_report_complete_and_merge(monkeypatch):
    _patch_registry(monkeypatch, {"M": _fake_info("M_GPU", lambda: None), "Tree": _fake_info("Tree", None)})
    report = pf.prefetch_weights(["M", "Tree"])
    assert report.complete
    assert report.summary() == "ok: M\nnothing: Tree\nprefetch complete (2 model(s))"
    other = pf.PrefetchReport((pf.PrefetchResult("X", None, "unknown", "unknown X"),))
    merged = pf.PrefetchReport.merge(report, other, pf.PrefetchReport())
    assert [result.name for result in merged.results] == ["M", "Tree", "X"]
    assert not merged.complete
    assert pf.PrefetchReport().complete  # nothing requested, nothing missing


# --- local-first Hub resolution -------------------------------------------------------------------


class _FakeHub:
    """Stand-in for ``hf_hub_download`` / ``snapshot_download`` recording the ``local_files_only`` flag.

    ``cached`` decides whether the local-only call succeeds; an online call always succeeds and
    returns ``target``.
    """

    def __init__(self, target, *, cached: bool):
        self.target = target
        self.cached = cached
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        from huggingface_hub.errors import LocalEntryNotFoundError

        self.calls.append(kwargs)
        if kwargs.get("local_files_only") and not self.cached:
            raise LocalEntryNotFoundError("not cached")
        return str(self.target)

    @property
    def local_flags(self) -> list[bool]:
        return [bool(call.get("local_files_only", False)) for call in self.calls]


def test_resolve_hf_file_uses_cache_then_downloads(monkeypatch, tmp_path):
    blob = tmp_path / "blobs" / "abc"
    blob.parent.mkdir()
    blob.write_bytes(b"x")
    link = tmp_path / "snapshots" / "sha" / "model.bin"
    link.parent.mkdir(parents=True)
    link.symlink_to(blob)

    hub = _FakeHub(link, cached=True)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", hub)
    path = pf.resolve_hf_file("org/repo", "model.bin", revision="v1", subfolder="sub", token=True)
    assert path == str(blob.resolve())  # symlink resolved to the blob
    assert hub.local_flags == [True]
    assert hub.calls[0]["repo_id"] == "org/repo"
    assert hub.calls[0]["filename"] == "model.bin"
    assert hub.calls[0]["revision"] == "v1"
    assert hub.calls[0]["subfolder"] == "sub"
    assert hub.calls[0]["token"] is True

    hub = _FakeHub(link, cached=False)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", hub)
    assert pf.resolve_hf_file("org/repo", "model.bin") == str(blob.resolve())
    assert hub.local_flags == [True, False]


def test_resolve_hf_file_without_download_raises_after_one_local_call(monkeypatch, tmp_path):
    hub = _FakeHub(tmp_path / "model.bin", cached=False)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", hub)
    with pytest.raises(pf.WeightsUnavailableError, match="org/repo@main:model.bin") as excinfo:
        pf.resolve_hf_file("org/repo", "model.bin", revision="main", allow_download=False)
    assert isinstance(excinfo.value, FileNotFoundError)
    assert hub.local_flags == [True]


def test_resolve_hf_snapshot_prefers_a_complete_cached_snapshot(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshots" / "0123456789abcdef0123456789abcdef01234567"
    (snapshot / "sub").mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "sub" / "model.safetensors").write_bytes(b"x")
    hub = _FakeHub(snapshot, cached=True)
    monkeypatch.setattr("huggingface_hub.snapshot_download", hub)

    path = pf.resolve_hf_snapshot(
        "org/repo", revision="main", allow_patterns=["*.json"], required_files=["config.json", "sub/model.safetensors"]
    )
    assert path == str(snapshot.resolve())
    assert hub.local_flags == [True]
    assert hub.calls[0]["allow_patterns"] == ["*.json"]
    assert hub.calls[0]["revision"] == "main"
    assert pf.commit_from_snapshot_path(path) == "0123456789abcdef0123456789abcdef01234567"


def test_resolve_hf_snapshot_redownloads_when_required_files_are_missing(monkeypatch, tmp_path):
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")

    class _CompletingHub(_FakeHub):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            if not kwargs.get("local_files_only"):
                (snapshot / "model.safetensors").write_bytes(b"x")
            return result

    hub = _CompletingHub(snapshot, cached=True)
    monkeypatch.setattr("huggingface_hub.snapshot_download", hub)
    path = pf.resolve_hf_snapshot("org/repo", required_files=["config.json", "model.safetensors"])
    assert path == str(snapshot.resolve())
    assert hub.local_flags == [True, False]


def test_resolve_hf_snapshot_without_download_raises_for_missing_or_partial(monkeypatch, tmp_path):
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    hub = _FakeHub(snapshot, cached=False)
    monkeypatch.setattr("huggingface_hub.snapshot_download", hub)
    with pytest.raises(pf.WeightsUnavailableError, match="not in the local"):
        pf.resolve_hf_snapshot("org/repo", allow_download=False)
    assert hub.local_flags == [True]

    hub = _FakeHub(snapshot, cached=True)
    monkeypatch.setattr("huggingface_hub.snapshot_download", hub)
    with pytest.raises(pf.WeightsUnavailableError, match="cached without \\['model.bin'\\]"):
        pf.resolve_hf_snapshot("org/repo", required_files=["model.bin"], allow_download=False)
    assert hub.local_flags == [True]


def test_resolve_hf_snapshot_raises_when_download_leaves_files_missing(monkeypatch, tmp_path):
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    hub = _FakeHub(snapshot, cached=False)
    monkeypatch.setattr("huggingface_hub.snapshot_download", hub)
    with pytest.raises(pf.WeightsUnavailableError, match="still lacks"):
        pf.resolve_hf_snapshot("org/repo", required_files=["model.bin"])
    assert hub.local_flags == [True, False]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/c/hub/models--org--repo/snapshots/0123456789abcdef0123456789abcdef01234567/model.bin", "0123456789abcdef0123456789abcdef01234567"),
        ("/c/hub/models--org--repo/snapshots/abcdef1", "abcdef1"),
        ("/c/hub/models--org--repo/blobs/0123456789abcdef", None),
        ("/some/other/path/snapshots", None),
        ("", None),
    ],
)  # fmt: skip
def test_commit_from_snapshot_path(path, expected):
    assert pf.commit_from_snapshot_path(path) == expected
