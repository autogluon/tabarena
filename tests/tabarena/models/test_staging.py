"""Tests for the node-staging plan (``tabarena.models.staging``), with a fake registry and fake paths."""

from __future__ import annotations

from types import SimpleNamespace

from tabarena.models.prefetch import PrefetchReport, PrefetchResult
from tabarena.models.staging import collect_weight_paths


def test_collect_weight_paths_classifies_hf_repo_dirs_tabpfn_files_and_unresolved(tmp_path, monkeypatch):
    hub = tmp_path / "hf" / "hub"
    blob = hub / "models--a--b" / "blobs" / "sha1"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"w")
    snapshot_file = hub / "models--a--b" / "snapshots" / "rev" / "model.bin"
    ckpt = tmp_path / "tabpfn" / "clf.ckpt"
    other = tmp_path / "elsewhere" / "net.pt"

    shared = lambda: [str(snapshot_file), str(blob)]  # two names share one prefetcher: called once
    infos = {
        "HF": SimpleNamespace(prefetch_weights=shared),
        "HF-alias": SimpleNamespace(prefetch_weights=shared),
        "PFN": SimpleNamespace(prefetch_weights=lambda: {"classifier": ckpt}),
        "Other": SimpleNamespace(prefetch_weights=lambda: str(other)),
        "Tree": SimpleNamespace(prefetch_weights=None),
        "Broken": SimpleNamespace(prefetch_weights=list),
        "Failed": SimpleNamespace(prefetch_weights=lambda: ["/never/called"]),
    }

    def fake_get(name):
        if name not in infos:
            raise ValueError(name)
        return infos[name]

    monkeypatch.setattr("tabarena.models.utils.get_model_info_from_name", fake_get)
    report = PrefetchReport((PrefetchResult("Failed", "Failed", "failed", "boom"),))

    plan = collect_weight_paths(["HF", "HF-alias", "PFN", "Other", "Tree", "Broken", "Failed", "Unknown"], report)

    assert plan["hf_repo_dirs"] == [str(hub / "models--a--b")]
    assert plan["hf_hub_cache_src"] == str(hub)
    assert plan["hf_home_src"] == str(tmp_path / "hf")
    assert plan["tabpfn_files"] == [str(ckpt)]
    assert plan["tabpfn_cache_dir_src"] == str(tmp_path / "tabpfn")
    assert plan["other_paths"] == [str(other)]
    assert plan["unresolved"] == ["Broken", "Failed"]
    assert plan["complete"] is False
