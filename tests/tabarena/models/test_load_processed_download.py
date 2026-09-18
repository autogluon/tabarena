"""`MethodMetadata.load_processed` fetches the processed artifacts from the remote store on a local miss."""

from __future__ import annotations

from pathlib import Path

import pytest

import tabarena.models._method_metadata as mm_module
from tabarena.models._method_metadata import MethodMetadata


class _Downloader:
    def __init__(self, on_download=None):
        self.calls = 0
        self.on_download = on_download

    def download_processed(self):
        self.calls += 1
        if self.on_download is not None:
            self.on_download()


def _metadata(cache_type: str) -> MethodMetadata:
    cache_kwargs = {"bucket": "b", "prefix": "p"} if cache_type != "local" else {}
    return MethodMetadata.config(
        method="Fake",
        display_name="Fake",
        compute="cpu",
        date="2026-01-01",
        date_introduced="2026-01-01",
        ag_key="FAKE",
        config_default="Fake_c1_BAG_L1",
        can_hpo=False,
        is_bag=False,
        has_processed=True,
        has_results=True,
        suite="fake-suite",
        cache_type=cache_type,
        cache_kwargs=cache_kwargs,
    )


def _wire(monkeypatch, downloader: _Downloader, present: dict) -> list[Path]:
    """`from_dir` succeeds only once `present["ok"]` is True; the downloader is handed out as is.

    Returns the list of paths `from_dir` was asked to load.
    """
    loaded: list[Path] = []

    class _Repo:
        @staticmethod
        def from_dir(path, **_kwargs):
            loaded.append(Path(path))
            if not present["ok"]:
                raise FileNotFoundError("context.json")
            return "repo"

    monkeypatch.setattr(mm_module, "EvaluationRepository", _Repo)
    monkeypatch.setattr(MethodMetadata, "method_downloader", lambda self, *a, **k: downloader)
    return loaded


class TestLoadProcessedDownload:
    def test_auto_downloads_once_on_a_miss_and_retries(self, monkeypatch):
        present = {"ok": False}
        downloader = _Downloader(on_download=lambda: present.update(ok=True))
        loaded = _wire(monkeypatch, downloader, present)
        metadata = _metadata("r2")
        assert metadata.load_processed() == "repo"
        assert downloader.calls == 1
        assert loaded == [metadata.path_processed, metadata.path_processed]

    def test_auto_does_not_download_when_present(self, monkeypatch):
        downloader = _Downloader()
        _wire(monkeypatch, downloader, {"ok": True})
        assert _metadata("r2").load_processed() == "repo"
        assert downloader.calls == 0

    def test_true_always_downloads_first(self, monkeypatch):
        downloader = _Downloader()
        _wire(monkeypatch, downloader, {"ok": True})
        assert _metadata("r2").load_processed(download=True) == "repo"
        assert downloader.calls == 1

    def test_false_raises_without_downloading(self, monkeypatch):
        downloader = _Downloader()
        _wire(monkeypatch, downloader, {"ok": False})
        with pytest.raises(FileNotFoundError, match="download=True"):
            _metadata("r2").load_processed(download=False)
        assert downloader.calls == 0

    def test_local_cache_raises_without_downloading(self, monkeypatch):
        downloader = _Downloader()
        _wire(monkeypatch, downloader, {"ok": False})
        with pytest.raises(FileNotFoundError, match="no remote store"):
            _metadata("local").load_processed()
        assert downloader.calls == 0

    def test_missing_after_download_still_raises(self, monkeypatch):
        downloader = _Downloader()  # the remote has no processed.zip: the download is a no-op
        _wire(monkeypatch, downloader, {"ok": False})
        with pytest.raises(FileNotFoundError, match="has no processed artifacts for this method either"):
            _metadata("r2").load_processed()
        assert downloader.calls == 1

    def test_explicit_other_path_is_loaded_as_given_without_download(self, monkeypatch, tmp_path):
        downloader = _Downloader()
        loaded = _wire(monkeypatch, downloader, {"ok": False})
        with pytest.raises(FileNotFoundError, match="context.json"):
            _metadata("r2").load_processed(path_processed=tmp_path / "elsewhere")
        assert downloader.calls == 0
        assert loaded == [tmp_path / "elsewhere"]

    def test_explicit_own_path_still_downloads(self, monkeypatch):
        present = {"ok": False}
        downloader = _Downloader(on_download=lambda: present.update(ok=True))
        _wire(monkeypatch, downloader, present)
        metadata = _metadata("r2")
        assert metadata.load_processed(path_processed=str(metadata.path_processed)) == "repo"
        assert downloader.calls == 1

    def test_true_with_other_path_is_rejected(self, monkeypatch, tmp_path):
        downloader = _Downloader()
        _wire(monkeypatch, downloader, {"ok": True})
        with pytest.raises(ValueError, match="download=True"):
            _metadata("r2").load_processed(path_processed=tmp_path / "elsewhere", download=True)
        assert downloader.calls == 0


class TestLoadRepoForwardsTheFlag:
    def test_load_repo_passes_download_processed(self, monkeypatch):
        from tabarena.contexts.abstract_arena_context import AbstractArenaContext

        seen = []

        def load_processed(self, *args, **kwargs):
            seen.append(kwargs.get("download"))
            return "repo"

        monkeypatch.setattr(MethodMetadata, "load_processed", load_processed)
        import tabarena.contexts.abstract_arena_context as ctx_module

        monkeypatch.setattr(ctx_module, "EvaluationRepositoryCollection", lambda repos, config_fallback: repos)
        repos = AbstractArenaContext.load_repo(
            None, methods=[_metadata("r2"), _metadata("r2")], download_processed=False, max_workers=1
        )
        assert repos == ["repo", "repo"]
        assert seen == [False, False]
