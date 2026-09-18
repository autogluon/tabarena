"""`load_repo` fetches a method's processed artifacts from its remote store when they are missing."""

from __future__ import annotations

import pytest

from tabarena.contexts.abstract_arena_context import AbstractArenaContext
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


def _wire(monkeypatch, metadata: MethodMetadata, downloader: _Downloader, present: dict):
    """Make `load_processed` succeed only once `present["ok"]` is True, and hand out `downloader`."""

    def load_processed(self, *args, **kwargs):
        if not present["ok"]:
            raise FileNotFoundError("context.json")
        return "repo"

    monkeypatch.setattr(MethodMetadata, "load_processed", load_processed)
    monkeypatch.setattr(MethodMetadata, "method_downloader", lambda self, *a, **k: downloader)


class TestLoadProcessedDownload:
    def test_auto_downloads_once_on_a_miss_and_retries(self, monkeypatch):
        present = {"ok": False}
        downloader = _Downloader(on_download=lambda: present.update(ok=True))
        metadata = _metadata("r2")
        _wire(monkeypatch, metadata, downloader, present)
        assert AbstractArenaContext._load_processed(metadata, download_processed="auto") == "repo"
        assert downloader.calls == 1

    def test_auto_does_not_download_when_present(self, monkeypatch):
        present = {"ok": True}
        downloader = _Downloader()
        metadata = _metadata("r2")
        _wire(monkeypatch, metadata, downloader, present)
        assert AbstractArenaContext._load_processed(metadata, download_processed="auto") == "repo"
        assert downloader.calls == 0

    def test_true_always_downloads_first(self, monkeypatch):
        present = {"ok": True}
        downloader = _Downloader()
        metadata = _metadata("r2")
        _wire(monkeypatch, metadata, downloader, present)
        assert AbstractArenaContext._load_processed(metadata, download_processed=True) == "repo"
        assert downloader.calls == 1

    def test_false_raises_without_downloading(self, monkeypatch):
        present = {"ok": False}
        downloader = _Downloader()
        metadata = _metadata("r2")
        _wire(monkeypatch, metadata, downloader, present)
        with pytest.raises(FileNotFoundError, match="download_processed=True"):
            AbstractArenaContext._load_processed(metadata, download_processed=False)
        assert downloader.calls == 0

    def test_local_cache_raises_without_downloading(self, monkeypatch):
        present = {"ok": False}
        downloader = _Downloader()
        metadata = _metadata("local")
        _wire(monkeypatch, metadata, downloader, present)
        with pytest.raises(FileNotFoundError, match="no remote store"):
            AbstractArenaContext._load_processed(metadata, download_processed="auto")
        assert downloader.calls == 0

    def test_missing_after_download_still_raises(self, monkeypatch):
        present = {"ok": False}
        downloader = _Downloader()  # the remote has no processed.zip: the download is a no-op
        metadata = _metadata("r2")
        _wire(monkeypatch, metadata, downloader, present)
        with pytest.raises(FileNotFoundError):
            AbstractArenaContext._load_processed(metadata, download_processed="auto")
        assert downloader.calls == 1
