"""`MethodMetadata` loaders fetch the artifacts from the method's remote store on a local cache miss."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import tabarena.models._method_metadata as mm_module
from tabarena.models._method_metadata import MethodMetadata


class _Downloader:
    """Records the downloads asked for; `on_download` can make the artifact appear."""

    def __init__(self, on_download=None):
        self.calls: list[str] = []
        self.on_download = on_download

    def _fetch(self, kind: str):
        self.calls.append(kind)
        if self.on_download is not None:
            self.on_download()

    def download_processed(self):
        self._fetch("processed")

    def download_results(self):
        self._fetch("results")

    def download_raw(self):
        self._fetch("raw")


def _metadata(cache_type: str = "r2", cache_root: Path | None = None, **overrides) -> MethodMetadata:
    cache_kwargs = {"bucket": "b", "prefix": "p"} if cache_type != "local" else {}
    fields = dict(
        method="Fake",
        display_name="Fake",
        compute="cpu",
        date="2026-01-01",
        date_introduced="2026-01-01",
        ag_key="FAKE",
        config_default="Fake_c1_BAG_L1",
        can_hpo=False,
        is_bag=False,
        has_raw=True,
        has_processed=True,
        has_results=True,
        suite="fake-suite",
        cache_type=cache_type,
        cache_kwargs=cache_kwargs,
        cache_root=cache_root,
    )
    fields.update(overrides)
    return MethodMetadata.config(**fields)


def _wire_processed(monkeypatch, downloader: _Downloader, present: dict) -> list[Path]:
    """`EvaluationRepository.from_dir` succeeds only once `present["ok"]` is True."""
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


class TestLoadProcessed:
    def test_auto_downloads_once_on_a_miss_and_retries(self, monkeypatch):
        present = {"ok": False}
        downloader = _Downloader(on_download=lambda: present.update(ok=True))
        loaded = _wire_processed(monkeypatch, downloader, present)
        metadata = _metadata()
        assert metadata.load_processed() == "repo"
        assert downloader.calls == ["processed"]
        assert loaded == [metadata.path_processed, metadata.path_processed]

    def test_auto_does_not_download_when_present(self, monkeypatch):
        downloader = _Downloader()
        _wire_processed(monkeypatch, downloader, {"ok": True})
        assert _metadata().load_processed() == "repo"
        assert downloader.calls == []

    def test_true_always_downloads_first(self, monkeypatch):
        downloader = _Downloader()
        _wire_processed(monkeypatch, downloader, {"ok": True})
        assert _metadata().load_processed(download=True) == "repo"
        assert downloader.calls == ["processed"]

    def test_false_raises_without_downloading(self, monkeypatch):
        downloader = _Downloader()
        _wire_processed(monkeypatch, downloader, {"ok": False})
        with pytest.raises(FileNotFoundError, match="download=True"):
            _metadata().load_processed(download=False)
        assert downloader.calls == []

    def test_local_cache_raises_without_downloading(self, monkeypatch):
        downloader = _Downloader()
        _wire_processed(monkeypatch, downloader, {"ok": False})
        with pytest.raises(FileNotFoundError, match="no remote store"):
            _metadata("local").load_processed()
        assert downloader.calls == []

    def test_not_hosted_raises_without_downloading(self, monkeypatch):
        downloader = _Downloader()
        _wire_processed(monkeypatch, downloader, {"ok": False})
        with pytest.raises(FileNotFoundError, match="has_processed=False"):
            _metadata(has_processed=False).load_processed()
        assert downloader.calls == []

    def test_true_downloads_even_when_not_hosted(self, monkeypatch):
        downloader = _Downloader()
        _wire_processed(monkeypatch, downloader, {"ok": True})
        assert _metadata(has_processed=False).load_processed(download=True) == "repo"
        assert downloader.calls == ["processed"]

    def test_missing_after_download_still_raises(self, monkeypatch):
        downloader = _Downloader()  # the remote has no processed.zip: the download is a no-op
        _wire_processed(monkeypatch, downloader, {"ok": False})
        with pytest.raises(FileNotFoundError, match="has no processed artifacts for this method either"):
            _metadata().load_processed()
        assert downloader.calls == ["processed"]

    def test_explicit_other_path_is_loaded_as_given_without_download(self, monkeypatch, tmp_path):
        downloader = _Downloader()
        loaded = _wire_processed(monkeypatch, downloader, {"ok": False})
        with pytest.raises(FileNotFoundError, match="context.json"):
            _metadata().load_processed(path_processed=tmp_path / "elsewhere")
        assert downloader.calls == []
        assert loaded == [tmp_path / "elsewhere"]

    def test_explicit_own_path_still_downloads(self, monkeypatch):
        present = {"ok": False}
        downloader = _Downloader(on_download=lambda: present.update(ok=True))
        _wire_processed(monkeypatch, downloader, present)
        metadata = _metadata()
        assert metadata.load_processed(path_processed=str(metadata.path_processed)) == "repo"
        assert downloader.calls == ["processed"]

    def test_true_with_other_path_is_rejected(self, monkeypatch, tmp_path):
        downloader = _Downloader()
        _wire_processed(monkeypatch, downloader, {"ok": True})
        with pytest.raises(ValueError, match="download=True"):
            _metadata().load_processed(path_processed=tmp_path / "elsewhere", download=True)
        assert downloader.calls == []


class TestLoadResults:
    """The results tables are real parquet files under a temporary cache root."""

    @staticmethod
    def _write(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"metric_error": [0.1]}).to_parquet(path)

    def test_auto_downloads_the_results_tables_on_a_miss(self, monkeypatch, tmp_path):
        metadata = _metadata(cache_root=tmp_path)  # method_type "config" -> hpo_results
        downloader = _Downloader(on_download=lambda: self._write(metadata.path_results_hpo()))
        monkeypatch.setattr(MethodMetadata, "method_downloader", lambda self, *a, **k: downloader)
        assert len(metadata.load_results()) == 1
        assert downloader.calls == ["results"]

    def test_model_results_present_loads_without_download(self, monkeypatch, tmp_path):
        metadata = _metadata(cache_root=tmp_path)
        self._write(metadata.path_results_model())
        downloader = _Downloader()
        monkeypatch.setattr(MethodMetadata, "method_downloader", lambda self, *a, **k: downloader)
        assert len(metadata.load_model_results()) == 1
        assert downloader.calls == []

    def test_not_hosted_raises_without_downloading(self, monkeypatch, tmp_path):
        metadata = _metadata(cache_root=tmp_path, has_results=False)
        downloader = _Downloader()
        monkeypatch.setattr(MethodMetadata, "method_downloader", lambda self, *a, **k: downloader)
        with pytest.raises(FileNotFoundError, match="has_results=False"):
            metadata.load_model_results()
        assert downloader.calls == []

    def test_false_raises_without_downloading(self, monkeypatch, tmp_path):
        metadata = _metadata(cache_root=tmp_path)
        downloader = _Downloader()
        monkeypatch.setattr(MethodMetadata, "method_downloader", lambda self, *a, **k: downloader)
        with pytest.raises(FileNotFoundError, match="download=True"):
            metadata.load_results(download=False)
        assert downloader.calls == []


class TestLoadRaw:
    @staticmethod
    def _wire(monkeypatch, downloader: _Downloader):
        monkeypatch.setattr(mm_module, "load_raw", lambda path_raw, **_kwargs: ["result", str(path_raw)])
        monkeypatch.setattr(MethodMetadata, "method_downloader", lambda self, *a, **k: downloader)

    @staticmethod
    def _place_result(path_raw: Path):
        (path_raw / "d" / "0").mkdir(parents=True, exist_ok=True)
        (path_raw / "d" / "0" / "results.pkl").write_bytes(b"")

    def test_auto_downloads_when_no_result_files_exist(self, monkeypatch, tmp_path):
        metadata = _metadata(cache_root=tmp_path)
        downloader = _Downloader(on_download=lambda: self._place_result(metadata.path_raw))
        self._wire(monkeypatch, downloader)
        assert metadata.load_raw() == ["result", str(metadata.path_raw)]
        assert downloader.calls == ["raw"]

    def test_present_loads_without_download(self, monkeypatch, tmp_path):
        metadata = _metadata(cache_root=tmp_path)
        self._place_result(metadata.path_raw)
        downloader = _Downloader()
        self._wire(monkeypatch, downloader)
        assert metadata.load_raw() == ["result", str(metadata.path_raw)]
        assert downloader.calls == []

    def test_not_hosted_raises_without_downloading(self, monkeypatch, tmp_path):
        metadata = _metadata(cache_root=tmp_path, has_raw=False)
        downloader = _Downloader()
        self._wire(monkeypatch, downloader)
        with pytest.raises(FileNotFoundError, match="has_raw=False"):
            metadata.load_raw()
        assert downloader.calls == []

    def test_explicit_other_path_skips_the_miss_check(self, monkeypatch, tmp_path):
        metadata = _metadata(cache_root=tmp_path)
        downloader = _Downloader()
        self._wire(monkeypatch, downloader)
        # no results.pkl there either, but the path is not the cache path: loaded as given
        assert metadata.load_raw(path_raw=tmp_path / "elsewhere") == ["result", str(tmp_path / "elsewhere")]
        assert downloader.calls == []


class TestContextForwardsTheFlags:
    @staticmethod
    def _context_module():
        import tabarena.contexts.abstract_arena_context as ctx_module

        return ctx_module

    def test_load_repo_passes_download_processed(self, monkeypatch):
        ctx_module = self._context_module()
        seen = []
        monkeypatch.setattr(
            MethodMetadata, "load_processed", lambda self, *a, **kw: seen.append(kw.get("download")) or "repo"
        )
        monkeypatch.setattr(ctx_module, "EvaluationRepositoryCollection", lambda repos, config_fallback: repos)
        repos = ctx_module.AbstractArenaContext.load_repo(
            None, methods=[_metadata(), _metadata()], download_processed=False, max_workers=1
        )
        assert repos == ["repo", "repo"]
        assert seen == [False, False]

    def test_load_results_passes_download_results(self, monkeypatch):
        ctx_module = self._context_module()
        seen = []
        frame = pd.DataFrame({"metric_error": [0.1]})
        monkeypatch.setattr(
            MethodMetadata, "load_results", lambda self, *a, **kw: seen.append(kw.get("download")) or frame
        )

        class _Ctx(ctx_module.AbstractArenaContext):
            def __init__(self):
                pass

            def method_metadata(self, method):
                return method

        out = ctx_module.AbstractArenaContext.load_results(_Ctx(), methods=[_metadata()], download_results=True)
        assert len(out) == 1
        assert seen == [True]
