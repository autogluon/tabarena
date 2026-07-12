"""Tests for the artifact uploader's streamed zip uploads (no real object store involved)."""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tabarena.models._artifacts.uploader import MethodUploader
from tabarena.models._artifacts.uploader_utils import _ZipPipe, zip_dir_stream
from tabarena.models._method_metadata import MethodMetadata


class _FakeClient:
    """Captures uploads; reads fileobj streams fully like a real transfer would."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.kwargs: dict[str, dict] = {}

    def upload_fileobj(self, Fileobj, Bucket, Key, **kwargs):
        data = Fileobj.read()
        self.objects[Key] = data
        self.kwargs[Key] = kwargs

    def upload_file(self, Filename, Bucket, Key, **kwargs):
        self.objects[Key] = Path(Filename).read_bytes()
        self.kwargs[Key] = kwargs


class _FakeUploader(MethodUploader):
    def _make_client(self):
        return _FakeClient()


@pytest.fixture
def uploader(tmp_path) -> _FakeUploader:
    method_metadata = MethodMetadata.config(method="Dummy", suite="dummy-suite", cache_root=tmp_path)
    return _FakeUploader(method_metadata, bucket="bucket")


def _write_tree(root: Path) -> dict[str, bytes]:
    """A nested file tree incl. a file large enough to span many pipe chunks."""
    rng = np.random.default_rng(0)
    files = {
        "Dummy_c1/3901/0_0/results.pkl": rng.bytes(300_000),
        "Dummy_c1/3902/0_0/results.pkl": b"small",
        "Dummy_c2/3901/0_0/results.pkl": rng.bytes(1_000),
    }
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return files


class TestUploadRaw:
    def test_roundtrip(self, uploader):
        files = _write_tree(uploader.method_metadata.path_raw)
        uploader.upload_raw()

        key = "cache/artifacts/dummy-suite/methods/Dummy/raw.zip"
        assert key in uploader.client.objects
        with zipfile.ZipFile(io.BytesIO(uploader.client.objects[key])) as zf:
            assert zf.testzip() is None
            assert sorted(zf.namelist()) == sorted(files)
            for rel, data in files.items():
                assert zf.read(rel) == data
        # The tuned multipart transfer config is passed through to the client.
        assert "Config" in uploader.client.kwargs[key]

    def test_empty_dir_raises(self, uploader):
        uploader.method_metadata.path_raw.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="No files found to zip"):
            uploader.upload_raw()

    def test_unreadable_file_fails_upload(self, uploader):
        files = _write_tree(uploader.method_metadata.path_raw)
        blocked = uploader.method_metadata.path_raw / next(iter(files))
        blocked.chmod(0o000)
        try:
            with pytest.raises(PermissionError):
                uploader.upload_raw()
        finally:
            blocked.chmod(0o644)
        # The failed stream must not have produced a stored object.
        assert not uploader.client.objects


class TestUploadResults:
    def test_uploads_result_files_with_config(self, uploader):
        results_dir = uploader.method_metadata.path_results
        results_dir.mkdir(parents=True)
        (results_dir / "model_results.parquet").write_bytes(b"model")
        (results_dir / "hpo_results.parquet").write_bytes(b"hpo")
        uploader.upload_results()

        prefix = "cache/artifacts/dummy-suite/methods/Dummy/results"
        assert uploader.client.objects[f"{prefix}/model_results.parquet"] == b"model"
        assert uploader.client.objects[f"{prefix}/hpo_results.parquet"] == b"hpo"
        assert all("Config" in kwargs for kwargs in uploader.client.kwargs.values())


class TestZipPipe:
    def test_read_sizes_are_exact_until_eof(self):
        pipe = _ZipPipe()
        pipe.write(b"abcdef")
        pipe.write(b"ghij")
        pipe.close_writer()
        assert pipe.read(4) == b"abcd"
        assert pipe.read(4) == b"efgh"
        assert pipe.read(100) == b"ij"
        assert pipe.read(100) == b""

    def test_writer_exception_raised_at_eof(self):
        pipe = _ZipPipe()
        pipe.write(b"partial")
        pipe.close_writer(exc=ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            pipe.read()

    def test_abort_unblocks_writer(self):
        pipe = _ZipPipe(max_chunks=1)
        pipe.write(b"fills the queue")
        pipe.abort()
        with pytest.raises(BrokenPipeError):
            pipe.write(b"more")


class TestZipDirStream:
    def test_partial_consumer_surfaces_writer_state(self, tmp_path):
        # A consumer that stops reading early must not hang the writer thread on exit.
        # Incompressible data far beyond the pipe capacity guarantees the writer is still
        # mid-archive (blocked on the full pipe) when the consumer exits.
        big_dir = tmp_path / "d"
        big_dir.mkdir()
        for i in range(30):
            (big_dir / f"f{i}").write_bytes(os.urandom(500_000))
        with pytest.raises(BrokenPipeError):
            with zip_dir_stream(big_dir) as stream:
                stream.read(10)  # stop early; writer gets aborted on exit
