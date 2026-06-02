"""Tests for transparent (raw + gzip) pickle helpers and CacheFunctionPickle compression."""

from __future__ import annotations

import pytest

from tabarena.utils.cache import CacheFunctionPickle
from tabarena.utils.pickle_utils import dumps_pickle, is_gzipped, load_pickle, read_pickle_bytes

_OBJ = {"task_metadata": {"tid": 1, "name": "x"}, "arr": list(range(1000))}


def test_dumps_pickle_compress_flag():
    assert not is_gzipped(dumps_pickle(_OBJ, compress=False))
    assert is_gzipped(dumps_pickle(_OBJ, compress=True))


@pytest.mark.parametrize("compress", [False, True], ids=["raw", "gzip"])
def test_load_pickle_roundtrips_both_formats(tmp_path, compress):
    p = tmp_path / "f.pkl"
    p.write_bytes(dumps_pickle(_OBJ, compress=compress))
    assert is_gzipped(p.read_bytes()) == compress  # on-disk format as requested
    assert load_pickle(p) == _OBJ  # loads transparently either way
    assert not is_gzipped(read_pickle_bytes(p))  # returns the decompressed pickle bytes


def test_cache_pickle_compressed_roundtrip(tmp_path):
    cache = CacheFunctionPickle(cache_name="results", cache_path=str(tmp_path), compress=True, verbose=False)
    cache.save_cache(_OBJ)
    assert is_gzipped((tmp_path / "results.pkl").read_bytes())  # stored gzip-compressed
    assert cache.load_cache() == _OBJ


def test_cache_pickle_compresses_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("TABARENA_DISABLE_RESULT_COMPRESSION", raising=False)
    cache = CacheFunctionPickle(cache_name="results", cache_path=str(tmp_path), verbose=False)
    cache.save_cache(_OBJ)
    assert is_gzipped((tmp_path / "results.pkl").read_bytes())  # on by default
    assert cache.load_cache() == _OBJ


def test_cache_pickle_compression_disabled_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TABARENA_DISABLE_RESULT_COMPRESSION", "1")
    cache = CacheFunctionPickle(cache_name="results", cache_path=str(tmp_path), verbose=False)
    cache.save_cache(_OBJ)
    assert not is_gzipped((tmp_path / "results.pkl").read_bytes())  # env-disabled
    assert cache.load_cache() == _OBJ


def test_cache_pickle_reads_raw_file_transparently(tmp_path):
    CacheFunctionPickle(cache_name="results", cache_path=str(tmp_path), compress=False, verbose=False).save_cache(_OBJ)
    assert not is_gzipped((tmp_path / "results.pkl").read_bytes())  # stored raw
    # A reader configured to compress still loads a pre-existing raw file fine.
    reader = CacheFunctionPickle(cache_name="results", cache_path=str(tmp_path), compress=True, verbose=False)
    assert reader.load_cache() == _OBJ
