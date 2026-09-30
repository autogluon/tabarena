"""The committed results checksums cover every hosted results table and match the hosted objects."""

from __future__ import annotations

import pytest
import requests

from tabarena.models._artifacts.results_checksums import load_results_checksums
from tabarena.tools.results_checksums import (
    UNREACHABLE,
    expected_keys,
    find_mismatches,
    hosted_etags,
    hosted_methods,
    unreachable,
)


def test_find_mismatches_reports_each_kind_of_problem():
    committed = {"ok": "a" * 32, "changed": "a" * 32, "gone": "a" * 32, "multi": "a" * 32}
    hosted = {
        "ok": "a" * 32,
        "changed": "b" * 32,
        "gone": None,
        "multi": "abc-3",
        "new": "c" * 32,
        "flaky": UNREACHABLE,
    }
    problems = find_mismatches(committed, hosted)
    assert [p.split(":")[0] for p in problems] == [
        "hosted bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb != committed aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "not hosted",
        "multipart ETag abc-3 is not an MD5",
        "no committed checksum",
    ]
    assert unreachable(hosted) == ["flaky"]


def test_every_hosted_results_table_has_a_committed_checksum():
    missing = expected_keys(hosted_methods()) - set(load_results_checksums())
    assert not missing, (
        f"Results tables without a committed checksum (run `python -m tabarena.tools.results_checksums --refresh`): "
        f"{sorted(missing)}"
    )


def test_committed_checksums_match_the_hosted_objects():
    try:
        requests.head("https://data.tabarena.ai/", timeout=(5, 10))
    except requests.RequestException:
        pytest.skip("data.tabarena.ai is unreachable")
    hosted = hosted_etags(hosted_methods())
    problems = find_mismatches(load_results_checksums(), hosted)
    assert not problems, (
        "results_checksums.json disagrees with the hosted results (a re-upload without a checksum update?); run "
        "`python -m tabarena.tools.results_checksums --refresh` and commit the file:\n" + "\n".join(problems)
    )
    if unreachable(hosted):
        pytest.skip(f"{len(unreachable(hosted))} results tables were unreachable: {unreachable(hosted)}")


def test_file_md5_is_memoized_until_the_file_changes(tmp_path, monkeypatch):
    import hashlib
    import os

    from tabarena.models._artifacts import results_checksums

    path = tmp_path / "t.parquet"
    path.write_bytes(b"old")
    reads = []
    real_md5 = hashlib.md5
    monkeypatch.setattr(results_checksums.hashlib, "md5", lambda data, **kw: reads.append(data) or real_md5(data, **kw))
    assert results_checksums.file_md5(path) == results_checksums.file_md5(path) == real_md5(b"old").hexdigest()
    assert len(reads) == 1
    path.write_bytes(b"new!")
    os.utime(path, ns=(1, 1))
    assert results_checksums.file_md5(path) == real_md5(b"new!").hexdigest()
    assert len(reads) == 2
