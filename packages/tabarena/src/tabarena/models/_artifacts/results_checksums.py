"""Committed MD5 checksums of the hosted results tables.

A re-upload keeps a method's suite and file names, so a present local results table alone does not show that it is
current. ``results_checksums.json`` maps each hosted table's path relative to the cache root to the MD5 of the
uploaded file. The loader compares the local file against it without a network call, ``run_upload_results.py``
records the checksums of every upload, and ``python -m tabarena.tools.results_checksums --check`` compares the file
with the store's ETags.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tabarena.models._method_metadata import MethodMetadata

RESULTS_CHECKSUMS_JSON = Path(__file__).parent / "results_checksums.json"


def file_md5(path: str | Path) -> str:
    """Hex MD5 of a file's bytes, the ETag of a single-part upload."""
    return hashlib.md5(Path(path).read_bytes(), usedforsecurity=False).hexdigest()


@functools.cache
def load_results_checksums() -> dict[str, str]:
    """The committed checksums, keyed by the table's path relative to the cache root (posix)."""
    if not RESULTS_CHECKSUMS_JSON.exists():
        return {}
    return json.loads(RESULTS_CHECKSUMS_JSON.read_text())


def checksum_key(method_metadata: MethodMetadata, path: str | Path) -> str:
    return method_metadata.relative_to_cache_root(Path(path)).as_posix()


def expected_results_md5(method_metadata: MethodMetadata, path: str | Path) -> str | None:
    """The committed MD5 of the hosted table at ``path``, or ``None`` when none is recorded."""
    return load_results_checksums().get(checksum_key(method_metadata, path))


def write_results_checksums(updates: dict[str, str]) -> None:
    """Merge ``updates`` into the committed file (sorted keys, so the diff stays reviewable)."""
    checksums = {**load_results_checksums(), **updates}
    RESULTS_CHECKSUMS_JSON.write_text(json.dumps(dict(sorted(checksums.items())), indent=1) + "\n")
    load_results_checksums.cache_clear()


def record_results_checksums(method_metadata: MethodMetadata) -> dict[str, str]:
    """Record the MD5s of a method's local results tables, as uploaded by ``MethodUploader.upload_results``."""
    updates = {
        checksum_key(method_metadata, path): file_md5(path)
        for path in method_metadata.path_results_files()
        if Path(path).exists()
    }
    write_results_checksums(updates)
    return updates
