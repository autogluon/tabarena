"""Compare the committed results checksums with the hosted results tables.

Every registered method with hosted results (TabArena and BeyondArena, including superseded entries) should have one
entry per results table in ``results_checksums.json``, equal to the ETag of the hosted object (the MD5 of a
single-part upload).

Usage:
    python -m tabarena.tools.results_checksums --check    # exit non-zero on any mismatch (CI)
    python -m tabarena.tools.results_checksums --refresh  # rewrite the file from the hosted ETags
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from tabarena.models._artifacts.results_checksums import checksum_key, load_results_checksums, write_results_checksums

if TYPE_CHECKING:
    from tabarena.models._method_metadata import MethodMetadata


def hosted_methods() -> list[MethodMetadata]:
    """Every registered method whose results tables live in a remote store, deduplicated by (method, suite)."""
    from tabarena.contexts.beyondarena.methods import beyond_method_metadata_complete_collection
    from tabarena.contexts.tabarena.methods import tabarena_method_metadata_complete_collection

    methods: dict[tuple[str, str], MethodMetadata] = {}
    for collection in (tabarena_method_metadata_complete_collection, beyond_method_metadata_complete_collection):
        for m in collection.method_metadata_lst:
            if m.has_results and m.has_remote_cache:
                methods.setdefault((m.method, m.suite), m)
    return list(methods.values())


def expected_keys(methods: list[MethodMetadata]) -> set[str]:
    """The checksum keys the given methods' results tables need."""
    return {checksum_key(m, path) for m in methods for path in m.path_results_files()}


#: Placeholder ETag of a table whose store could not be asked (connection error), as opposed to a missing object.
UNREACHABLE = "unreachable"


def hosted_etags(methods: list[MethodMetadata], max_workers: int = 8) -> dict[str, str | None]:
    """The hosted ETag of each results table: ``None`` where the store has no object, :data:`UNREACHABLE` where the
    request failed twice (once concurrently, once in a serial retry pass).
    """

    def etag(item: tuple[MethodMetadata, Path]) -> str | None:
        m, path = item
        downloader = m.method_downloader()
        try:
            return downloader.remote_etag(downloader.local_to_key(path))
        except Exception:  # recorded as unreachable and retried
            return UNREACHABLE

    items = {checksum_key(m, path): (m, Path(path)) for m in methods for path in m.path_results_files()}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        etags = dict(zip(items, pool.map(etag, items.values()), strict=True))
    for key in [k for k, v in etags.items() if v == UNREACHABLE]:
        etags[key] = etag(items[key])
    return etags


def find_mismatches(committed: dict[str, str], hosted: dict[str, str | None]) -> list[str]:
    """Human-readable problems: tables without a committed checksum, missing objects, and differing checksums.

    Unreachable tables are not problems here; see :func:`unreachable`.
    """
    problems = []
    for key, etag in sorted(hosted.items()):
        if etag == UNREACHABLE:
            continue
        if key not in committed:
            problems.append(f"no committed checksum: {key}")
        elif etag is None:
            problems.append(f"not hosted: {key}")
        elif "-" in etag:
            problems.append(f"multipart ETag {etag} is not an MD5: {key}")
        elif etag != committed[key]:
            problems.append(f"hosted {etag} != committed {committed[key]}: {key}")
    return problems


def unreachable(hosted: dict[str, str | None]) -> list[str]:
    return sorted(k for k, v in hosted.items() if v == UNREACHABLE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Exit non-zero when a checksum is missing or differs.")
    mode.add_argument("--refresh", action="store_true", help="Write the hosted ETags into results_checksums.json.")
    args = parser.parse_args()

    methods = hosted_methods()
    hosted = hosted_etags(methods)
    if args.refresh:
        updates = {k: v for k, v in hosted.items() if v not in (None, UNREACHABLE) and "-" not in v}
        write_results_checksums(updates)
        print(f"Recorded {len(updates)} of {len(hosted)} results tables from {len(methods)} methods.")
        for key in sorted(set(hosted) - set(updates)):
            print(f"\tskipped ({hosted[key] or 'not hosted'}): {key}")
        return
    problems = find_mismatches(load_results_checksums(), hosted)
    problems += [f"unreachable: {key}" for key in unreachable(hosted)]
    print(f"Checked {len(hosted)} results tables from {len(methods)} methods: {len(problems)} problem(s).")
    for problem in problems:
        print(f"\t{problem}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
