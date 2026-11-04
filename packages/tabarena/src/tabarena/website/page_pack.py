"""Pack a leaderboard's HTML pages into one file, ``pages.sqlite``.

Every published cell carries six self-contained pages (the leaderboard table and five explorers):
9,344 files and 1.4 GB over TabArena's 1,856 cells. Nearly all of it repeats. Pages of one kind
share their code and styles, twin pools share their numbers, and neighbouring cells differ in a few
values. As files, that is a commit of thousands of files per update and half the Space's
20,000-file budget. Packed, each distinct piece of a page is stored once, and the pieces of one page
kind are compressed together, so the redundancy between cells compresses away too: about 10 MB.

A page is cut at line boundaries (:func:`split_page`). A long line, which is where a page embeds
its data, is a piece of its own. The short lines between them (code and styles) are cut wherever a
line's checksum says so, so the same code yields the same pieces in every page, whatever differs
around it.

The file is SQLite, so any client reads it without this package:

- ``meta(key, value)``: ``format`` (``1``) and ``version``, a hash over every page that changes
  whenever any page does. The Space puts it in its page URLs, so a browser never shows a page it
  cached from an older pack.
- ``stream(id, data)``: the distinct pieces of one page kind, concatenated in order of first use and
  xz-compressed. A kind larger than ``max_stream_bytes`` spans several streams, compressed in
  parallel.
- ``chunk(id, stream, start, length)``: where each piece sits in its decompressed stream.
- ``page(path, chunks)``: each page by its path under the data root (its cell's folder and file
  name), with its chunk ids, space-separated. The page is the concatenation of those chunks.

The leaderboard Space reads it with a copy of :func:`read_page_pack` (its ``page_pack.py``).
"""

from __future__ import annotations

import hashlib
import lzma
import sqlite3
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

#: Lines longer than this (in bytes) are a piece of their own: the data a page embeds.
LONG_LINE = 1024
#: A short line ends a piece when the low bits of its checksum are zero, one line in 16 on average.
_CUT_MASK = 15
FORMAT = "1"
PAGES_FILE = "pages.sqlite"


def split_page(page: bytes) -> list[bytes]:
    """Cut ``page`` into pieces at line boundaries; their concatenation is ``page``."""
    pieces: list[bytes] = []
    run: list[bytes] = []
    lines = page.split(b"\n")
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            line += b"\n"
        elif not line:
            break
        if len(line) > LONG_LINE:
            if run:
                pieces.append(b"".join(run))
                run = []
            pieces.append(line)
            continue
        run.append(line)
        if zlib.crc32(line) & _CUT_MASK == 0:
            pieces.append(b"".join(run))
            run = []
    if run:
        pieces.append(b"".join(run))
    return pieces


def chunk_pages(pages: dict[str, bytes]) -> tuple[dict[str, list[bytes]], dict[bytes, bytes]]:
    """Cut every page and key the pieces by their digest.

    Returns each page's list of piece digests and the distinct pieces by digest, which is what
    :func:`write_page_pack` takes. Pages built in separate processes are chunked there and merged by
    the caller: a digest names the same piece wherever it was cut.
    """
    layout: dict[str, list[bytes]] = {}
    pieces: dict[bytes, bytes] = {}
    for path, page in pages.items():
        digests = []
        for piece in split_page(page):
            digest = hashlib.blake2b(piece, digest_size=16).digest()
            pieces.setdefault(digest, piece)
            digests.append(digest)
        layout[path] = digests
    return layout, pieces


def _compress(data: bytes) -> bytes:
    return lzma.compress(data, preset=9)


def write_page_pack(
    path: str | Path,
    layout: dict[str, list[bytes]],
    pieces: dict[bytes, bytes],
    *,
    max_stream_bytes: int = 32 << 20,
    n_jobs: int | None = None,
) -> str:
    """Write the pages of ``layout`` (path -> piece digests) to a pack at ``path``; return its version.

    The output depends on the pages alone, not on the order they were built in. Pieces go into the
    stream of the page kind (file name) of the first page, in path order, that uses them, so the
    pieces that compress against each other sit together.
    """
    path = Path(path)
    streams: list[list[bytes]] = []
    stream_size: list[int] = []
    current_stream: dict[str, int] = {}
    chunk_ids: dict[bytes, int] = {}
    chunk_rows: list[tuple[int, int, int, int]] = []
    page_rows: list[tuple[str, str]] = []
    version = hashlib.sha256(FORMAT.encode())
    for page_path in sorted(layout):
        kind = page_path.rsplit("/", 1)[-1]
        ids = []
        for digest in layout[page_path]:
            if digest not in chunk_ids:
                piece = pieces[digest]
                stream = current_stream.get(kind)
                if stream is None or stream_size[stream] + len(piece) > max_stream_bytes:
                    stream = current_stream[kind] = len(streams)
                    streams.append([])
                    stream_size.append(0)
                chunk_ids[digest] = len(chunk_rows)
                chunk_rows.append((len(chunk_rows), stream, stream_size[stream], len(piece)))
                streams[stream].append(piece)
                stream_size[stream] += len(piece)
            ids.append(chunk_ids[digest])
            version.update(digest)
        page_rows.append((page_path, " ".join(map(str, ids))))
        version.update(page_path.encode() + b"\0")

    with ProcessPoolExecutor(max_workers=n_jobs or min(len(streams), 32) or 1) as pool:
        compressed = list(pool.map(_compress, (b"".join(s) for s in streams)))

    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.executescript(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            "CREATE TABLE stream (id INTEGER PRIMARY KEY, data BLOB NOT NULL);"
            "CREATE TABLE chunk (id INTEGER PRIMARY KEY, stream INTEGER NOT NULL, start INTEGER NOT NULL,"
            " length INTEGER NOT NULL);"
            "CREATE TABLE page (path TEXT PRIMARY KEY, chunks TEXT NOT NULL) WITHOUT ROWID;"
        )
        version_hex = version.hexdigest()[:16]
        con.executemany("INSERT INTO meta VALUES (?, ?)", [("format", FORMAT), ("version", version_hex)])
        con.executemany("INSERT INTO stream VALUES (?, ?)", enumerate(compressed))
        con.executemany("INSERT INTO chunk VALUES (?, ?, ?, ?)", chunk_rows)
        con.executemany("INSERT INTO page VALUES (?, ?)", page_rows)
        con.commit()
        con.execute("VACUUM")
    finally:
        con.close()
    tmp.replace(path)
    return version_hex


def read_page_pack(path: str | Path) -> tuple[str, dict[str, bytes]]:
    """Every page of the pack at ``path`` by its path, with the pack's version."""
    con = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    try:
        meta = dict(con.execute("SELECT key, value FROM meta"))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path} is page-pack format {meta.get('format')!r}, this reader knows {FORMAT!r}")
        streams = {sid: lzma.decompress(data) for sid, data in con.execute("SELECT id, data FROM stream")}
        chunks = {cid: (sid, start, length) for cid, sid, start, length in con.execute("SELECT * FROM chunk")}
        pages = {}
        for page_path, ids in con.execute("SELECT path, chunks FROM page"):
            parts = []
            for cid in ids.split():
                sid, start, length = chunks[int(cid)]
                parts.append(streams[sid][start : start + length])
            pages[page_path] = b"".join(parts)
    finally:
        con.close()
    return meta["version"], pages
