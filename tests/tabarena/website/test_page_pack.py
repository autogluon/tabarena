from __future__ import annotations

import sqlite3

import pytest

from tabarena.website.page_pack import LONG_LINE, chunk_pages, read_page_pack, split_page, write_page_pack


def _page(data: str, title: str = "Cell") -> bytes:
    code = "\n".join(f"  function f{i}() {{ return {i}; }}" for i in range(200))
    return (
        f"<html><head><title>{title}</title></head>\n<script>\n  const POINTS = {data};\n{code}\n</script>\n".encode()
    )


@pytest.mark.parametrize("page", [b"", b"a", b"a\n", b"a\nb", b"\n\n", _page("[1, 2, 3]"), b"x" * 5000 + b"\ny\n"])
def test_a_page_is_the_concatenation_of_its_pieces(page):
    assert b"".join(split_page(page)) == page


def test_a_long_line_is_a_piece_of_its_own():
    data = "[" + ",".join(["1"] * LONG_LINE) + "]"
    assert f"  const POINTS = {data};\n".encode() in split_page(_page(data))


def test_the_code_of_two_pages_is_stored_once(tmp_path):
    pages = {
        "entrants_models/a/table.html": _page("[" + "1," * LONG_LINE + "1]", title="A"),
        "entrants_models/b/table.html": _page("[" + "2," * LONG_LINE + "2]", title="B"),
    }
    layout, pieces = chunk_pages(pages)
    shared = set(layout["entrants_models/a/table.html"]) & set(layout["entrants_models/b/table.html"])
    assert sum(len(pieces[d]) for d in shared) > 0.5 * sum(len(p) for p in pieces.values())

    write_page_pack(tmp_path / "pages.sqlite", layout, pieces)
    version, back = read_page_pack(tmp_path / "pages.sqlite")
    assert back == pages
    con = sqlite3.connect(tmp_path / "pages.sqlite")
    assert con.execute("SELECT COUNT(*) FROM chunk").fetchone()[0] == len(pieces)


def test_the_pack_depends_on_the_pages_only(tmp_path):
    pages = {f"cell{i}/page{j}.html": _page(f"[{i}, {j}]") for i in range(3) for j in range(2)}
    layout, pieces = chunk_pages(pages)
    forward = write_page_pack(tmp_path / "a.sqlite", layout, pieces, max_stream_bytes=1 << 10)
    reverse = write_page_pack(tmp_path / "b.sqlite", dict(reversed(layout.items())), pieces, max_stream_bytes=1 << 10)
    assert forward == reverse
    assert (tmp_path / "a.sqlite").read_bytes() == (tmp_path / "b.sqlite").read_bytes()

    changed = dict(pages, **{"cell0/page0.html": _page("[9, 9]")})
    assert write_page_pack(tmp_path / "c.sqlite", *chunk_pages(changed)) != forward
