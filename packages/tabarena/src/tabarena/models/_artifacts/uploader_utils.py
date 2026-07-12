"""Streaming ZIP creation for artifact uploads.

:func:`zip_dir_stream` produces a directory's ZIP archive as a readable stream, written by a
background thread through a bounded in-memory pipe — so compressing and consuming (uploading)
overlap and memory stays bounded regardless of archive size (nothing is buffered whole in RAM
or written to disk).
"""

from __future__ import annotations

import contextlib
import os
import queue
import threading
import zipfile
from pathlib import Path

from tqdm import tqdm

#: Deflate level for artifact archives. Measured on raw prediction pickles: level 1 compresses
#: ~3x faster than the zlib default (6) for a nearly identical ratio (0.63 vs 0.61).
DEFAULT_ZIP_COMPRESSLEVEL = 1

#: Bound on the pipe between the zip writer and the upload reader (chunks are whatever sizes
#: the zip writer emits, typically <= 64 KiB — the buffer stays a few MiB at most).
_PIPE_MAX_CHUNKS = 256


class _ZipPipe:
    """Bounded in-memory pipe: the zip-writing thread ``write``s, the uploader ``read``s.

    The writer side is file-like enough for ``zipfile.ZipFile`` (non-seekable, so zipfile uses
    data-descriptor entries); the reader side is file-like enough for ``upload_fileobj``
    (sequential ``read``). A writer failure is re-raised to the reader at end-of-stream, so a
    consumer that reads to EOF can never silently treat a partial archive as complete.
    """

    _EOF = object()

    def __init__(self, max_chunks: int = _PIPE_MAX_CHUNKS):
        self._queue: queue.Queue = queue.Queue(maxsize=max_chunks)
        self._reader_leftover = b""
        self._reader_at_eof = False
        self._aborted = False
        self.writer_exception: BaseException | None = None

    # -- writer side (used by zipfile) ---------------------------------------------------------
    def write(self, data) -> int:
        data = bytes(data)
        if not data:
            return 0
        while True:
            if self._aborted:
                raise BrokenPipeError("Consumer stopped reading the zip stream.")
            try:
                self._queue.put(data, timeout=0.1)
                return len(data)
            except queue.Full:
                continue

    def flush(self) -> None:
        pass

    def seekable(self) -> bool:
        return False

    def close_writer(self, exc: BaseException | None = None) -> None:
        """Mark end-of-stream (with the writer's failure, if any); called exactly once."""
        self.writer_exception = exc
        while True:
            if self._aborted:
                return
            try:
                self._queue.put(self._EOF, timeout=0.1)
                return
            except queue.Full:
                continue

    # -- reader side (used by upload_fileobj) --------------------------------------------------
    def read(self, size: int = -1) -> bytes:
        parts = [self._reader_leftover]
        n = len(self._reader_leftover)
        self._reader_leftover = b""
        while not self._reader_at_eof and (size is None or size < 0 or n < size):
            chunk = self._queue.get()
            if chunk is self._EOF:
                self._reader_at_eof = True
                if self.writer_exception is not None:
                    raise self.writer_exception
                break
            parts.append(chunk)
            n += len(chunk)
        data = b"".join(parts)
        if size is not None and size >= 0 and len(data) > size:
            self._reader_leftover = data[size:]
            data = data[:size]
        return data

    def readable(self) -> bool:
        return True

    def abort(self) -> None:
        """Unblock a still-running writer after the reader stopped consuming."""
        self._aborted = True
        with contextlib.suppress(queue.Empty):
            while True:
                self._queue.get_nowait()


def _write_zip(
    pipe: _ZipPipe,
    files: list[tuple[Path, Path, int]],
    compresslevel: int,
    progress_desc: str | None,
) -> None:
    progress = None
    if progress_desc is not None:
        total_bytes = sum(size for _, _, size in files)
        progress = tqdm(total=total_bytes, desc=progress_desc, unit="B", unit_scale=True)
    try:
        with zipfile.ZipFile(pipe, "w", zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as zf:
            for file_path, arcname, size in files:
                zf.write(file_path, arcname=arcname)
                if progress is not None:
                    progress.update(size)
    except BaseException as e:  # must reach the reader, whatever it is
        pipe.close_writer(exc=e)
    else:
        pipe.close_writer()
    finally:
        if progress is not None:
            progress.close()


@contextlib.contextmanager
def zip_dir_stream(
    path: str | Path,
    compresslevel: int = DEFAULT_ZIP_COMPRESSLEVEL,
    progress_desc: str | None = None,
):
    """Yield a readable stream of ``path``'s ZIP archive, produced on the fly.

    A background thread writes the archive of ``path``'s files (paths inside the archive are
    relative to it) into a bounded pipe while the caller reads — so e.g. a multipart upload
    runs concurrently with the compression, and memory use is bounded by the pipe (+ whatever
    the consumer buffers) rather than the archive size.

    ``progress_desc`` enables a byte-level progress bar with that label, advanced as each
    source file is archived. Because the consumer trails the writer only by the bounded
    buffers, it tracks overall progress (incl. an ETA) closely.

    Raises:
    ------
    FileNotFoundError
        If no files exist under ``path``.
    """
    path = Path(path)
    files: list[tuple[Path, Path, int]] = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = Path(root) / filename
            # Store relative path inside the zip
            files.append((file_path, file_path.relative_to(path), file_path.stat().st_size))
    if not files:
        raise FileNotFoundError(f"No files found to zip in directory: {path}")

    pipe = _ZipPipe()
    writer = threading.Thread(
        target=_write_zip,
        args=(pipe, files, compresslevel, progress_desc),
        name=f"zip-stream:{path.name}",
        daemon=True,
    )
    writer.start()
    try:
        yield pipe
    finally:
        # No-op after a full read (the writer is done once EOF was consumed); otherwise unblocks
        # a stuck writer so join() cannot hang (e.g. the upload failed mid-stream).
        pipe.abort()
        writer.join()
    if pipe.writer_exception is not None:
        # The consumer exited cleanly without reading to EOF, so it never saw the failure.
        raise pipe.writer_exception
