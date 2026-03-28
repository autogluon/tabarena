from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path


def zip_in_memory(path: Path) -> io.BytesIO:
    """
    Create an in-memory ZIP archive of a directory.

    This method recursively traverses the given directory, compresses
    all contained files into a ZIP archive, and returns the archive
    as an in-memory `io.BytesIO` buffer. Paths inside the archive
    are stored relative to the provided root directory.

    Parameters
    ----------
    path : Path
        The root directory whose contents will be compressed into
        the ZIP archive.

    Returns
    -------
    io.BytesIO
        A binary buffer positioned at the beginning, containing the
        ZIP archive data. The caller can read from this buffer or
        upload it directly (e.g., to S3) without writing a local file.

    Raises
    ------
    FileNotFoundError
    If no files exist in the given directory to be zipped.
    """
    # Create an in-memory buffer
    buffer = io.BytesIO()
    file_count = 0

    # Write the zip archive into the buffer
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(path):
            for file in files:
                file_count += 1
                file_path = Path(root) / file
                # Store relative path inside the zip
                arcname = file_path.relative_to(path)
                zf.write(file_path, arcname=arcname)

    if file_count == 0:
        raise FileNotFoundError(f"No files found to zip in directory: {path}")

    # Reset buffer position for reading
    buffer.seek(0)
    return buffer
