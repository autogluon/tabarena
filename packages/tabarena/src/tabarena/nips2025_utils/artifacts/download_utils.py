from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import requests


def download_and_extract_zip(url: str, path_local: str | Path):
    path_local = Path(path_local)

    # Send a GET request to the URL
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()  # Check for HTTP request errors

    path_local.mkdir(parents=True, exist_ok=True)
    # Use BytesIO to handle the downloaded content as a file-like object
    with ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall(str(path_local))  # Extract to the specified directory
        print("Extraction complete.")
