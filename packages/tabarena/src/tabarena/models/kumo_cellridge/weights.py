from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen


def checkpoint_path(task):
    if task not in ("classification", "regression"):
        raise ValueError(task)
    manifest = json.loads(Path(__file__).with_name("weights.json").read_text())
    info = manifest[task]
    root = Path(os.environ.get("KUMO_CELLRIDGE_WEIGHTS", Path.home() / ".cache" / "kumo-cellridge" / "1024"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / info["filename"]

    def verify(p):
        with p.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != info["sha256"]:
                raise ValueError(f"Invalid CellRidge {task} weights digest")

    if not path.is_file():
        url = "https://github.com/1587causalai/kumo-small-cellridge-1024/releases/download/v1.0/" + path.name
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as out:
            tmp = Path(out.name)
        try:
            with urlopen(url, timeout=90) as response, tmp.open("wb") as out:  # noqa: S310 -- fixed HTTPS GitHub release and verified digest
                copyfileobj(response, out)
            verify(tmp)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
    else:
        verify(path)
    return path
