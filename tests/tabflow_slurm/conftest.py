"""Shared fixtures for the tabflow_slurm tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest


class LocalDirStorage:
    """A directory-backed stand-in for ``tabflow_slurm.setup.sky_storage.Storage``.

    Maps ``gs://<bucket>/<path>`` to ``<root>/<bucket>/<path>`` so the SkyPilot scheduler and the
    environment staging can be exercised without gcloud or a network. Counts uploads so tests can
    assert what was (not) re-staged.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.uploads: list[str] = []

    def path_for(self, uri: str) -> Path:
        assert uri.startswith("gs://"), uri
        return self.root / uri[len("gs://") :]

    def exists(self, uri: str) -> bool:
        return self.path_for(uri).is_file()

    def exists_prefix(self, uri: str) -> bool:
        path = self.path_for(uri.rstrip("/"))
        return path.is_dir() and any(path.rglob("*"))

    def list_names(self, uri: str) -> list[str]:
        path = self.path_for(uri.rstrip("/"))
        return sorted(p.name for p in path.iterdir()) if path.is_dir() else []

    def read_text(self, uri: str) -> str:
        return self.path_for(uri).read_text()

    def write_text(self, uri: str, text: str) -> None:
        target = self.path_for(uri)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        self.uploads.append(uri)

    def create_exclusive(self, uri: str, text: str) -> bool:
        target = self.path_for(uri)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("x") as fh:
                fh.write(text)
        except FileExistsError:
            return False
        self.uploads.append(uri)
        return True

    def upload_file(self, local: Path, uri: str) -> None:
        target = self.path_for(uri)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, target)
        self.uploads.append(uri)

    def download_file(self, uri: str, local: Path) -> None:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.path_for(uri), local)

    def upload_dir(self, local: Path, uri: str) -> None:
        shutil.copytree(local, self.path_for(uri), dirs_exist_ok=True)
        self.uploads.append(uri)

    def copy_tree(self, local: Path, uri: str) -> None:
        shutil.copytree(local, self.path_for(uri), dirs_exist_ok=True)
        self.uploads.append(uri)

    def download_dir(self, uri: str, local: Path) -> None:
        Path(local).mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.path_for(uri), local, dirs_exist_ok=True)


@pytest.fixture
def local_storage(tmp_path) -> LocalDirStorage:
    return LocalDirStorage(tmp_path / "gcs")
