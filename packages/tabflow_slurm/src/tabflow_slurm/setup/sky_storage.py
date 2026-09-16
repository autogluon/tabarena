"""Object storage a SkyPilot run stages its artifacts in, behind a small protocol.

Everything the head node and the workers exchange (the environment manifest and repo archives,
the job batch, the task queue with its claim and done markers, results and logs) lives under one
``gs://`` prefix. Both sides talk to it through :class:`GcsStorage`, a thin wrapper around the
``gcloud storage`` CLI that is present on this cluster's login nodes and on every SkyPilot GCP VM.
Tests substitute a directory backed implementation of the same :class:`Storage` protocol.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Storage(Protocol):
    """What the SkyPilot scheduler and its worker need from an object store."""

    def exists(self, uri: str) -> bool:
        """Whether the object ``uri`` exists."""

    def exists_prefix(self, uri: str) -> bool:
        """Whether at least one object lives under the prefix ``uri``."""

    def list_names(self, uri: str) -> list[str]:
        """The names directly under the prefix ``uri`` (objects and sub-prefixes, without the prefix)."""

    def list_files(self, uri: str) -> list[str]:
        """Every object under the prefix ``uri``, recursively, as paths relative to it."""

    def read_text(self, uri: str) -> str:
        """The content of the text object ``uri``."""

    def write_text(self, uri: str, text: str) -> None:
        """Create or replace the text object ``uri``."""

    def create_exclusive(self, uri: str, text: str) -> bool:
        """Create the text object ``uri`` only if it does not exist; ``True`` when this call created it."""

    def upload_file(self, local: Path, uri: str) -> None:
        """Copy one local file to the object ``uri``."""

    def download_file(self, uri: str, local: Path) -> None:
        """Copy the object ``uri`` to the local file ``local``."""

    def upload_dir(self, local: Path, uri: str) -> None:
        """Mirror the directory ``local`` under the prefix ``uri`` (new and changed files)."""

    def copy_tree(self, local: Path, uri: str) -> None:
        """Copy every file under ``local`` to ``<uri>/<relative path>`` without listing the destination."""

    def download_dir(self, uri: str, local: Path) -> None:
        """Mirror the prefix ``uri`` into the directory ``local`` (new and changed files, nothing deleted)."""


@dataclass
class GcsStorage:
    """:class:`Storage` over the ``gcloud storage`` CLI.

    Commands run without a shell (``subprocess.run`` with an argument list). ``gcloud storage ls``
    exits non-zero when nothing matches, which is how :meth:`exists` and :meth:`exists_prefix`
    decide, and ``cp --if-generation-match=0`` is the exclusive create behind :meth:`create_exclusive`.
    """

    gcloud: str = "gcloud"
    """The ``gcloud`` executable (a path or a name resolved on ``PATH``)."""

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(  # noqa: S603
            [self.gcloud, "storage", *args],
            check=check,
            capture_output=True,
            text=True,
        )

    def exists(self, uri: str) -> bool:
        """Whether the object ``uri`` exists (``gcloud storage ls`` succeeds on it)."""
        return self._run("ls", uri, check=False).returncode == 0

    def exists_prefix(self, uri: str) -> bool:
        """Whether the prefix ``uri`` has any object (``ls`` on ``<uri>/`` lists something)."""
        return self._run("ls", uri.rstrip("/") + "/", check=False).returncode == 0

    def list_names(self, uri: str) -> list[str]:
        """Names directly under ``uri`` (``ls`` prints one full URI per line; sub-prefixes end in ``/``)."""
        prefix = uri.rstrip("/") + "/"
        result = self._run("ls", prefix, check=False)
        if result.returncode != 0:
            return []
        names = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith(prefix) and len(line) > len(prefix):
                names.append(line[len(prefix) :].rstrip("/"))
        return names

    def list_files(self, uri: str) -> list[str]:
        """Objects under ``uri`` at any depth (``ls <uri>/**``), relative to the prefix."""
        prefix = uri.rstrip("/") + "/"
        result = self._run("ls", prefix + "**", check=False)
        if result.returncode != 0:
            return []
        files = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith(prefix) and len(line) > len(prefix) and not line.endswith(("/", ":")):
                files.append(line[len(prefix) :])
        return files

    def read_text(self, uri: str) -> str:
        """``gcloud storage cat <uri>``."""
        return self._run("cat", uri).stdout

    def write_text(self, uri: str, text: str) -> None:
        """Write ``text`` to a temporary file and ``cp`` it to ``uri``."""
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write(text)
            path = fh.name
        try:
            self._run("cp", path, uri)
        finally:
            Path(path).unlink(missing_ok=True)

    def create_exclusive(self, uri: str, text: str) -> bool:
        """``cp --if-generation-match=0``: succeeds only when no object exists at ``uri`` yet."""
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write(text)
            path = fh.name
        try:
            return self._run("cp", "--if-generation-match=0", path, uri, check=False).returncode == 0
        finally:
            Path(path).unlink(missing_ok=True)

    def upload_file(self, local: Path, uri: str) -> None:
        """``gcloud storage cp <local> <uri>``."""
        self._run("cp", str(local), uri)

    def download_file(self, uri: str, local: Path) -> None:
        """``gcloud storage cp <uri> <local>`` (the parent directory is created first)."""
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        self._run("cp", uri, str(local))

    def upload_dir(self, local: Path, uri: str) -> None:
        """``gcloud storage rsync -r <local> <uri>``."""
        self._run("rsync", "-r", str(local), uri)

    def copy_tree(self, local: Path, uri: str) -> None:
        """One ``cp`` per directory that holds files, with explicit object names.

        Unlike ``rsync`` this never lists the destination prefix (a full sweep's results run to
        hundreds of thousands of objects), and unlike ``cp -r`` on a directory it does not depend on
        whether the destination prefix already exists.
        """
        local = Path(local)
        for directory in sorted({p.parent for p in local.rglob("*") if p.is_file()}):
            files = sorted(str(p) for p in directory.iterdir() if p.is_file())
            rel = directory.relative_to(local).as_posix()
            target = uri.rstrip("/") + ("/" if rel == "." else f"/{rel}/")
            self._run("cp", *files, target)

    def download_dir(self, uri: str, local: Path) -> None:
        """``gcloud storage rsync -r <uri> <local>`` (the directory is created first)."""
        Path(local).mkdir(parents=True, exist_ok=True)
        self._run("rsync", "-r", uri, str(local))
