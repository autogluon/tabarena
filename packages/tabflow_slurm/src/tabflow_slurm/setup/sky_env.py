"""Replicate the head node's Python environment on a SkyPilot worker.

A worker VM shares nothing with the head node, so it cannot use its venv. This module captures
that venv once at setup time: the pinned wheels reported by ``uv pip freeze`` plus an archive of
every local checkout the venv installs from disk (editable installs such as this repository and
the AutoGluon fork, and non-editable local installs such as a C++ sdist). The archives capture
the working tree, tracked files as they are on disk plus untracked files that are not ignored,
so a worker runs the same code a SLURM node would. The result is an ``env.json`` manifest in the
bucket; :func:`render_env_setup_script` is the shell a worker runs to turn it back into a venv.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

if TYPE_CHECKING:
    from tabflow_slurm.setup.sky_storage import Storage

_REQUIREMENTS_FILE = "requirements.txt"
_MANIFEST_FILE = "env.json"


def _exe(name: str) -> str:
    """The executable ``name`` resolved on ``PATH`` (falls back to the bare name so the error names it)."""
    return shutil.which(name) or name


@dataclass(frozen=True)
class FreezeSpec:
    """The three kinds of line in a ``uv pip freeze`` output."""

    requirements: tuple[str, ...]
    """Lines to reinstall verbatim: version pins and ``pkg @ git+...`` references."""
    editable: tuple[Path, ...]
    """Local directories installed with ``-e`` (``-e file:///...``)."""
    installs: tuple[Path, ...]
    """Local directories installed from a path without ``-e`` (``name @ file:///...``)."""


def _local_path(spec: str) -> Path | None:
    """The local directory a ``file://`` requirement spec points at, or ``None`` for anything else."""
    spec = spec.strip()
    if spec.startswith("file://"):
        return Path(unquote(urlparse(spec).path))
    return None


def parse_freeze(text: str) -> FreezeSpec:
    """Split a ``uv pip freeze`` output into pins, editable local paths and non-editable local paths.

    ``-e file:///path`` lines are editable installs, ``name @ file:///path`` lines are non-editable
    local installs, blank lines and comments are dropped, everything else (``name==1.2``,
    ``name @ git+https://...``) is kept as a requirement line.
    """
    requirements: list[str] = []
    editable: list[Path] = []
    installs: list[Path] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-e "):
            path = _local_path(line[3:])
            if path is None:
                raise ValueError(f"Cannot stage the editable requirement {line!r}: only file:// paths are supported.")
            editable.append(path)
            continue
        if " @ " in line:
            path = _local_path(line.split(" @ ", 1)[1])
            if path is not None:
                installs.append(path)
                continue
        requirements.append(line)
    return FreezeSpec(tuple(requirements), tuple(editable), tuple(installs))


def run_freeze(python_path: str | Path) -> str:
    """``uv pip freeze`` for the interpreter at ``python_path``."""
    return subprocess.run(  # noqa: S603
        [_exe("uv"), "pip", "freeze", "--python", str(python_path)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout


def git_repo_root(path: Path) -> Path:
    """The top-level directory of the git repository containing ``path``."""
    out = subprocess.run(  # noqa: S603
        [_exe("git"), "-C", str(path), "rev-parse", "--show-toplevel"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    return Path(out.strip())


def group_paths_by_repo(paths: tuple[Path, ...], repo_root_of: Callable[[Path], Path]) -> dict[Path, list[Path]]:
    """Group local install paths by the repository root that contains them (first-seen order)."""
    groups: dict[Path, list[Path]] = {}
    for path in paths:
        groups.setdefault(repo_root_of(path), []).append(path)
    return groups


@dataclass(frozen=True)
class WorkingTree:
    """A repository's working tree packed into a tarball."""

    sha: str
    """Short hash of ``HEAD``."""
    dirty: bool
    """Whether the tree differs from ``HEAD`` (modified tracked files or untracked, non-ignored files)."""
    content_hash: str
    """First eight hex digits of the tarball's sha256; the same tree packs to the same hash."""


def pack_working_tree(root: Path, out: Path) -> WorkingTree:
    """Pack ``root``'s working tree into the gzip tarball ``out`` (deterministic for the same content).

    Includes tracked files as they are on disk and untracked files that are not ignored; deleted
    tracked files are skipped. ``tar`` runs with fixed ownership, mtime and file order and ``gzip -n``
    drops its timestamp, so re-packing an unchanged tree yields byte-identical output.
    """
    listed = subprocess.run(  # noqa: S603
        [_exe("git"), "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    files = sorted({name for name in listed.decode().split("\0") if name and (root / name).exists()})
    tar = subprocess.Popen(  # noqa: S603
        [
            _exe("tar"),
            "-C",
            str(root),
            "--null",
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "-cf",
            "-",
            "-T",
            "-",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    with out.open("wb") as fh:
        gzip = subprocess.Popen([_exe("gzip"), "-n"], stdin=tar.stdout, stdout=fh)  # noqa: S603
        tar.stdout.close()
        tar.stdin.write("\0".join(files).encode() + b"\0")
        tar.stdin.close()
        if gzip.wait() != 0 or tar.wait() != 0:
            raise RuntimeError(f"Packing the working tree of {root} failed.")
    sha = subprocess.run(  # noqa: S603
        [_exe("git"), "-C", str(root), "rev-parse", "--short", "HEAD"], check=True, stdout=subprocess.PIPE, text=True
    ).stdout.strip()
    status = subprocess.run(  # noqa: S603
        [_exe("git"), "-C", str(root), "status", "--porcelain"], check=True, stdout=subprocess.PIPE, text=True
    ).stdout
    return WorkingTree(
        sha=sha, dirty=bool(status.strip()), content_hash=hashlib.sha256(out.read_bytes()).hexdigest()[:8]
    )


@dataclass(frozen=True)
class RepoArchive:
    """One repository staged in the bucket, with the install paths it provides."""

    name: str
    """The repository directory's name, also the directory the worker unpacks it into."""
    sha: str
    dirty: bool
    uri: str
    """``gs://`` location of the tarball."""
    editable: tuple[str, ...]
    """Editable install paths, relative to the repository root."""
    installs: tuple[str, ...]
    """Non-editable local install paths, relative to the repository root."""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sha": self.sha,
            "dirty": self.dirty,
            "archive": self.uri,
            "editable": list(self.editable),
            "install": list(self.installs),
        }


def archive_repo(
    root: Path,
    *,
    storage: Storage,
    repo_prefix: str,
    editable: tuple[Path, ...],
    installs: tuple[Path, ...],
    pack: Callable[[Path, Path], WorkingTree] = pack_working_tree,
) -> RepoArchive:
    """Pack ``root`` and upload it as ``<repo_prefix>/<name>-<sha>-<content hash>.tar.gz`` unless present.

    The name is content addressed, so re-staging an unchanged tree is a no-op and a modified tree
    gets its own archive. A dirty tree is staged as it is (that is the point: the worker runs what
    the head node runs) with a warning naming the repository.
    """
    root = Path(root).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / f"{root.name}.tar.gz"
        tree = pack(root, tarball)
        uri = f"{repo_prefix.rstrip('/')}/{root.name}-{tree.sha}-{tree.content_hash}.tar.gz"
        if not storage.exists(uri):
            storage.upload_file(tarball, uri)
    if tree.dirty:
        warnings.warn(
            f"{root} has uncommitted or untracked changes; the workers run the tree as it is on disk ({uri}).",
            stacklevel=2,
        )
    return RepoArchive(
        name=root.name,
        sha=tree.sha,
        dirty=tree.dirty,
        uri=uri,
        editable=tuple(str(p.resolve().relative_to(root)) for p in editable),
        installs=tuple(str(p.resolve().relative_to(root)) for p in installs),
    )


@dataclass(frozen=True)
class EnvSpec:
    """The staged environment: what the worker's setup script reads from ``env.json``."""

    python_version: str
    requirements_uri: str
    repos: tuple[RepoArchive, ...]
    manifest_uri: str
    env_hash: str
    """sha256 of the manifest; identifies the environment (a different hash means a different venv)."""

    def to_manifest(self) -> dict:
        return {
            "python": self.python_version,
            "requirements": self.requirements_uri,
            "repos": [repo.to_dict() for repo in self.repos],
        }


_STAGED: dict[tuple, EnvSpec] = {}


def stage_environment(
    *,
    python_path: str | Path,
    storage: Storage,
    env_prefix: str,
    repo_prefix: str,
    python_version: str,
    extra_requirement_lines: tuple[str, ...] = (),
    freeze: Callable[[str | Path], str] = run_freeze,
    repo_root_of: Callable[[Path], Path] = git_repo_root,
    archive: Callable[..., RepoArchive] = archive_repo,
) -> EnvSpec:
    """Capture the venv at ``python_path`` into the bucket; return the staged :class:`EnvSpec`.

    Freezes the venv, archives every repository its local installs come from (see
    :func:`archive_repo`), writes ``requirements.txt`` (the pins plus ``extra_requirement_lines``)
    and ``env.json`` under ``<env_prefix>/<env hash>/`` and returns the manifest's location.
    Memoized per process on the arguments, so a plan with several run groups stages once.
    """
    key = (str(python_path), env_prefix, repo_prefix, python_version, tuple(extra_requirement_lines))
    if key in _STAGED:
        return _STAGED[key]

    spec = parse_freeze(freeze(python_path))
    by_repo = group_paths_by_repo(spec.editable + spec.installs, repo_root_of)
    editable_set = {p.resolve() for p in spec.editable}
    repos = tuple(
        archive(
            root,
            storage=storage,
            repo_prefix=repo_prefix,
            editable=tuple(p for p in paths if p.resolve() in editable_set),
            installs=tuple(p for p in paths if p.resolve() not in editable_set),
        )
        for root, paths in by_repo.items()
    )
    requirements = "\n".join([*spec.requirements, *extra_requirement_lines]) + "\n"
    manifest_body = {
        "python": python_version,
        "repos": [repo.to_dict() for repo in repos],
        "requirements": requirements,
    }
    env_hash = hashlib.sha256(json.dumps(manifest_body, sort_keys=True).encode()).hexdigest()[:12]
    base = f"{env_prefix.rstrip('/')}/{env_hash}"
    env = EnvSpec(
        python_version=python_version,
        requirements_uri=f"{base}/{_REQUIREMENTS_FILE}",
        repos=repos,
        manifest_uri=f"{base}/{_MANIFEST_FILE}",
        env_hash=env_hash,
    )
    with tempfile.TemporaryDirectory() as tmp:
        req_path = Path(tmp) / _REQUIREMENTS_FILE
        req_path.write_text(requirements)
        manifest_path = Path(tmp) / _MANIFEST_FILE
        manifest_path.write_text(json.dumps(env.to_manifest(), indent=2))
        if not storage.exists(env.manifest_uri):
            storage.upload_file(req_path, env.requirements_uri)
            storage.upload_file(manifest_path, env.manifest_uri)
    _STAGED[key] = env
    return env


def render_env_setup_script() -> str:
    """The shell that builds ``$HOME/tabarena_sky/venv`` from the manifest at ``$ENV_SPEC``.

    Reads the manifest with the system ``python3`` (no ``jq``), unpacks each repository archive,
    creates the venv with ``uv`` and installs the pins, then the editable and local installs with
    ``--no-deps`` (their dependencies are among the pins). It is idempotent: a worker that already
    built this exact manifest (``env.done``) exits early, which is what a pool worker or a recovered
    job hits on its second run. ``build-essential`` is installed when ``g++`` is missing, for local
    sdists with C++ extensions. ``$CACHE_ROOT`` (the worker's cache directory) is created here too.
    """
    return r"""set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ROOT="$HOME/tabarena_sky"
mkdir -p "$ROOT" "$CACHE_ROOT"
gcloud storage cp "$ENV_SPEC" "$ROOT/env.json"
if [ -f "$ROOT/env.done" ] && cmp -s "$ROOT/env.json" "$ROOT/env.done"; then
    echo "venv already built for $ENV_SPEC"
    exit 0
fi
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v g++ >/dev/null 2>&1 || { sudo apt-get update -q && sudo apt-get install -y -q build-essential; }
rm -rf "$ROOT/src" "$ROOT/venv" "$ROOT/env.done"
mkdir -p "$ROOT/src"
python3 - "$ROOT/env.json" > "$ROOT/env.sh" <<'PY'
import json, shlex, sys
m = json.load(open(sys.argv[1]))
print(f"PYTHON_VERSION={shlex.quote(m['python'])}")
print(f"REQUIREMENTS_URI={shlex.quote(m['requirements'])}")
for r in m["repos"]:
    print(f"REPOS+=({shlex.quote(r['name'])} {shlex.quote(r['archive'])})")
    for rel in r["editable"]:
        print(f"EDITABLE+=({shlex.quote(r['name'] + '/' + rel)})")
    for rel in r["install"]:
        print(f"INSTALL+=({shlex.quote(r['name'] + '/' + rel)})")
PY
REPOS=(); EDITABLE=(); INSTALL=()
source "$ROOT/env.sh"
gcloud storage cp "$REQUIREMENTS_URI" "$ROOT/requirements.txt"
for ((i = 0; i < ${#REPOS[@]}; i += 2)); do
    mkdir -p "$ROOT/src/${REPOS[i]}"
    gcloud storage cat "${REPOS[i + 1]}" | tar -xz -C "$ROOT/src/${REPOS[i]}"
done
uv venv --python "$PYTHON_VERSION" "$ROOT/venv"
uv pip install --python "$ROOT/venv/bin/python" -r "$ROOT/requirements.txt"
for rel in ${EDITABLE[@]+"${EDITABLE[@]}"}; do
    uv pip install --python "$ROOT/venv/bin/python" --no-deps -e "$ROOT/src/$rel"
done
for rel in ${INSTALL[@]+"${INSTALL[@]}"}; do
    uv pip install --python "$ROOT/venv/bin/python" --no-deps "$ROOT/src/$rel"
done
"$ROOT/venv/bin/python" -c "import tabflow_slurm"
cp "$ROOT/env.json" "$ROOT/env.done"
"""
