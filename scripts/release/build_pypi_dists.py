"""Build the PyPI distributions of ``bencheval`` and ``tabarena``.

The pyproject files in the repository describe the git-based install, which stays the recommended
way to use TabArena. A PyPI build differs in a few controlled ways. They are applied to a staged
copy of each package under a temporary directory, so the checkout itself is never modified:

1. Both packages receive the same version, and ``tabarena`` pins ``bencheval==<version>``, so a
   PyPI release is always a matching pair.
2. Requirements with a direct URL (``name @ git+https://...``) are removed from ``tabarena``'s
   extras because PyPI rejects distributions that carry them. The extra itself stays, empty. The
   model's ``pip_extra`` in its ``info.py`` still names the exact git dependency, so the install
   hint users see is unchanged.
3. The root ``README.md`` (relative links rewritten to absolute GitHub URLs) and ``LICENSE`` are
   copied next to ``tabarena``'s pyproject, and ``LICENSE`` next to ``bencheval``'s, so the wheels
   carry a description and a license file without duplicating either file in git.

After building, every wheel is checked: no direct URLs, a non-empty description, a license file,
matching versions, the ``bencheval`` pin, and every git-tracked non-Python file under ``src/``
present in the wheel.

Usage, from the repository root (``uv`` must be on PATH)::

    python scripts/release/build_pypi_dists.py                     # version from pyproject
    python scripts/release/build_pypi_dists.py --version 0.1.0a1   # explicit version
    python scripts/release/build_pypi_dists.py --dev               # <next patch>.dev<UTC timestamp>
    python scripts/release/build_pypi_dists.py --check-tag v0.1.0  # fail unless pyproject == tag

Distributions land in ``dist/`` (``--out-dir``). The GitHub workflows ``release.yml``,
``prerelease.yml`` and ``dev-release.yml`` call this script and then run ``uv publish``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("bencheval", "tabarena")
GITHUB_BLOB_URL = "https://github.com/autogluon/tabarena/blob/main/"

# The `[project]` version line, keeping any trailing comment (`version = "0.1.0"  # keep equal to ...`).
_VERSION_LINE_RE = re.compile(r'^version = "(?P<version>[^"]+)"(?P<rest>[ \t]*(?:#[^\n]*)?)$', re.MULTILINE)
_PEP440_RE = re.compile(r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$")
_PRERELEASE_RE = re.compile(r"(a|b|rc|\.dev)\d+$")
_PLAIN_RELEASE_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
# A double-quoted requirement string with a direct URL, plus the comma that follows it (if any).
# Works for one-per-line arrays and for inline arrays such as `tabfm = ["tabfm @ git+..."]`.
_DIRECT_URL_TOKEN_RE = re.compile(r'"(?P<req>[^"\n]+ @ [^"\n]+)"[ \t]*,?')
_BENCHEVAL_REQ_RE = re.compile(r'"bencheval(?P<extras>\[[^\]]*\])?"')
_MARKDOWN_LINK_RE = re.compile(r"(?P<prefix>\]\()(?P<target>[^)\s]+)")
_GITHUB_ALERT_RE = re.compile(r"^> \[!(?P<kind>NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]", re.MULTILINE)


def set_version(pyproject_text: str, version: str) -> str:
    """Return ``pyproject_text`` with its single ``[project]`` ``version`` line set to ``version``."""
    matches = list(_VERSION_LINE_RE.finditer(pyproject_text))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one top-level `version = ...` line, found {len(matches)}")
    return _VERSION_LINE_RE.sub(rf'version = "{version}"\g<rest>', pyproject_text)


def pin_bencheval(pyproject_text: str, version: str) -> str:
    """Pin every ``bencheval`` / ``bencheval[extra]`` requirement to ``==version``."""
    text, n = _BENCHEVAL_REQ_RE.subn(lambda m: f'"bencheval{m.group("extras") or ""}=={version}"', pyproject_text)
    if n == 0:
        raise SystemExit("no `bencheval` requirement found to pin")
    return text


def strip_direct_url_requirements(pyproject_text: str) -> tuple[str, list[str]]:
    """Remove requirement strings with a direct URL (``name @ url``) and return them.

    A TOML parse afterwards verifies that no direct URL survives, which also catches requirement
    strings written in a form the regex does not handle (e.g. single-quoted literal strings).
    """
    removed = [m.group("req") for m in _DIRECT_URL_TOKEN_RE.finditer(pyproject_text)]
    text = _DIRECT_URL_TOKEN_RE.sub("", pyproject_text)
    leftover = [req for req in _all_requirements(text) if " @ " in req]
    if leftover:
        raise SystemExit(f"direct-URL requirements survived stripping (unexpected TOML layout): {leftover}")
    return text, removed


def _all_requirements(pyproject_text: str) -> list[str]:
    project = tomllib.loads(pyproject_text)["project"]
    reqs = list(project.get("dependencies", []))
    for extra_reqs in project.get("optional-dependencies", {}).values():
        reqs.extend(extra_reqs)
    return reqs


def rewrite_readme_for_pypi(readme_text: str, base_url: str = GITHUB_BLOB_URL) -> str:
    """Make the GitHub README render on PyPI: absolute link targets and plain alert blockquotes."""

    def _absolutize(match: re.Match) -> str:
        target = match.group("target")
        if target.startswith(("http://", "https://", "mailto:", "#")):
            return match.group(0)
        return f"{match.group('prefix')}{base_url}{target.removeprefix('./')}"

    text = _MARKDOWN_LINK_RE.sub(_absolutize, readme_text)
    return _GITHUB_ALERT_RE.sub(lambda m: f"> **{m.group('kind').title()}**", text)


def compute_dev_version(base_version: str, now: datetime | None = None) -> str:
    """``X.Y.(Z+1).dev<UTC timestamp>``: sorts after ``X.Y.Z``, before ``X.Y.(Z+1)``, monotonic across commits."""
    match = _PLAIN_RELEASE_RE.match(base_version)
    if match is None:
        raise SystemExit(f"pyproject version {base_version!r} is not a plain X.Y.Z; refusing to derive a dev version")
    major, minor, patch = (int(part) for part in match.groups())
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d%H%M%S")
    return f"{major}.{minor}.{patch + 1}.dev{stamp}"


def prepare_pyproject_text(package: str, pyproject_text: str, version: str) -> tuple[str, list[str]]:
    """Apply the PyPI-build edits for ``package`` and return the new text plus the stripped requirements."""
    text = set_version(pyproject_text, version)
    removed: list[str] = []
    if package == "tabarena":
        text = pin_bencheval(text, version)
        text, removed = strip_direct_url_requirements(text)
    return text, removed


def repo_version() -> str:
    """The version shared by both pyproject files; fails if they disagree."""
    versions = {
        package: tomllib.loads((REPO_ROOT / "packages" / package / "pyproject.toml").read_text())["project"]["version"]
        for package in PACKAGES
    }
    if len(set(versions.values())) != 1:
        raise SystemExit(f"package versions must match, got {versions}")
    return next(iter(versions.values()))


def stage_package(package: str, stage_root: Path, version: str) -> Path:
    """Copy ``packages/<package>`` into ``stage_root`` and apply the PyPI-build edits to the copy."""
    pkg_dir = stage_root / package
    shutil.copytree(
        REPO_ROOT / "packages" / package,
        pkg_dir,
        ignore=shutil.ignore_patterns("*.egg-info", "__pycache__", "*.so", "build", "dist"),
    )
    pyproject = pkg_dir / "pyproject.toml"
    text, removed = prepare_pyproject_text(package, pyproject.read_text(), version)
    pyproject.write_text(text)
    for req in removed:
        print(f"  [{package}] stripped direct-URL requirement: {req}")
    shutil.copy(REPO_ROOT / "LICENSE", pkg_dir / "LICENSE")
    if package == "tabarena":
        (pkg_dir / "README.md").write_text(rewrite_readme_for_pypi((REPO_ROOT / "README.md").read_text()))
    return pkg_dir


def build_package(pkg_dir: Path, out_dir: Path) -> None:
    """Run ``uv build`` for a staged package; the build log is shown only on failure."""
    cmd = ["uv", "build", str(pkg_dir), "--out-dir", str(out_dir)]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)  # noqa: S603
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"`{' '.join(cmd)}` failed with exit code {result.returncode}")


def wheel_metadata(wheel: Path) -> tuple[dict[str, list[str]], str]:
    """Return the wheel's METADATA as (header fields, each a list of values; long-description body)."""
    with zipfile.ZipFile(wheel) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        raw = zf.read(name).decode()
    header, _, body = raw.partition("\n\n")
    fields: dict[str, list[str]] = {}
    for line in header.splitlines():
        key, _, value = line.partition(": ")
        fields.setdefault(key, []).append(value)
    return fields, body


def tracked_data_files(package: str) -> list[str] | None:
    """Git-tracked non-``.py`` files under ``src/<package>`` as wheel-relative paths; None if git is unavailable."""
    src_root = Path("packages") / package / "src"
    result = subprocess.run(  # noqa: S603
        ["git", "ls-files", "--", str(src_root / package)],  # noqa: S607
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return [
        Path(line).relative_to(src_root).as_posix()
        for line in result.stdout.splitlines()
        if line and not line.endswith(".py")
    ]


def check_wheel(wheel: Path, package: str, version: str, tracked_files: list[str] | None) -> None:
    """Fail with a readable list of problems if ``wheel`` is not fit for PyPI."""
    fields, body = wheel_metadata(wheel)
    problems: list[str] = []
    if fields.get("Version") != [version]:
        problems.append(f"Version is {fields.get('Version')}, expected [{version!r}]")
    requires = fields.get("Requires-Dist", [])
    if direct := [req for req in requires if " @ " in req]:
        problems.append(f"direct-URL requirements (PyPI rejects these): {direct}")
    if not body.strip():
        problems.append("empty long description (README not staged?)")
    if not fields.get("License-File"):
        problems.append("no License-File (LICENSE not staged?)")
    if package == "tabarena":
        bad_pins = [req for req in requires if req.startswith("bencheval") and f"=={version}" not in req]
        if bad_pins:
            problems.append(f"bencheval not pinned to =={version}: {bad_pins}")
    if tracked_files is not None:
        with zipfile.ZipFile(wheel) as zf:
            shipped = set(zf.namelist())
        if missing := sorted(set(tracked_files) - shipped):
            problems.append(f"git-tracked data files missing from the wheel (fix package-data): {missing}")
    if problems:
        raise SystemExit(f"{wheel.name} is not publishable:\n  - " + "\n  - ".join(problems))


def verify_dists(out_dir: Path, version: str) -> list[Path]:
    """Check that ``out_dir`` holds one wheel and one sdist per package and that every wheel passes."""
    files: list[Path] = []
    for package in PACKAGES:
        wheels = sorted(out_dir.glob(f"{package}-*.whl"))
        sdists = sorted(out_dir.glob(f"{package}-*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise SystemExit(f"expected one wheel and one sdist for {package} in {out_dir}, found {wheels + sdists}")
        check_wheel(wheels[0], package, version, tracked_data_files(package))
        files.extend([*wheels, *sdists])
    return files


def resolve_version(args: argparse.Namespace) -> str:
    base = repo_version()
    if args.version:
        if not _PEP440_RE.match(args.version):
            raise SystemExit(f"{args.version!r} is not a canonical PEP 440 version (e.g. 0.1.0, 0.1.0a1, 0.1.0.dev3)")
        version = args.version
    elif args.dev:
        version = compute_dev_version(base)
    else:
        version = base
        if args.check_tag is not None and args.check_tag.removeprefix("v") != base:
            raise SystemExit(f"tag {args.check_tag} does not match the pyproject version {base}")
    if args.require_prerelease and not _PRERELEASE_RE.search(version):
        raise SystemExit(f"{version!r} has no pre-release segment; use the Release workflow (tag v{version}) instead")
    return version


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter
    )
    how = parser.add_mutually_exclusive_group()
    how.add_argument("--version", help="explicit version for both packages (the pyproject files are not modified)")
    how.add_argument(
        "--dev", action="store_true", help="derive <next patch>.dev<UTC timestamp> from the pyproject version"
    )
    how.add_argument(
        "--check-tag", metavar="TAG", help="use the pyproject version; fail unless it equals TAG (v prefix ok)"
    )
    parser.add_argument(
        "--require-prerelease", action="store_true", help="fail if the resolved version is a stable release"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "dist", help="where to write the dists (default: dist/)"
    )
    args = parser.parse_args(argv)

    version = resolve_version(args)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in [p for package in PACKAGES for p in out_dir.glob(f"{package}-*")]:
        print(f"  removing stale {stale.name}")
        stale.unlink()

    print(f"Building bencheval + tabarena {version} into {out_dir}")
    with tempfile.TemporaryDirectory(prefix="tabarena-pypi-") as tmp:
        for package in PACKAGES:
            build_package(stage_package(package, Path(tmp), version), out_dir)
    files = verify_dists(out_dir, version)
    print("Built and verified:")
    for f in files:
        print(f"  {f.relative_to(out_dir)}")
    print(f"version={version}")


if __name__ == "__main__":
    main()
