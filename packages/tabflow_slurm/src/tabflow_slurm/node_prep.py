"""Once-per-node-boot page-cache pre-touch of the job's model libraries.

Compute nodes boot with an empty page cache and read the venv from NFS, so the first fit on a
fresh node pays the library reads (torch, catboost, ray and friends) inside the timed fit when
the fit runs in Ray fold workers that no in-process warm-up reaches. ``pretouch`` reads the files
of the given packages once so they are page-cache resident, bounded by a byte budget, and records a
marker so later jobs on the same node skip it. Nothing is imported or executed. Invoked by
``submit_template.sh`` as ``python -P -m tabflow_slurm.node_prep pretouch ...``; it never raises and
always exits 0, a failed pre-touch only costs the benefit.
"""

from __future__ import annotations

import argparse
import os
import sys
import sysconfig
import time
from pathlib import Path

_CHUNK = 8 << 20


def boot_time() -> float | None:
    """The node's boot time from ``/proc/stat`` (``btime``); ``None`` where unavailable."""
    try:
        with Path("/proc/stat").open() as handle:
            for line in handle:
                if line.startswith("btime "):
                    return float(line.split()[1])
    except OSError:
        return None
    return None


def marker_is_fresh(marker: Path, *, purelib: Path | None = None) -> bool:
    """Whether ``marker`` was written after the last boot and after the venv's last change.

    A marker on a disk that survives a reboot would otherwise skip the pre-touch exactly when the
    page cache is empty.
    """
    try:
        stamp = marker.stat().st_mtime
    except OSError:
        return False
    booted = boot_time()
    if booted is not None and stamp < booted:
        return False
    if purelib is not None:
        try:
            if stamp < purelib.stat().st_mtime:
                return False
        except OSError:
            pass
    return True


def package_roots(packages: list[str]) -> list[Path]:
    """The directories (or module files) of ``packages`` without importing them."""
    import importlib.util

    roots: list[Path] = []
    for name in packages:
        try:
            spec = importlib.util.find_spec(name)
        except (ModuleNotFoundError, ValueError):
            spec = None
        if spec is None:
            continue
        if spec.submodule_search_locations:
            roots.extend(Path(location) for location in spec.submodule_search_locations)
        elif spec.origin and spec.origin not in ("built-in", "frozen"):
            roots.append(Path(spec.origin))
    return roots


def read_files(roots: list[Path], *, max_bytes: int) -> tuple[int, int]:
    """Read every regular file under ``roots`` once, stopping at ``max_bytes``; returns (bytes, files)."""
    total = 0
    files = 0
    for root in roots:
        candidates = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in candidates:
            if total >= max_bytes:
                return total, files
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            try:
                while total < max_bytes:
                    chunk = os.read(fd, min(_CHUNK, max_bytes - total))
                    if not chunk:
                        break
                    total += len(chunk)
            except OSError:
                pass
            finally:
                os.close(fd)
            files += 1
    return total, files


def pretouch(*, packages: list[str], paths: list[str], max_bytes: int, marker: Path | None) -> int:
    """Pre-read ``packages`` (import names) and ``paths`` (relative to site-packages) into the page cache.

    Skipped when ``marker`` is fresh (see :func:`marker_is_fresh`); the marker is written after a
    full pass. Returns the number of bytes read (0 when skipped).
    """
    purelib = Path(sysconfig.get_paths()["purelib"])
    if marker is not None and marker_is_fresh(marker, purelib=purelib):
        print(f"pretouch: skipped, marker {marker} is fresh")
        return 0
    roots = package_roots(packages)
    roots.extend(purelib / rel for rel in paths if (purelib / rel).exists())
    started = time.monotonic()
    total, files = read_files(roots, max_bytes=max_bytes)
    elapsed = time.monotonic() - started
    print(f"pretouch: read {total / 1e6:.0f} MB from {files} files in {elapsed:.1f}s (budget {max_bytes / 1e6:.0f} MB)")
    if marker is not None:
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"{total}\n")
        except OSError as exc:
            print(f"pretouch: could not write marker {marker}: {exc}")
    return total


def _split(csv: str) -> list[str]:
    return [item for item in (part.strip() for part in csv.split(",")) if item]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tabflow_slurm.node_prep")
    sub = parser.add_subparsers(dest="command", required=True)
    touch = sub.add_parser("pretouch", help="Read model libraries into the page cache once per node boot.")
    touch.add_argument("--packages", default="", help="Comma-separated import names.")
    touch.add_argument("--paths", default="", help="Comma-separated paths relative to site-packages.")
    touch.add_argument("--max-bytes", type=int, default=2 * 1024**3)
    touch.add_argument("--marker", default=None)
    args = parser.parse_args(argv)
    try:
        pretouch(
            packages=_split(args.packages),
            paths=_split(args.paths),
            max_bytes=args.max_bytes,
            marker=Path(args.marker) if args.marker else None,
        )
    except Exception as exc:  # a failed pre-touch only costs the benefit
        print(f"pretouch: WARNING {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
