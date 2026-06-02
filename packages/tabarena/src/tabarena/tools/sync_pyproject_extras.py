"""Compare per-model `ModelInfo.pip_extra` against `pyproject.toml` extras.

Each `tabarena/models/<key>/info.py` declares its pip dependencies via
`ModelInfo.pip_extra`. The `[project.optional-dependencies]` section in
`packages/tabarena/pyproject.toml` should mirror that data — this script flags
drift so the two stay in sync.

Usage:
    python -m tabarena.tools.sync_pyproject_extras [--check]

By default, prints a report comparing declared vs. expected entries.
With `--check`, exits non-zero if any drift is detected (for CI).
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

from tabarena.models import get_model_registry


def _expected_extras() -> dict[str, list[str]]:
    """Aggregate `pip_extra` across MODEL_REGISTRY, keyed by folder name
    (the `<key>` in `tabarena/models/<key>/`). Models that share a folder
    (e.g. tabicl/tabiclv2) and a class have their pip_extras unioned.
    """
    extras: dict[str, set[str]] = defaultdict(set)
    for info in get_model_registry().values():
        # Resolve the package short name from the model class module path.
        # Example: tabarena.models.ebm.model → "ebm"
        # For multi-class folders (tabicl), this lumps both TabICL and
        # TabICLv2 into the "tabicl" extra.
        module = info.model_cls.__module__
        parts = module.split(".")
        if "ag" in parts:
            folder = parts[parts.index("ag") + 1]
        else:
            folder = parts[-2] if len(parts) >= 2 else parts[-1]
        for spec in info.pip_extra:
            extras[folder].add(spec)
    return {k: sorted(v) for k, v in extras.items() if v}


def _pyproject_extras(pyproject_path: Path) -> dict[str, list[str]]:
    data = tomllib.loads(pyproject_path.read_text())
    return data.get("project", {}).get("optional-dependencies", {})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Exit non-zero on drift.")
    parser.add_argument("--pyproject", type=Path,
                        default=Path(__file__).resolve().parents[3] / "pyproject.toml",
                        help="Path to pyproject.toml (default: packages/tabarena/pyproject.toml).")
    args = parser.parse_args()

    expected = _expected_extras()
    declared = _pyproject_extras(args.pyproject)

    drifted = False
    print(f"Comparing ModelInfo.pip_extra against {args.pyproject}\n")
    print(f"{'folder':<24} {'status':<10} details")
    print("-" * 78)

    all_keys = sorted(set(expected) | (set(declared) & {k for k in declared if not k.startswith("benchmark")}))
    for key in all_keys:
        exp = expected.get(key)
        dec = declared.get(key)
        if exp is None and dec is None:
            continue
        if exp == dec:
            status = "OK"
        else:
            status = "DRIFT"
            drifted = True
        print(f"{key:<24} {status:<10} expected={exp!r}  declared={dec!r}")

    print()
    if drifted:
        print("FAIL: pip_extra drift detected. Update either the per-model `info.py` or `pyproject.toml`.")
        return 1 if args.check else 0
    print("OK: pip_extra matches pyproject.toml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
