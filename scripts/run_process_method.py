"""Generic single-method entry point: inspect / verify / process one local raw-results directory.

A parameterized counterpart to the pinned ``run_process_<run>.py`` scripts (e.g.
``run_process_nori_regression_18062026.py``): instead of hardcoding the run directory and the model's
``MethodMetadata`` import, pass them on the command line. There is **no download and no upload**; this
is a thin CLI wrapper around :func:`tabarena.tools.process_local_raw_data.process_method`.

``path_raw`` is searched recursively for ``results.pkl`` files, so point it at the run's ``data/``
directory (or any parent of the raw artifacts). Processing requires an explicit ``MethodMetadata``,
passed as a dotted ``module:attr`` (or ``module.attr``) reference to a committed one.

Examples:
--------
    # Inspect only — print the inferred fields + a suggested MethodMetadata snippet:
    python scripts/run_process_method.py /path/to/run/data

    # Verify + process, using a committed MethodMetadata (raw + HPO-trajectory caching are on by
    # default; pass --no-cache-raw / --no-cache-hpo-trajectories to skip):
    python scripts/run_process_method.py /path/to/run/data \
        --method-metadata tabarena.models.nori.info:nori_method_metadata --process
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from tabarena.models._method_metadata import MethodMetadata
from tabarena.tools.process_local_raw_data import RawMethod, process_method


def _load_method_metadata(ref: str) -> MethodMetadata:
    """Import a ``MethodMetadata`` from a ``module.path:attr`` (or ``module.path.attr``) reference."""
    module_path, _, attr = ref.partition(":") if ":" in ref else ref.rpartition(".")
    if not module_path or not attr:
        raise ValueError(
            f"Invalid --method-metadata reference {ref!r}; expected 'module.path:attr' "
            f"(e.g. 'tabarena.models.nori.info:nori_method_metadata')."
        )
    obj = getattr(importlib.import_module(module_path), attr)
    if not isinstance(obj, MethodMetadata):
        raise TypeError(f"{ref!r} resolved to {type(obj).__name__}, not a MethodMetadata.")
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect / verify / process a single local raw-results directory (no upload).",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path_raw",
        type=Path,
        help="Directory of already-present raw `results.pkl` files (searched recursively).",
    )
    parser.add_argument(
        "--method-metadata",
        metavar="MODULE:ATTR",
        help="Dotted reference to a committed MethodMetadata (e.g. "
        "'tabarena.models.nori.info:nori_method_metadata'). Required for --process.",
    )
    parser.add_argument(
        "--no-inspect",
        action="store_true",
        help="Skip the inspect step (inferred fields + suggested-metadata snippet).",
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Build + cache the processed repo and results locally (requires --method-metadata).",
    )
    parser.add_argument(
        "--ignore-metadata-mismatch",
        action="store_true",
        help="Downgrade verification failures to a warning instead of erroring.",
    )
    parser.add_argument(
        "--cache-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy the raw `results.pkl` files into the TabArena cache (on by default; pass --no-cache-raw to skip).",
    )
    parser.add_argument(
        "--cache-hpo-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate + cache HPO trajectories, config methods only (on by default; "
        "pass --no-cache-hpo-trajectories to skip).",
    )
    parser.add_argument(
        "--backend",
        choices=["ray", "native"],
        default="ray",
        help="'ray' (parallel) or 'native' (sequential) for the HPO/model-result simulation.",
    )
    args = parser.parse_args()

    if args.process and not args.method_metadata:
        parser.error("--process requires --method-metadata (processing needs an explicit MethodMetadata).")

    method_metadata = _load_method_metadata(args.method_metadata) if args.method_metadata else None
    method = RawMethod(path_raw=args.path_raw, method_metadata=method_metadata)

    process_method(
        method,
        inspect=not args.no_inspect,
        process=args.process,
        ignore_metadata_mismatch=args.ignore_metadata_mismatch,
        cache_raw=args.cache_raw,
        cache_hpo_trajectories=args.cache_hpo_trajectories,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
