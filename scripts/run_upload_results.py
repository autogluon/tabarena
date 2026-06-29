"""Upload one method's already-generated artifacts to its configured object store (r2).

Companion to ``scripts/run_process_method.py``: that script processes raw results into a local cache
(``metadata.yaml`` + ``processed/`` + ``results/`` under the TabArena cache root); this one uploads
those already-cached artifacts for a single method. It assumes the artifacts and the method's
``MethodMetadata`` already exist locally — it does not generate anything.

This script only supports the **r2** backend. ``MethodMetadata.method_uploader`` also implements
``"s3"``; s3 support could be enabled here in the future by adding it to ``SUPPORTED_CACHE_TYPES``
(it is intentionally left out for now). The method's ``MethodMetadata`` is verified before any work
(see :func:`verify_upload_metadata` and :func:`plan_and_verify_upload`): its ``cache_type`` /
``cache_kwargs`` must describe a supported remote store, and every part to be uploaded must exist
locally and be non-empty — otherwise the upload is aborted with a clear, aggregated error, so a
method is never partially uploaded.

``--dry-run`` (the default) runs the verification and prints what would be uploaded and where,
without uploading anything and without needing credentials. Pass ``--no-dry-run`` to perform the
real upload; r2 then reads ``R2_ACCOUNT_ID`` / ``R2_ACCESS_KEY_ID`` / ``R2_SECRET_ACCESS_KEY`` from
the environment (``method_uploader`` raises a descriptive error if they are missing).

Specify the method either by a dotted reference to a committed ``MethodMetadata`` (a model's
``info.py``) or by loading one from the local cache by ``(method, suite)``:

    # dry-run (default) — verify + print what would be uploaded:
    python scripts/run_upload_results.py --method-metadata tabarena.models.nori.info:nori_method_metadata

    # real upload from the local cache (raw uploaded by default; --no-upload-raw to skip):
    python scripts/run_upload_results.py --from-cache "TabPFN-3" "tabarena-2026-05-13" --no-dry-run
"""

from __future__ import annotations

import argparse
import importlib
import os
import shlex
import sys
from pathlib import Path

from tabarena.models._method_metadata import MethodMetadata

# Backends this script will upload to. "s3" is implemented in MethodMetadata.method_uploader and can
# be enabled here in the future by adding it below; intentionally r2-only for now.
SUPPORTED_CACHE_TYPES = ("r2",)

# R2 credentials are read from the environment by MethodMetadata.method_uploader (never passed as
# CLI flags, which would leak secrets into shell history and the process table).
R2_ENV_VARS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")


def _suggested_upload_command(args: argparse.Namespace) -> str:
    """A copy-paste command for the real upload, with R2 credentials supplied inline as env vars.

    Reconstructs this invocation with ``--no-dry-run`` and prefixes it with the three R2 env vars as
    ``<placeholder>`` values. They are passed as a shell env-var prefix (not ``--flags``) so real
    secrets stay out of the process ``argv``; ``export``-ing them or using a ``.env`` file (so they
    also stay out of shell history) is preferable for repeated runs.
    """
    invocation = ["python", sys.argv[0]]
    if args.method_metadata:
        invocation += ["--method-metadata", args.method_metadata]
    else:
        invocation += ["--from-cache", *args.from_cache]
    if not args.upload_raw:
        invocation.append("--no-upload-raw")
    invocation.append("--no-dry-run")
    command = " ".join(shlex.quote(tok) for tok in invocation)
    creds = " ".join(f"{var}=<{var.removeprefix('R2_').lower()}>" for var in R2_ENV_VARS)
    return f"{creds} {command}"


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


def verify_upload_metadata(method_metadata: MethodMetadata) -> None:
    """Verify the ``MethodMetadata`` describes an upload destination this script supports.

    Checks that ``cache_type`` is one of :data:`SUPPORTED_CACHE_TYPES` (``"r2"``) and that
    ``cache_kwargs`` carries the remote location (``bucket`` + ``prefix``). ``__post_init__`` already
    enforces the remote location for r2/s3, but this re-checks it to give a precise, actionable
    message. Raises ``ValueError`` listing every problem at once (no partial upload is attempted).
    """
    problems: list[str] = []
    cache_type = method_metadata.cache_type

    if cache_type not in SUPPORTED_CACHE_TYPES:
        if cache_type == "s3":
            problems.append(
                "cache_type='s3' is not supported by this script yet (only 'r2' is). s3 upload "
                "exists in MethodMetadata.method_uploader and can be enabled by adding 's3' to "
                "SUPPORTED_CACHE_TYPES."
            )
        elif cache_type in (None, "local"):
            problems.append(
                f"cache_type={cache_type!r} has no remote store to upload to; set cache_type='r2' "
                f"and cache_kwargs={{'bucket': ..., 'prefix': ...}} on the MethodMetadata."
            )
        else:
            problems.append(
                f"cache_type={cache_type!r} is unsupported; this script only uploads to {SUPPORTED_CACHE_TYPES}."
            )
    else:
        # Remote location is only meaningful (and required) for a supported remote backend.
        cache_kwargs = method_metadata.cache_kwargs or {}
        for key in ("bucket", "prefix"):
            if not cache_kwargs.get(key):
                problems.append(
                    f"cache_kwargs is missing '{key}' (got cache_kwargs={cache_kwargs!r}); "
                    f"required for a {cache_type!r} upload."
                )

    if problems:
        raise ValueError(
            f"Cannot upload '{method_metadata.method}' (suite={method_metadata.suite!r}); "
            f"{len(problems)} metadata check(s) failed:\n  - " + "\n  - ".join(problems)
        )


def _check_file(path: Path, problems: list[str], label: str) -> None:
    """Record a problem if ``path`` is not an existing, non-empty file."""
    if not path.is_file():
        problems.append(f"{label}: missing file '{path}'")
    elif path.stat().st_size == 0:
        problems.append(f"{label}: file is empty '{path}'")


def _dir_has_files(path: Path) -> bool:
    """True if ``path`` is a directory containing at least one (recursive) file."""
    return path.is_dir() and any(p.is_file() for p in path.rglob("*"))


def plan_and_verify_upload(method_metadata: MethodMetadata, *, upload_raw: bool = True) -> list[str]:
    """Decide which parts of the method to upload and verify each is present and non-empty.

    Returns the ordered list of part names to upload (a subset of ``metadata``, ``results``,
    ``processed``, ``hpo_trajectories``, ``raw``). ``metadata`` is always included and is
    generated in memory from the ``MethodMetadata`` (so there is nothing to check on disk);
    ``results`` is always required. ``processed`` and ``hpo_trajectories`` are uploaded when
    present locally, and ``raw`` only when ``upload_raw=True``. Raises ``FileNotFoundError`` —
    listing every problem at once — if any planned part is missing or empty, so we never start a
    partial upload.
    """
    problems: list[str] = []
    parts = ["metadata"]

    # results — always uploaded and required (the per-task result parquet files for this
    # method_type: model/hpo for configs, model for baselines, portfolio for portfolios).
    for path_local in method_metadata.path_results_files():
        _check_file(path_local, problems, "results")
    parts.append("results")

    # processed — uploaded when cached; upload_processed also (re)uploads the processed dir's
    # configs_hyperparameters.json, so verify both.
    if method_metadata.path_processed_exists:
        if not _dir_has_files(method_metadata.path_processed):
            problems.append(f"processed: directory is empty '{method_metadata.path_processed}'")
        _check_file(
            method_metadata.path_configs_hyperparameters(),
            problems,
            "processed/configs_hyperparameters.json",
        )
        parts.append("processed")

    # hpo trajectories — config methods only, when the trajectory file was generated/cached.
    if method_metadata.method_type == "config" and method_metadata.path_results_hpo_trajectories().exists():
        _check_file(method_metadata.path_results_hpo_trajectories(), problems, "hpo_trajectories")
        parts.append("hpo_trajectories")

    # raw — uploaded by default; the dir must exist, be non-empty, and contain the per-run
    # results.pkl files that upload_raw zips.
    if upload_raw:
        path_raw = method_metadata.path_raw
        if not _dir_has_files(path_raw):
            problems.append(f"raw: directory missing or empty '{path_raw}'")
        elif not any(path_raw.rglob("results.pkl")):
            problems.append(f"raw: no 'results.pkl' files found under '{path_raw}'")
        else:
            parts.append("raw")

    if problems:
        raise FileNotFoundError(
            f"Refusing to upload '{method_metadata.method}' (suite="
            f"{method_metadata.suite!r}); {len(problems)} artifact check(s) failed:\n  - " + "\n  - ".join(problems)
        )
    return parts


def _destination(method_metadata: MethodMetadata) -> str:
    """The remote location this method would upload to, computed from the metadata alone (no client).

    Rendered as ``<cache_type>://bucket/prefix/...`` (e.g. ``r2://...``) so the scheme reflects the
    actual backend rather than always reading ``s3://``.
    """
    ck = method_metadata.cache_kwargs
    root = f"{method_metadata.cache_type}://{ck.get('bucket')}/{ck.get('prefix')}"
    try:
        rel = method_metadata.relative_to_cache_root(method_metadata.path).as_posix()
        return f"{root}/{rel}"
    except ValueError:
        # path not under the global cache root (e.g. an explicit artifact_dir override)
        return f"{root}/ (method={method_metadata.method})"


def upload_method(
    method_metadata: MethodMetadata,
    *,
    upload_raw: bool = True,
    dry_run: bool = False,
) -> None:
    """Verify then upload one method's cached artifacts to its configured r2 backend.

    All verification runs first (before the client is created): :func:`verify_upload_metadata`
    confirms the ``cache_type`` / ``cache_kwargs`` describe a supported remote store, then
    :func:`plan_and_verify_upload` confirms every planned artifact is present and non-empty. So a
    misconfigured or incomplete method is reported without needing credentials and without starting
    a partial upload.

    ``dry_run=True`` runs the same verification and prints what *would* be uploaded and where, but
    performs no upload and does not construct the client — so it needs no credentials.
    """
    verify_upload_metadata(method_metadata)
    parts = plan_and_verify_upload(method_metadata, upload_raw=upload_raw)

    if dry_run:
        print(
            f"[dry-run] '{method_metadata.method}' (artifact={method_metadata.suite}) "
            f"verified OK; would upload parts={parts} -> {_destination(method_metadata)}"
        )
        return

    uploader = method_metadata.method_uploader()
    print(
        f"Uploading '{method_metadata.method}' (artifact={method_metadata.suite}) "
        f"to {method_metadata.cache_type}://{uploader.bucket}/{uploader.prefix} | parts={parts}"
    )
    uploader.upload_metadata()
    uploader.upload_results()
    if "processed" in parts:
        uploader.upload_processed()
    if "hpo_trajectories" in parts:
        uploader.upload_results_hpo_trajectories()
    if "raw" in parts:
        uploader.upload_raw()
    print(f"\tDone: {method_metadata.method}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify + upload a single method's cached artifacts to its r2 store.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--method-metadata",
        metavar="MODULE:ATTR",
        help="Dotted reference to a committed MethodMetadata (e.g. 'tabarena.models.nori.info:nori_method_metadata').",
    )
    source.add_argument(
        "--from-cache",
        nargs=2,
        metavar=("METHOD", "SUITE"),
        help="Load the MethodMetadata from the local cache via MethodMetadata.from_yaml(method, suite).",
    )
    parser.add_argument(
        "--upload-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload the raw `results.pkl` files too (on by default; pass --no-upload-raw to skip).",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify + print what would be uploaded without uploading (on by default; "
        "pass --no-dry-run to perform the real upload).",
    )
    args = parser.parse_args()

    if args.method_metadata:
        method_metadata = _load_method_metadata(args.method_metadata)
    else:
        method, suite = args.from_cache
        method_metadata = MethodMetadata.from_yaml(method=method, suite=suite)

    print(
        f"{'[dry-run] ' if args.dry_run else ''}Uploading '{method_metadata.method}' "
        f"(suite={method_metadata.suite}, upload_raw={args.upload_raw})"
    )
    try:
        upload_method(method_metadata, upload_raw=args.upload_raw, dry_run=args.dry_run)
    except OSError as e:
        # Missing R2 credentials (raised by MethodMetadata.method_uploader, with where-to-find-them
        # instructions). Re-surface those, then append the exact command to re-run with the
        # credentials supplied as env vars.
        if all(os.environ.get(var) for var in R2_ENV_VARS):
            raise  # an unrelated OSError — leave it untouched
        raise SystemExit(
            f"{e}\n\nThen re-run with the credentials set as env vars:\n  {_suggested_upload_command(args)}"
        ) from e

    if args.dry_run:
        print(
            "\nTo upload for real, re-run with your R2 credentials set as env vars and --no-dry-run:\n"
            f"  {_suggested_upload_command(args)}"
        )
        missing = [var for var in R2_ENV_VARS if not os.environ.get(var)]
        if missing:
            print(
                f"\nR2 credentials are not set yet ({', '.join(missing)}). How to obtain them:\n"
                + MethodMetadata.r2_credentials_help()
            )


if __name__ == "__main__":
    main()
