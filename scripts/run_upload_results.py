"""Upload already-generated method artifacts to the method's configured object store.

Companion to ``scripts/run_process_local_raw_data.py``: that script processes raw results into a
local cache (``metadata.yaml`` + ``processed/`` + ``results/`` under the TabArena cache root); this
one uploads those already-cached artifacts. It assumes the artifacts and the method's
``MethodMetadata`` already exist locally — it does not generate anything.

The destination backend is whatever the method's ``MethodMetadata.cache_type`` says (``"s3"`` or
``"r2"``); this script is not specific to any one backend — it just calls
``method_metadata.method_uploader()`` (``cache_type="auto"``).

Before uploading, every part that will be uploaded is verified to exist locally and be non-empty
(see ``plan_and_verify_upload``); the upload is aborted with a clear, aggregated error if anything
is missing, so a method is never partially uploaded.

Set ``dry_run=True`` (the default in ``__main__``) to run the verification and print what would be
uploaded and where, without uploading anything and without needing credentials.

Credentials are required only for the real upload, and depend on the backend: ``"r2"`` reads
``R2_ACCOUNT_ID`` / ``R2_ACCESS_KEY_ID`` / ``R2_SECRET_ACCESS_KEY`` from the environment, while
``"s3"`` uses ambient AWS credentials. ``method_uploader`` raises a descriptive error if the
required credentials are missing.

Specify the methods to upload either by importing a ``MethodMetadata`` from a model's ``info.py``,
or by loading one from the local cache with
``MethodMetadata.from_yaml(method=..., artifact_name=...)``.
"""

from __future__ import annotations

from pathlib import Path

# Imported at runtime (not under TYPE_CHECKING) so this script can be edited to build the
# `methods` list below via `MethodMetadata.from_yaml(...)`.
from tabarena.models._method_metadata import MethodMetadata  # noqa: TC001


def _check_file(path: Path, problems: list[str], label: str) -> None:
    """Record a problem if ``path`` is not an existing, non-empty file."""
    if not path.is_file():
        problems.append(f"{label}: missing file '{path}'")
    elif path.stat().st_size == 0:
        problems.append(f"{label}: file is empty '{path}'")


def _dir_has_files(path: Path) -> bool:
    """True if ``path`` is a directory containing at least one (recursive) file."""
    return path.is_dir() and any(p.is_file() for p in path.rglob("*"))


def plan_and_verify_upload(method_metadata: MethodMetadata, *, upload_raw: bool = False) -> list[str]:
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

    # raw — only when explicitly requested; the dir must exist, be non-empty, and contain the
    # per-run results.pkl files that upload_raw zips.
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
            f"Refusing to upload '{method_metadata.method}' (artifact_name="
            f"{method_metadata.artifact_name!r}); {len(problems)} artifact check(s) failed:\n  - "
            + "\n  - ".join(problems)
        )
    return parts


def _destination(method_metadata: MethodMetadata) -> str:
    """The remote location this method would upload to, computed from the metadata alone (no client).

    Uses ``s3://bucket/prefix`` semantics, which is what both the S3 and R2 backends use.
    """
    ck = method_metadata.cache_kwargs
    root = f"s3://{ck.get('bucket')}/{ck.get('prefix')}"
    try:
        rel = method_metadata.relative_to_cache_root(method_metadata.path).as_posix()
        return f"{root}/{rel}"
    except ValueError:
        # path not under the global cache root (e.g. a flat cache_root override)
        return f"{root}/ (method={method_metadata.method})"


def upload_method(
    method_metadata: MethodMetadata,
    *,
    upload_raw: bool = False,
    dry_run: bool = False,
) -> None:
    """Verify then upload one method's cached artifacts to its configured backend.

    The backend follows ``method_metadata.cache_type`` (``method_uploader(cache_type="auto")``).
    All local checks run first (before the client is created), so missing/empty artifacts are
    reported without needing credentials and without starting a partial upload.

    ``dry_run=True`` runs the same verification and prints what *would* be uploaded and where, but
    performs no upload and does not construct the client — so it needs no credentials.
    """
    parts = plan_and_verify_upload(method_metadata, upload_raw=upload_raw)

    if dry_run:
        print(
            f"[dry-run] '{method_metadata.method}' (artifact={method_metadata.artifact_name}) "
            f"verified OK; would upload parts={parts} -> {_destination(method_metadata)} "
            f"(cache_type={method_metadata.cache_type})"
        )
        if not method_metadata.has_remote_cache:
            print(
                f"[dry-run]   WARNING: cache_type={method_metadata.cache_type!r} has no remote store; "
                "a real upload would fail."
            )
        return

    uploader = method_metadata.method_uploader()
    print(
        f"Uploading '{method_metadata.method}' (artifact={method_metadata.artifact_name}) "
        f"to s3://{uploader.bucket}/{uploader.prefix} | cache_type={method_metadata.cache_type} | parts={parts}"
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


if __name__ == "__main__":
    # When True, verify + print what would be uploaded (and where) without uploading anything.
    # No credentials are needed for a dry run. Flip to False to perform the real upload.
    dry_run = True

    # raw/ is large and usually not cached locally; flip on only if you intend to upload it too.
    upload_raw = False

    # Methods to upload. Either import a fully-specified MethodMetadata from a model's info.py:
    #     from tabarena.models.tabpfn_3.info import tabpfn_3_method_metadata
    #     methods = [tabpfn_3_method_metadata]
    # or load one from the local cache by (method, artifact_name):
    #     MethodMetadata.from_yaml(method="TabPFN-3", artifact_name="tabarena-2026-05-13")
    methods: list[MethodMetadata] = [
        # MethodMetadata.from_yaml(method="...", artifact_name="..."),
    ]

    if len(methods) == 0:
        raise AssertionError("Populate `methods` with one or more MethodMetadata to upload. Currently empty.")

    print(f"{'[dry-run] ' if dry_run else ''}Uploading {len(methods)} method(s) (upload_raw={upload_raw})")
    for i, method_metadata in enumerate(methods):
        print(f"\n({i + 1}/{len(methods)}) {method_metadata.method}")
        upload_method(method_metadata, upload_raw=upload_raw, dry_run=dry_run)
