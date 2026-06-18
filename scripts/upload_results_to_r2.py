"""Cache local TabArena run artifacts and upload them to Cloudflare R2 (maintainer tool).

You ran some methods on TabArena and have their results locally (or as a downloadable zip). Edit
``DEST`` + ``SPECS`` below, then run::

    python scripts/upload_results_to_r2.py        # cache locally only (rehearsal)
    UPLOAD=1 python scripts/upload_results_to_r2.py   # cache + push to R2

For each spec, this caches the full artifact tree (raw -> processed -> results, + HPO trajectories
for ``config`` methods) into ``~/.cache/tabarena`` and uploads it to::

    s3://{bucket}/{prefix}/artifacts/{artifact_name}/methods/{method}/

``source`` may be a local results directory, a local ``.zip``, or an ``http(s)://`` URL to a ``.zip``.
Identity comes from a registered ``model`` name (e.g. ``"LimiX"``), a full ``MethodMetadata`` (e.g.
``from tabarena.models.limix.info import limix_method_metadata``), or — if neither — is inferred
from the raw results (then supply ``name`` / ``artifact_name`` / ``model_key``).

R2 credentials (only needed to upload), from https://dash.cloudflare.com/ -> R2 Object Storage::

    export R2_ACCOUNT_ID=...  R2_ACCESS_KEY_ID=...  R2_SECRET_ACCESS_KEY=...
"""

from __future__ import annotations

import copy
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

from tabarena.models._method_metadata import MethodMetadata
from tabarena.nips2025_utils.artifacts.download_utils import download_and_extract_zip
from tabarena.nips2025_utils.end_to_end_single import EndToEndSingle


@dataclass
class UploadSpec:
    """One method's results location plus how to identify it."""

    source: str | Path  # local dir, local .zip, or http(s):// URL to a .zip of the raw results
    model: str | MethodMetadata | None = None  # registered name, full metadata, or None to infer
    name: str | None = None  # inference only: cache dir `.../methods/{name}/`
    artifact_name: str | None = None  # inference only: cache group `.../artifacts/{artifact_name}/`
    model_key: str | None = None  # inference only: ag_key family used in portfolio simulation


# ---- configure here ------------------------------------------------------------------------------
BUCKET, PREFIX, PUBLIC = "tabarena", "cache", True  # R2 destination (same for every spec)
STAGING = Path("local_data")  # where url/zip sources are extracted
UPLOAD = bool(os.environ.get("UPLOAD"))  # safety: only push to R2 when UPLOAD is set

SPECS: list[UploadSpec] = [
    # UploadSpec("/data/runs/limix", model="LimiX"),                       # by registered name
    # UploadSpec("/data/runs/limix.zip", model=limix_method_metadata),    # by imported metadata
    # UploadSpec("https://host/my_method.zip", name="MyMethod_GPU",        # infer from raw
    #            model_key="MYMODEL_GPU", artifact_name="tabarena-2026-06-18"),
]
# ------------------------------------------------------------------------------------------------


def _metadata(spec: UploadSpec) -> MethodMetadata | None:
    """Resolve a spec's identity to a MethodMetadata (deep-copied), or None to infer from the raw."""
    if isinstance(spec.model, MethodMetadata):
        return copy.deepcopy(spec.model)
    if isinstance(spec.model, str):
        from tabarena.models.utils import get_model_info_from_name

        return copy.deepcopy(get_model_info_from_name(spec.model).method_metadata)
    return None


def _raw_dir(source: str | Path) -> Path:
    """Resolve ``source`` (local dir / local .zip / http(s) .zip URL) to a local raw-results dir."""
    text = str(source)
    if text.startswith(("http://", "https://")):
        target = STAGING / Path(text.split("?", 1)[0]).stem
        if not _populated(target):
            download_and_extract_zip(url=text, path_local=target)
        return target
    path = Path(source).expanduser()
    if path.is_dir():
        return path
    if path.suffix == ".zip":
        target = STAGING / path.stem
        if not _populated(target):
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path) as zip_file:
                zip_file.extractall(target)
        return target
    raise ValueError(f"source must be a dir, a .zip, or an http(s) URL to a .zip; got {source!r}")


def _populated(path: Path) -> bool:
    """True if ``path`` is a non-empty directory (a staging target we can reuse as-is)."""
    return path.is_dir() and any(path.iterdir())


def upload_spec(spec: UploadSpec) -> None:
    """Cache ``spec``'s artifacts locally, then (when ``UPLOAD``) push them to R2."""
    md = _metadata(spec)
    if md is not None and md.path_results_exists:
        print(f"  already cached -> {md.path}")
    else:
        md = EndToEndSingle.from_path_raw(
            path_raw=_raw_dir(spec.source),
            method_metadata=md,
            # method/model_key/artifact_name are ignored when method_metadata is given.
            method=None if md is not None else spec.name,
            model_key=None if md is not None else spec.model_key,
            artifact_name=None if md is not None else spec.artifact_name,
            cache=True,
            cache_raw=True,
            cache_hpo_trajectories=True,  # so upload_all() finds the trajectories for config methods
        ).method_metadata
        print(f"  cached -> {md.path}")

    # Stamp the shared R2 destination so every method lands in the same place over R2.
    md.cache_type, md.s3_bucket, md.s3_prefix, md.upload_as_public = "r2", BUCKET, PREFIX, PUBLIC
    if UPLOAD:
        md.to_yaml()  # persist the destination into the cached metadata.yaml
        md.method_uploader(cache_type="r2").upload_all()
        print(f"  uploaded -> s3://{BUCKET}/{PREFIX}/{md.relative_to_cache_root(md.path)}")


if __name__ == "__main__":
    if not SPECS:
        raise SystemExit("Edit the SPECS list near the top of this script first.")
    STAGING.mkdir(parents=True, exist_ok=True)
    for i, spec in enumerate(SPECS, start=1):
        print(f"[{i}/{len(SPECS)}] {spec.model or spec.name or spec.source}")
        upload_spec(spec)
    print(f"\nDone ({len(SPECS)} method(s); {'uploaded' if UPLOAD else 'cache-only — set UPLOAD=1 to push'}).")
