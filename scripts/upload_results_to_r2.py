"""Cache local TabArena run artifacts, upload them to Cloudflare R2, and rebuild the website data.

One button: you ran some methods on TabArena and have their results locally (or as a downloadable
zip). Edit the config block (``BUCKET``/``PREFIX``/...) + ``SPECS`` below, then run::

    python scripts/upload_results_to_r2.py                 # cache locally only (rehearsal)
    UPLOAD=1 python scripts/upload_results_to_r2.py        # cache + push to R2
    UPLOAD=1 WEBSITE=1 python scripts/upload_results_to_r2.py   # ... then rebuild website artifacts

For each spec this caches the full artifact tree (raw -> processed -> results, + HPO trajectories
for ``config`` methods) into ``~/.cache/tabarena`` and, with ``UPLOAD``, uploads it to::

    s3://{bucket}/{prefix}/artifacts/{artifact_name}/methods/{method}/

With ``WEBSITE`` it then regenerates the leaderboard/figures with the processed methods included
(as ``extra_methods``) and converts them to the website-ready ``website_data/`` the HF Space serves
(this last step is the merge of ``examples/plots/run_generate_website_artifacts.py``). ``WEBSITE``
works with or without ``UPLOAD`` — handy to preview the site with your method before pushing.

``source`` may be a local results directory, a local ``.zip``, or an ``http(s)://`` URL to a
``.zip``. Identity comes from a registered ``model`` name (e.g. ``"LimiX"``), a full
``MethodMetadata`` (e.g. ``from tabarena.models.limix.info import limix_method_metadata``), or — if
neither — is inferred from the raw results (then supply ``name`` / ``artifact_name`` / ``model_key``).

R2 credentials (only needed to upload), from https://dash.cloudflare.com/ -> R2 Object Storage::

    export R2_ACCOUNT_ID=...  R2_ACCESS_KEY_ID=...  R2_SECRET_ACCESS_KEY=...
"""

from __future__ import annotations

import copy
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from tabarena.models._method_metadata import MethodMetadata
from tabarena.nips2025_utils.artifacts.download_utils import download_and_extract_zip
from tabarena.nips2025_utils.end_to_end_single import EndToEndSingle
from tabarena.nips2025_utils.tabarena_context import TabArenaContext


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
UPLOAD = bool(os.environ.get("UPLOAD"))  # only push to R2 when set
WEBSITE = bool(os.environ.get("WEBSITE"))  # rebuild website artifacts after processing when set
ELO_BOOTSTRAP_ROUNDS = 200  # website: 1 for a quick toy run, 200 for official numbers

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


def process_spec(spec: UploadSpec) -> MethodMetadata:
    """Cache ``spec``'s artifacts locally, then (when ``UPLOAD``) push them to R2; return its metadata."""
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
    return md


def build_website_artifacts(new_methods: list[MethodMetadata]) -> Path:
    """Rebuild the website-ready ``website_data/`` (leaderboards/figures) including ``new_methods``.

    Methods whose name is already in the official TabArena set are shown from their registered
    artifacts; only genuinely new names are added on top via ``extra_methods``. Returns the path to
    the cleaned ``website_data/`` directory to copy into the HuggingFace Space's ``data/``.
    """
    from tabarena.plot.tuning_trajectories.plot_pareto_over_tuning_time import plot_tuning_trajectories_all
    from tabarena.website.process_artifacts_to_website import process_one_folder
    from tabarena.website.process_pngs import process_png_bulk

    known = set(TabArenaContext().methods)
    extra = [m for m in new_methods if m.method not in known]
    if extra:
        print(f"website: adding {len(extra)} new method(s): {[m.method for m in extra]}")
    context = TabArenaContext(extra_methods=extra) if extra else TabArenaContext()

    raw_out, clean_out = Path("output_website_artifacts"), Path("clean_website_artifacts")

    # 1. Heavy: compute leaderboards/tables/figures + tuning trajectories (zipped for archival).
    context.evaluate_all(
        save_path=raw_out,
        elo_bootstrap_rounds=ELO_BOOTSTRAP_ROUNDS,
        use_website_folder_names=True,
        evaluator_kwargs={"figure_file_type": "png"},
        engine="ray",
    )
    plot_tuning_trajectories_all(
        tabarena_context=context,
        fig_save_dir=raw_out,
        ban_bad_methods=True,
        file_ext=".png",
        engine="ray",
        use_elo_method_order=False,
    )
    shutil.make_archive("tabarena_website_artifacts", "zip", root_dir=raw_out)

    # 2. Fast: convert each leaderboard folder to the cleaned website layout (zipped for upload).
    for path in raw_out.glob("**/website_leaderboard.csv"):
        base_out = clean_out / path.parent.relative_to(raw_out)
        process_one_folder(base_input_path=path.parent, base_output_path=base_out)
        process_png_bulk(path=base_out)
    website_data = clean_out / "website_data"
    shutil.make_archive(str(clean_out), "zip", root_dir=website_data)
    return website_data


if __name__ == "__main__":
    if not SPECS:
        raise SystemExit("Edit the SPECS list near the top of this script first.")
    STAGING.mkdir(parents=True, exist_ok=True)

    methods: list[MethodMetadata] = []
    for i, spec in enumerate(SPECS, start=1):
        print(f"[{i}/{len(SPECS)}] {spec.model or spec.name or spec.source}")
        methods.append(process_spec(spec))
    print(
        f"\nProcessed {len(methods)} method(s); {'uploaded to R2' if UPLOAD else 'cache-only (set UPLOAD=1 to push)'}."
    )

    if WEBSITE:
        print("\nRebuilding website artifacts ...")
        website_data = build_website_artifacts(methods)
        print(f"\nWebsite-ready data -> {website_data}  (copy its contents into the HF Space's data/ dir)")
    else:
        print("Set WEBSITE=1 to also rebuild the website artifacts.")
