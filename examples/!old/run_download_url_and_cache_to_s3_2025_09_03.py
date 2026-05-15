from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_2025_10_20 import tabdpt_metadata
from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_2025_11_12 import (
    contexttab_metadata,
    realtabpfn25_metadata,
)
from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_2026_02_16 import tabiclv2_metadata
from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_2026_03_18 import (
    perpetualbooster_metadata,
    tabpfn26_metadata,
    tabstar_metadata,
)
from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_2026_05_13 import limix_metadata as limix_metadata_new, tabpfnv3_method_metadata, orionmsp_metadata
from tabarena.nips2025_utils.artifacts.method_artifact_manager import MethodArtifactManager
from tabarena.nips2025_utils.method_processor import get_info_from_result, load_raw


def _compare_with_provided_metadata(
    method_info: MethodArtifactManager,
    *,
    method_types: list,
    ag_keys: list,
    inferred_compute: str,
    inferred_can_hpo: bool,
    inferred_config_default: str | None,
    is_bag_any: bool,
) -> None:
    """Print a side-by-side comparison of raw/manager-inferred values against the
    user-provided `MethodMetadata` (if any).
    """
    if method_info.method_metadata is None:
        return
    m = method_info.method_metadata

    inferred_method_type = method_types[0] if len(method_types) == 1 else None
    inferred_ag_key = ag_keys[0] if len(ag_keys) == 1 else None

    rows: list[tuple[str, object, object]] = [
        ("method", method_info.name, m.method),
        ("artifact_name", method_info.artifact_name, m.artifact_name),
        ("method_type", inferred_method_type, m.method_type),
        ("compute", inferred_compute, m.compute),
        ("ag_key", inferred_ag_key, m.ag_key),
        ("model_key", method_info.model_key, m.model_key),
        ("config_default", inferred_config_default, m.config_default),
        ("can_hpo", inferred_can_hpo, m.can_hpo),
        ("is_bag", is_bag_any, m.is_bag),
        ("s3_bucket", method_info.s3_bucket, m.s3_bucket),
        ("s3_prefix", method_info.s3_prefix, m.s3_prefix),
        ("upload_as_public", method_info.upload_as_public, m.upload_as_public),
    ]

    field_w = max(len("field"), *(len(f) for f, _, _ in rows))
    inf_w = max(len("inferred"), *(len(str(i)) for _, i, _ in rows))
    prov_w = max(len("provided"), *(len(str(p)) for _, _, p in rows))

    print("[raw-info]   Comparison: inferred (raw/manager) vs provided MethodMetadata:")
    header = (
        f"{'field'.ljust(field_w)} | "
        f"{'inferred'.ljust(inf_w)} | "
        f"{'provided'.ljust(prov_w)} | match"
    )
    print(f"[raw-info]     {header}")
    print(f"[raw-info]     {'-' * len(header)}")

    n_mismatch = 0
    for field, inferred, provided in rows:
        if inferred is None:
            match = "?"  # not enough info to infer; skip mismatch counting
        elif inferred == provided:
            match = "yes"
        else:
            match = "NO"
            n_mismatch += 1
        print(
            f"[raw-info]     {field.ljust(field_w)} | "
            f"{str(inferred).ljust(inf_w)} | "
            f"{str(provided).ljust(prov_w)} | {match}"
        )

    if n_mismatch == 0:
        print("[raw-info]   All inferable fields match the provided MethodMetadata.")
    else:
        print(
            f"[raw-info]   WARNING: {n_mismatch} field(s) differ between inferred and "
            f"provided MethodMetadata (see 'NO' rows above)."
        )


def _format_py_value(val: object) -> str:
    """Render a Python literal that lints clean under ruff (double-quoted strings)."""
    if isinstance(val, str):
        return json.dumps(val)  # ensures double-quoted, properly escaped
    return repr(val)  # bool/int/float/None render the same under repr and ruff


def _print_method_metadata_snippet(
    method_info: MethodArtifactManager,
    *,
    method_types: list,
    ag_keys: list,
    inferred_compute: str,
    inferred_can_hpo: bool,
    inferred_config_default: str | None,
    is_bag_any: bool,
) -> None:
    """Print a copy-pasteable `MethodMetadata(...)` snippet from raw-inferred values."""
    snippet_fields: list[tuple[str, object]] = [
        ("method", method_info.name),
        ("artifact_name", method_info.artifact_name),
        ("method_type", method_types[0] if len(method_types) == 1 else None),
        ("compute", inferred_compute),
        ("ag_key", ag_keys[0] if len(ag_keys) == 1 else None),
        ("model_key", method_info.model_key),
        ("config_default", inferred_config_default),
        ("can_hpo", inferred_can_hpo),
        ("is_bag", is_bag_any),
        ("has_raw", True),
        ("has_processed", True),
        ("has_results", True),
        ("s3_bucket", method_info.s3_bucket),
        ("s3_prefix", method_info.s3_prefix),
        ("upload_as_public", method_info.upload_as_public),
    ]
    slug = re.sub(r"[^a-z0-9]", "", method_info.name.lower()) or "method"
    print("[raw-info]   Suggested MethodMetadata (copy-paste):")
    print(f"{slug}_metadata = MethodMetadata(")
    for key, val in snippet_fields:
        if val is None:
            continue
        print(f"    {key}={_format_py_value(val)},")
    print(")")


def log_raw_data_info(method_info: MethodArtifactManager) -> None:
    """Inspect the raw artifacts at `method_info.path_raw` and print a summary of
    the fields that drive `MethodMetadata` construction (method_type, model_type,
    ag_key, name_prefix, num_gpus -> compute, is_bag, config_default / can_hpo)
    plus task/problem-type/metric coverage.

    Intended to be run after `download_raw()` and before `cache()` so the user can
    sanity-check or build a `MethodMetadata` for this artifact.
    """
    print(f"[raw-info] Loading raw results from '{method_info.path_raw}' for inspection...")
    results_lst = load_raw(path_raw=method_info.path_raw, engine="ray")
    info_df = pd.DataFrame([get_info_from_result(r) for r in results_lst])

    print(f"[raw-info] {method_info.name}: {len(results_lst)} result files")

    def _unique(col: str):
        return sorted(info_df[col].dropna().unique().tolist())

    method_types = _unique("method_type")
    model_types = _unique("model_type")
    ag_keys = _unique("ag_key")
    name_prefixes = _unique("name_prefix")
    problem_types = _unique("problem_type")
    metrics = _unique("metric")
    frameworks = _unique("framework")
    num_gpus_vals = _unique("num_gpus")
    is_bag_any = bool(info_df["is_bag"].any())

    inferred_compute = "gpu" if (num_gpus_vals and max(num_gpus_vals) > 0) else "cpu"
    inferred_can_hpo = len(frameworks) > 1
    inferred_config_default = frameworks[0] if len(frameworks) == 1 else None

    n_tasks = info_df["tid"].nunique() if "tid" in info_df.columns else None
    n_datasets = info_df["name"].nunique() if "name" in info_df.columns else None
    n_folds = info_df["split_idx"].nunique() if "split_idx" in info_df.columns else None

    print(f"[raw-info]   method_type   = {method_types}")
    print(f"[raw-info]   model_type    = {model_types}")
    print(f"[raw-info]   ag_key        = {ag_keys}")
    print(f"[raw-info]   name_prefix   = {name_prefixes}")
    print(f"[raw-info]   problem_type  = {problem_types}")
    print(f"[raw-info]   metric        = {metrics}")
    print(f"[raw-info]   num_gpus      = {num_gpus_vals} -> compute='{inferred_compute}'")
    print(f"[raw-info]   is_bag (any)  = {is_bag_any}")
    print(f"[raw-info]   n_frameworks  = {len(frameworks)} -> can_hpo={inferred_can_hpo}, "
          f"config_default={inferred_config_default!r}")
    print(f"[raw-info]   n_datasets    = {n_datasets}")
    print(f"[raw-info]   n_tasks (tid) = {n_tasks}")
    print(f"[raw-info]   n_folds       = {n_folds}")

    _compare_with_provided_metadata(
        method_info,
        method_types=method_types,
        ag_keys=ag_keys,
        inferred_compute=inferred_compute,
        inferred_can_hpo=inferred_can_hpo,
        inferred_config_default=inferred_config_default,
        is_bag_any=is_bag_any,
    )
    _print_method_metadata_snippet(
        method_info,
        method_types=method_types,
        ag_keys=ag_keys,
        inferred_compute=inferred_compute,
        inferred_can_hpo=inferred_can_hpo,
        inferred_config_default=inferred_config_default,
        is_bag_any=is_bag_any,
    )


"""
Process methods from `tabarena-2025-09-03` from their original raw data URLs to S3 uploads.

Uncomment methods in `method_infos` to execute processing.
"""
if __name__ == "__main__":
    download = True  # Note: Requires a large amount of available disk space
    log_raw_details = True
    cache = False  # Note: Requires a large amount of available disk space
    upload = False  # Requires s3 write permissions to the intended s3 location

    path_args = dict(
        download_prefix="https://data.lennart-purucker.com/tabarena/",
        local_prefix=Path("local_data"),
    )

    shared_args = dict(
        s3_bucket="tabarena",
        s3_prefix="cache",
        upload_as_public=True,
        **path_args,
    )

    # 31 GB
    # Uploaded to s3, artifact_name="tabarena-2025-09-03", s3_prefix="cache", upload_as_public=True
    xrfm_info = MethodArtifactManager(
        path_suffix=Path("leaderboard_submissions") / "data_xRFM_11092025.zip",
        name="xRFM_GPU",
        artifact_name="tabarena-2025-09-03",
        model_key="XRFM_GPU",
        **shared_args,
    )

    # 33 MB
    # Uploaded to s3, artifact_name="tabarena-2025-09-03", s3_prefix="cache", upload_as_public=True
    mitra_info = MethodArtifactManager(
        path_suffix=Path("leaderboard_submissions") / "data_Mitra_14082025.zip",
        name="Mitra_GPU",
        artifact_name="tabarena-2025-09-03",
        model_key="MITRA_GPU",
        **shared_args,
    )

    # 37 GB
    # Uploaded to s3, artifact_name="tabarena-2025-09-03", s3_prefix="cache", upload_as_public=True
    ebm_info = MethodArtifactManager(
        path_suffix=Path("leaderboard_submissions") / "data_EBM_12082025.zip",
        name="ExplainableBM",
        artifact_name="tabarena-2025-09-03",
        model_key="EBM",
        **shared_args,
    )

    # 37 GB
    # Uploaded to s3, artifact_name="tabarena-2025-09-03", s3_prefix="cache", upload_as_public=True
    realmlp_info = MethodArtifactManager(
        path_suffix=Path("leaderboard_submissions") / "data_RealMLP_20082025.zip",
        name="RealMLP_GPU",
        artifact_name="tabarena-2025-09-03",
        model_key="REALMLP_GPU",
        **shared_args,
    )

    # 74 MB
    # Uploaded to s3, artifact_name="tabarena-2025-09-03", s3_prefix="cache", upload_as_public=True
    tabflex_info = MethodArtifactManager(
        path_suffix=Path("data_TabFlex.zip"),
        name="TabFlex_GPU",
        artifact_name="tabarena-2025-09-03",
        model_key="TABFLEX_GPU",
        **shared_args,
    )

    # 71 MB
    # Uploaded to s3, artifact_name="tabarena-2025-09-03", s3_prefix="cache", upload_as_public=True
    limix_info = MethodArtifactManager(
        path_suffix=Path("data_LimiX.zip"),
        name="LimiX_GPU",
        artifact_name="tabarena-2025-09-03",
        model_key="LIMIX_GPU",
        **shared_args,
    )

    # 122 MB
    # Uploaded to s3, artifact_name="tabarena-2025-09-03", s3_prefix="cache", upload_as_public=True
    betatabpfn_info = MethodArtifactManager(
        path_suffix=Path("data_BetaTabPFN.zip"),
        name="BetaTabPFN_GPU",
        artifact_name="tabarena-2025-09-03",
        model_key="BETA_GPU",
        **shared_args,
    )


    # 17 GB
    # Uploaded to s3, artifact_name="tabarena-2025-10-20", s3_prefix="cache", upload_as_public=True
    tabdpt_info = MethodArtifactManager.from_method_metadata(
        method_metadata=tabdpt_metadata,
        path_suffix=Path("leaderboard_submissions") / "data_TabDPT_28102025.zip",
        **path_args,
    )

    # 11 GB
    realtabpfn25_info = MethodArtifactManager.from_method_metadata(
        method_metadata=realtabpfn25_metadata,
        path_suffix=Path("leaderboard_submissions") / "data_RealTabPFN-v2.5_17112025.zip",
        **path_args,
    )

    contexttab_info = MethodArtifactManager.from_method_metadata(
        method_metadata=contexttab_metadata,
        path_suffix=Path("leaderboard_submissions") / "data_SAP-RPT-OSS_25112025.zip",
        **path_args,
    )

    tabiclv2_info = MethodArtifactManager.from_method_metadata(
        method_metadata=tabiclv2_metadata,
        path_suffix=Path("leaderboard_submissions") / "data_TabICLv2_1502026.zip",
        **path_args,
    )

    tabstar_info = MethodArtifactManager.from_method_metadata(
        method_metadata=tabstar_metadata,
        path_suffix=Path("leaderboard_submissions") / "data_TabSTAR_02032026.zip",
        **path_args,
    )

    perpetualbooster_info = MethodArtifactManager.from_method_metadata(
        method_metadata=perpetualbooster_metadata,
        path_suffix=Path("leaderboard_submissions") / "data_PerpertualBooster_06032026.zip",
        **path_args,
    )

    tabpfnv26_info = MethodArtifactManager.from_method_metadata(
        method_metadata=tabpfn26_metadata,
        path_suffix=Path("leaderboard_submissions") / "data_TabPFN_2pt6_26032026.zip",
        **path_args,
    )

    limix_new_info = MethodArtifactManager.from_method_metadata(
        method_metadata=limix_metadata_new,
        path_suffix=Path("leaderboard_submissions") / "data_LimiX_13052026.zip",
        **path_args,
    )

    tabpfnv3_info = MethodArtifactManager.from_method_metadata(
        method_metadata=tabpfnv3_method_metadata,
        path_suffix=Path("leaderboard_submissions") / "data_TabPFN_3_12052026.zip",
        **path_args,
    )

    orionmsp_info = MethodArtifactManager.from_method_metadata(
        method_metadata=orionmsp_metadata,
        path_suffix=Path("leaderboard_submissions") / "data_OrionMSP_14052026.zip",
        **path_args,
    )

    # Uncomment whichever artifacts you want to process
    method_infos = [
        # orionmsp_info,
        # tabpfnv3_info,
        # limix_new_info,
        # tabiclv2_info,
        # contexttab_info,
        # realtabpfn25_info,
        # xrfm_info,
        # mitra_info,
        # ebm_info,
        # realmlp_info,
        # limix_info,
        # tabflex_info,
        # betatabpfn_info,
        # tabdpt_info,
        # perpetualbooster_info,
        # tabstar_info,
        # tabpfnv26_info,
    ]

    if len(method_infos) == 0:
        raise AssertionError("Uncomment methods in `method_infos` to run processing. Currently empty.")

    print(f"Processing {len(method_infos)} methods: {[method_info.name for method_info in method_infos]}")
    for i, method_info in enumerate(method_infos):
        print(
            f"({i+1}/{len(method_infos)}) Processing {method_info.name}... "
            f"(download={download}, cache={cache}, upload={upload})"
        )
        ts = time.time()
        if download:
            print(f"Downloading '{method_info.url}' -> '{method_info.path_raw}'")
            method_info.download_raw()
        if log_raw_details:
            log_raw_data_info(method_info)
        if cache:
            method_info.cache(cache_hpo_trajectories=True)
        if upload:
            method_info.upload_to_s3()
        te = time.time()
        print(f"Finished processing {method_info.name}... (duration={te-ts:.1f}s)")
