from __future__ import annotations

import time
from pathlib import Path

from tabarena.nips2025_utils.artifacts.method_artifact_manager import MethodArtifactManager

from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_2025_10_20 import tabdpt_metadata
from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_2025_11_12 import realtabpfn25_metadata


"""
Process methods from `tabarena-2025-09-03` from their original raw data URLs to S3 uploads.

Uncomment methods in `method_infos` to execute processing.
"""
if __name__ == '__main__':
    download = True  # Note: Requires a large amount of available disk space
    cache = True  # Note: Requires a large amount of available disk space
    upload = True  # Requires s3 write permissions to the intended s3 location

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

    # Uncomment whichever artifacts you want to process
    method_infos = [
        # realtabpfn25_info,
        # xrfm_info,
        # mitra_info,
        # ebm_info,
        # realmlp_info,
        # limix_info,
        # tabflex_info,
        # betatabpfn_info,
        # tabdpt_info,
    ]

    if len(method_infos) == 0:
        raise AssertionError(f"Uncomment methods in `method_infos` to run processing. Currently empty.")

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
        if cache:
            method_info.cache()
        if upload:
            method_info.upload_to_s3()
        te = time.time()
        print(f"Finished processing {method_info.name}... (duration={te-ts:.1f}s)")
