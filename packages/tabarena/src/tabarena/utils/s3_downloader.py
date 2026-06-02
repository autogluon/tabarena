from __future__ import annotations

import fnmatch
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _should_exclude(key: str, exclude_globs: Sequence[str]) -> bool:
    """Return True if key matches any of the exclude glob patterns."""
    if not exclude_globs:
        return False
    base = key.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(key, pat) or fnmatch.fnmatch(base, pat) for pat in exclude_globs)


def _safe_mkdirs(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _skip_because_same_size(local_path: Path, size: int) -> bool:
    try:
        return local_path.exists() and local_path.stat().st_size == size
    except OSError:
        return False


def copy_s3_prefix_to_local(
    bucket: str,
    prefix: str | Path,
    dest_dir: str | Path,
    *,
    exclude: Sequence[str] = (),
    max_workers: int = min(32, (os.cpu_count() or 8) * 4),
    multipart_threshold_bytes: int = 8 * 1024 * 1024,
    multipart_chunksize_bytes: int = 16 * 1024 * 1024,
    connect_timeout_s: int = 10,
    read_timeout_s: int = 180,
    dry_run: bool = False,
) -> dict[str, list[str]]:
    """Recursively copy all objects under an S3 prefix to a local directory with progress tracking."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    norm_prefix = prefix.lstrip("/")
    if norm_prefix and not norm_prefix.endswith("/"):
        norm_prefix += "/"

    s3 = boto3.client(
        "s3",
        config=Config(
            retries={"max_attempts": 10, "mode": "adaptive"},
            connect_timeout=connect_timeout_s,
            read_timeout=read_timeout_s,
            max_pool_connections=max_workers * 2,
        ),
    )

    transfer_cfg = TransferConfig(
        multipart_threshold=multipart_threshold_bytes,
        multipart_chunksize=multipart_chunksize_bytes,
        max_concurrency=max_workers,
        use_threads=True,
    )

    paginator = s3.get_paginator("list_objects_v2")

    to_download: list[tuple[str, int, Path]] = []
    excluded: list[str] = []
    skipped: list[str] = []

    logger.info("Listing s3://%s/%s ...", bucket, norm_prefix or "")

    pages = paginator.paginate(Bucket=bucket, Prefix=norm_prefix)
    for page in pages:
        for obj in page.get("Contents", []):
            key: str = obj["Key"]
            size: int = obj["Size"]
            if key.endswith("/") and size == 0:
                continue
            if _should_exclude(key[len(norm_prefix) :], exclude):
                excluded.append(key)
                continue
            rel_key = key[len(norm_prefix) :]
            local_path = dest_dir / rel_key
            if _skip_because_same_size(local_path, size):
                skipped.append(key)
            else:
                to_download.append((key, size, local_path))

    logger.log(
        30,
        "Found %d objects: %d to download, %d skipped, %d excluded.",
        len(to_download) + len(skipped) + len(excluded),
        len(to_download),
        len(skipped),
        len(excluded),
    )

    downloaded: list[str] = []
    failed: list[str] = []

    if dry_run:
        logger.info("[DRY RUN] Would download %d files.", len(to_download))
        return {"downloaded": [], "skipped": skipped, "excluded": excluded, "failed": []}

    def _download_one(item: tuple[str, int, Path]) -> tuple[str, bool, str | None]:
        key, _size, local_path = item
        tmp_path = local_path.with_suffix(local_path.suffix + ".part")
        try:
            _safe_mkdirs(local_path)
            s3.download_file(
                Bucket=bucket,
                Key=key,
                Filename=str(tmp_path),
                Config=transfer_cfg,
            )
            os.replace(tmp_path, local_path)
            return key, True, None
        except Exception as e:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:  # noqa: S110
                pass
            return key, False, str(e)

    if to_download:
        with (
            ThreadPoolExecutor(max_workers=max_workers) as executor,
            tqdm(
                total=len(to_download),
                desc="Downloading files",
                unit="file",
                ncols=100,
                dynamic_ncols=True,
            ) as pbar,
        ):
            futures = {executor.submit(_download_one, item): item[0] for item in to_download}
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    k, ok, err = fut.result()
                    if ok:
                        downloaded.append(k)
                    else:
                        failed.append(f"{k} -> {err}")
                except Exception as e:
                    failed.append(f"{key} -> {e}")
                finally:
                    pbar.update(1)

    logger.info(
        "Download complete. %d succeeded, %d failed, %d skipped, %d excluded.",
        len(downloaded),
        len(failed),
        len(skipped),
        len(excluded),
    )

    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "excluded": excluded,
        "failed": failed,
    }


if __name__ == "__main__":
    summary = copy_s3_prefix_to_local(
        bucket="my-bucket",
        prefix="datasets/",
        dest_dir="./data/local_dir",
        exclude=["*.log"],
        max_workers=64,
    )
    print("Summary:", {k: len(v) for k, v in summary.items()})
