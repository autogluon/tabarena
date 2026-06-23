from __future__ import annotations

import io
from typing import TYPE_CHECKING

from tabarena.models._artifacts.downloader import MethodDownloader

if TYPE_CHECKING:
    from pathlib import Path

    from tabarena.models._method_metadata import MethodMetadata


class MethodDownloaderR2(MethodDownloader):
    """Download a method's cached artifacts from Cloudflare R2 using explicit credentials.

    Backend specifics over :class:`MethodDownloader`: a ``boto3`` client pointed at the account's
    R2 endpoint with explicit access keys. R2 uses ``s3://`` bucket/key semantics, so all of the
    generic key-mapping logic is inherited unchanged.
    """

    def __init__(
        self,
        method_metadata: MethodMetadata,
        r2_account_id: str,
        r2_bucket: str,
        r2_access_key_id: str,
        r2_secret_access_key: str,
        r2_prefix: str = "cache",
        verbose: bool = True,
        clear_dirs: bool = True,
    ):
        super().__init__(method_metadata, bucket=r2_bucket, prefix=r2_prefix, verbose=verbose, clear_dirs=clear_dirs)
        self.r2_account_id = r2_account_id
        self.r2_access_key_id = r2_access_key_id
        self.r2_secret_access_key = r2_secret_access_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3",
                endpoint_url=f"https://{self.r2_account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=self.r2_access_key_id,
                aws_secret_access_key=self.r2_secret_access_key,
                region_name="auto",
            )
        return self._client

    def _is_missing_error(self, e: Exception) -> bool:
        from botocore.exceptions import ClientError

        if not isinstance(e, ClientError):
            return False
        code = e.response.get("Error", {}).get("Code")
        return code in ("404", "NoSuchKey", "NotFound")

    def _download_to_local_if_exists(self, key: str, path_local: Path):
        """Attempts to download a single file to `path_local`. Skips quietly if not found."""
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            if self._is_missing_error(e):
                self._log(f"[WARN] Missing on R2, skipping: r2://{self.bucket}/{key}")
                return
            raise

        path_local.parent.mkdir(parents=True, exist_ok=True)
        self._log(f"[INFO] Downloading r2://{self.bucket}/{key} -> {path_local}")

        self.client.download_file(Bucket=self.bucket, Key=key, Filename=str(path_local))

    def _download_and_unzip_if_exists(self, key: str, dest_dir: Path, clear_dir: bool = True):
        """Downloads a zip from R2 into memory and extracts into `dest_dir`.
        Skips if the object does not exist.
        """
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            if self._is_missing_error(e):
                self._log(f"[WARN] Missing on R2, skipping unzip: r2://{self.bucket}/{key}")
                return
            raise

        self._log(f"[INFO] Downloading r2://{self.bucket}/{key} -> extracting to {dest_dir}")

        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        self._extract_zip(io.BytesIO(obj["Body"].read()), dest_dir=dest_dir, clear_dir=clear_dir)
