from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

from tabarena.models._artifacts.downloader import MethodDownloader

if TYPE_CHECKING:
    from tabarena.models._method_metadata import MethodMetadata


class MethodDownloaderS3(MethodDownloader):
    """Download a method's cached artifacts from AWS S3.

    Backend specifics over :class:`MethodDownloader`: signed (credentialed) access with a fallback
    to an unsigned (anonymous) client for publicly-readable objects, so the same downloader works
    for both private and public buckets.
    """

    def __init__(
        self,
        method_metadata: MethodMetadata,
        s3_bucket: str,
        s3_prefix: str = "cache",
        verbose: bool = True,
        clear_dirs: bool = True,
    ):
        super().__init__(method_metadata, bucket=s3_bucket, prefix=s3_prefix, verbose=verbose, clear_dirs=clear_dirs)

    def _download_to_local_if_exists(self, key: str, path_local: Path):
        """Attempts to download a single file to `path_local`. Skips quietly if not found."""
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

        sess = boto3.session.Session()
        s3_signed = sess.client("s3")
        s3_unsigned = sess.client("s3", config=Config(signature_version=UNSIGNED))

        def _head(client):
            return client.head_object(Bucket=self.bucket, Key=key)

        # ---------- Existence check (HEAD) ----------
        client_for_get = None
        try:
            _head(s3_signed)
            client_for_get = s3_signed
        except (NoCredentialsError, PartialCredentialsError, ClientError) as e:
            # Treat definitely-missing as skip
            if isinstance(e, ClientError):
                code = e.response.get("Error", {}).get("Code")
                if code in ("404", "NoSuchKey", "NotFound"):
                    self._log(f"[WARN] Missing on S3, skipping: s3://{self.bucket}/{key}")
                    return
            # Retry anonymously for publicly readable objects
            try:
                _head(s3_unsigned)
                client_for_get = s3_unsigned
            except ClientError as e2:
                code2 = e2.response.get("Error", {}).get("Code")
                if code2 in ("404", "NoSuchKey", "NotFound"):
                    self._log(f"[WARN] Missing on S3, skipping: s3://{self.bucket}/{key}")
                    return
                # Still denied or other error -> propagate
                raise

        # ---------- Download ----------
        path_local.parent.mkdir(parents=True, exist_ok=True)
        self._log(f"[INFO] Downloading s3://{self.bucket}/{key} -> {path_local}")

        try:
            client_for_get.download_file(Bucket=self.bucket, Key=key, Filename=str(path_local))
        except (NoCredentialsError, PartialCredentialsError, ClientError):
            # If we tried signed and it failed but the object might be public, try unsigned once.
            if client_for_get is s3_signed:
                try:
                    s3_unsigned.download_file(Bucket=self.bucket, Key=key, Filename=str(path_local))
                    return
                except ClientError:
                    pass
            # propagate original error if unsigned also failed or we were already unsigned
            raise

    def _download_and_unzip_if_exists(self, key: str, dest_dir: Path, clear_dir: bool = True):
        """Downloads a zip from S3 into memory and extracts into `dest_dir`.
        Skips if the object does not exist. Supports public objects by retrying
        with an unsigned client when a signed request is denied or creds are missing.
        """
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

        sess = boto3.session.Session()
        s3_signed = sess.client("s3")
        s3_unsigned = sess.client("s3", config=Config(signature_version=UNSIGNED))

        def _head(c):
            return c.head_object(Bucket=self.bucket, Key=key)

        # ---- Existence check (HEAD) with unsigned fallback ----
        client_for_get = None
        try:
            _head(s3_signed)
            client_for_get = s3_signed
        except (NoCredentialsError, PartialCredentialsError, ClientError) as e:
            if isinstance(e, ClientError):
                code = e.response.get("Error", {}).get("Code")
                if code in ("404", "NoSuchKey", "NotFound"):
                    self._log(f"[WARN] Missing on S3, skipping unzip: s3://{self.bucket}/{key}")
                    return
            try:
                _head(s3_unsigned)
                client_for_get = s3_unsigned
            except ClientError as e2:
                code2 = e2.response.get("Error", {}).get("Code")
                if code2 in ("404", "NoSuchKey", "NotFound"):
                    self._log(f"[WARN] Missing on S3, skipping unzip: s3://{self.bucket}/{key}")
                    return
                raise

        self._log(f"[INFO] Downloading s3://{self.bucket}/{key} -> extracting to {dest_dir}")

        # ---- GET with fallback to unsigned only for credential-related failures ----
        try:
            obj = client_for_get.get_object(Bucket=self.bucket, Key=key)
        except (NoCredentialsError, PartialCredentialsError, ClientError) as e:
            # Only retry unsigned for specific credential/signing errors
            if isinstance(e, ClientError):
                code = e.response.get("Error", {}).get("Code")
                if code not in {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"}:
                    raise
            if client_for_get is s3_signed:
                obj = s3_unsigned.get_object(Bucket=self.bucket, Key=key)
            else:
                raise

        # ---- In-memory unzip ----
        self._extract_zip(io.BytesIO(obj["Body"].read()), dest_dir=dest_dir, clear_dir=clear_dir)
