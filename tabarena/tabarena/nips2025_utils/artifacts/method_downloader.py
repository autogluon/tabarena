from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix

from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata


class MethodDownloaderS3:
    """
    Download a method's cached artifacts from S3 and restore the original local layout
    expected by MethodUploaderS3 / MethodMetadata.

    Artifacts handled:
      - metadata YAML
      - raw.zip  -> extracted into `method_metadata.path_raw`
      - processed.zip -> extracted into `method_metadata.path_processed`
      - configs_hyperparameters (standalone YAML/JSON/etc. per your metadata)
      - results files (the set returned by `method_metadata.path_results_files()`)

    Notes
    -----
    - S3 keys are reconstructed from the local target paths via MethodMetadata.to_s3_cache_loc
      so the local <-> S3 mapping stays perfectly symmetric with MethodUploaderS3.
    - If an optional artifact is missing on S3, it is skipped with a warning (no exception).
    """

    def __init__(
        self,
        method_metadata: MethodMetadata,
        s3_bucket: str,
        s3_prefix: str = "cache",
        verbose: bool = True,
        clear_dirs: bool = True,
    ):
        self.method_metadata = method_metadata
        self.method = method_metadata.method
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.prefix = Path(self.s3_prefix) / method_metadata.relative_to_cache_root(method_metadata.path)
        self.verbose = verbose
        self.clear_dirs = clear_dirs

    # --------------------
    # Properties / mapping
    # --------------------
    @property
    def s3_cache_root(self) -> str:
        return f"s3://{self.s3_bucket}/{self.s3_prefix}"

    def local_to_s3_path(self, path_local: str | Path) -> str:
        s3_path_loc = self.method_metadata.to_s3_cache_loc(path=Path(path_local), s3_cache_root=self.s3_cache_root)
        _, s3_key = s3_path_to_bucket_prefix(s3_path_loc)
        return s3_key

    # ---------------
    # Public entrypoint
    # ---------------
    def download_all(self):
        self.download_metadata()
        self.download_raw()
        self.download_processed()
        self.download_results()

    # --------
    # Downloads
    # --------
    def download_metadata(self):
        path_local = Path(self.method_metadata.path_metadata)
        s3_key = self.local_to_s3_path(path_local=path_local)
        self._download_to_local_if_exists(s3_key=s3_key, path_local=path_local)

    def download_raw(self):
        dest_dir = Path(self.method_metadata.path_raw)
        s3_key = (self.prefix / "raw.zip").as_posix()
        self._download_and_unzip_if_exists(s3_key=s3_key, dest_dir=dest_dir, clear_dir=self.clear_dirs)

    def download_processed(self):
        dest_dir = Path(self.method_metadata.path_processed)
        s3_key = (self.prefix / "processed.zip").as_posix()
        self._download_and_unzip_if_exists(s3_key=s3_key, dest_dir=dest_dir, clear_dir=self.clear_dirs)

    def download_configs_hyperparameters(self, holdout: bool = False):
        path_local = Path(self.method_metadata.path_configs_hyperparameters(holdout=holdout))
        s3_key = self.local_to_s3_path(path_local=path_local)
        self._download_to_local_if_exists(s3_key=s3_key, path_local=path_local)

    def download_results(self, holdout: bool = False):
        file_names: Iterable[Path | str] = self.method_metadata.path_results_files(holdout=holdout)
        for path_local in file_names:
            path_local = Path(path_local)
            s3_key = self.local_to_s3_path(path_local=path_local)
            self._download_to_local_if_exists(s3_key=s3_key, path_local=path_local)

    def download_results_hpo_trajectories(self, holdout: bool = False):
        path_local = self.method_metadata.path_results_hpo_trajectories(holdout=holdout)
        s3_key = self.local_to_s3_path(path_local=path_local)
        self._download_to_local_if_exists(s3_key=s3_key, path_local=path_local)

    # --------------
    # Helper methods
    # --------------
    def _download_to_local_if_exists(self, s3_key: str | Path, path_local: Path):
        """
        Attempts to download a single file to `path_local`. Skips quietly if not found.
        """
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

        if isinstance(s3_key, Path):
            s3_key = s3_key.as_posix()

        sess = boto3.session.Session()
        s3_signed = sess.client("s3")
        s3_unsigned = sess.client("s3", config=Config(signature_version=UNSIGNED))

        def _head(client):
            return client.head_object(Bucket=self.s3_bucket, Key=s3_key)

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
                    if self.verbose:
                        print(f"[WARN] Missing on S3, skipping: s3://{self.s3_bucket}/{s3_key}")
                    return
            # Retry anonymously for publicly readable objects
            try:
                _head(s3_unsigned)
                client_for_get = s3_unsigned
            except ClientError as e2:
                code2 = e2.response.get("Error", {}).get("Code")
                if code2 in ("404", "NoSuchKey", "NotFound"):
                    if self.verbose:
                        print(f"[WARN] Missing on S3, skipping: s3://{self.s3_bucket}/{s3_key}")
                    return
                # Still denied or other error -> propagate
                raise

        # ---------- Download ----------
        path_local.parent.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print(f"[INFO] Downloading s3://{self.s3_bucket}/{s3_key} -> {path_local}")

        try:
            client_for_get.download_file(Bucket=self.s3_bucket, Key=s3_key, Filename=str(path_local))
        except (NoCredentialsError, PartialCredentialsError, ClientError) as e:
            # If we tried signed and it failed but the object might be public, try unsigned once.
            if client_for_get is s3_signed:
                try:
                    s3_unsigned.download_file(Bucket=self.s3_bucket, Key=s3_key, Filename=str(path_local))
                    return
                except ClientError:
                    pass
            # propagate original error if unsigned also failed or we were already unsigned
            raise

    def _download_and_unzip_if_exists(self, s3_key: str | Path, dest_dir: Path, clear_dir: bool = True):
        """
        Downloads a zip from S3 into memory and extracts into `dest_dir`.
        Skips if the object does not exist. Supports public objects by retrying
        with an unsigned client when a signed request is denied or creds are missing.
        """
        import io
        import shutil
        import zipfile
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

        if isinstance(s3_key, Path):
            s3_key = s3_key.as_posix()

        sess = boto3.session.Session()
        s3_signed = sess.client("s3")
        s3_unsigned = sess.client("s3", config=Config(signature_version=UNSIGNED))

        def _head(c):
            return c.head_object(Bucket=self.s3_bucket, Key=s3_key)

        # ---- Existence check (HEAD) with unsigned fallback ----
        client_for_get = None
        try:
            _head(s3_signed)
            client_for_get = s3_signed
        except (NoCredentialsError, PartialCredentialsError, ClientError) as e:
            if isinstance(e, ClientError):
                code = e.response.get("Error", {}).get("Code")
                if code in ("404", "NoSuchKey", "NotFound"):
                    if self.verbose:
                        print(f"[WARN] Missing on S3, skipping unzip: s3://{self.s3_bucket}/{s3_key}")
                    return
            try:
                _head(s3_unsigned)
                client_for_get = s3_unsigned
            except ClientError as e2:
                code2 = e2.response.get("Error", {}).get("Code")
                if code2 in ("404", "NoSuchKey", "NotFound"):
                    if self.verbose:
                        print(f"[WARN] Missing on S3, skipping unzip: s3://{self.s3_bucket}/{s3_key}")
                    return
                raise

        if self.verbose:
            print(f"[INFO] Downloading s3://{self.s3_bucket}/{s3_key} -> extracting to {dest_dir}")

        # ---- GET with fallback to unsigned only for credential-related failures ----
        try:
            obj = client_for_get.get_object(Bucket=self.s3_bucket, Key=s3_key)
        except (NoCredentialsError, PartialCredentialsError, ClientError) as e:
            # Only retry unsigned for specific credential/signing errors
            if isinstance(e, ClientError):
                code = e.response.get("Error", {}).get("Code")
                if code not in {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"}:
                    raise
            if client_for_get is s3_signed:
                obj = s3_unsigned.get_object(Bucket=self.s3_bucket, Key=s3_key)
            else:
                raise

        # ---- In-memory unzip ----
        body = obj["Body"].read()
        buf = io.BytesIO(body)

        if clear_dir and dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(buf, "r") as zf:
            zf.extractall(path=dest_dir)
