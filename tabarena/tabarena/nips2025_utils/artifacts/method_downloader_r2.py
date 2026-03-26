from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix

from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata


class MethodDownloaderR2:
    """
    Download a method's cached artifacts from Cloudflare R2 and restore the original
    local layout expected by MethodUploaderR2 / MethodMetadata.

    Artifacts handled:
      - metadata YAML
      - raw.zip  -> extracted into `method_metadata.path_raw`
      - processed.zip -> extracted into `method_metadata.path_processed`
      - configs_hyperparameters (standalone YAML/JSON/etc. per your metadata)
      - results files (the set returned by `method_metadata.path_results_files()`)

    Notes
    -----
    - Object keys are reconstructed from the local target paths via
      MethodMetadata.to_s3_cache_loc so the local <-> R2 mapping stays perfectly
      symmetric with MethodUploaderR2.
    - If an optional artifact is missing on R2, it is skipped with a warning.
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
        self.method_metadata = method_metadata
        self.method = method_metadata.method

        self.r2_account_id = r2_account_id
        self.r2_bucket = r2_bucket
        self.r2_access_key_id = r2_access_key_id
        self.r2_secret_access_key = r2_secret_access_key
        self.r2_prefix = r2_prefix

        self.prefix = Path(self.r2_prefix) / method_metadata.relative_to_cache_root(method_metadata.path)
        self.verbose = verbose
        self.clear_dirs = clear_dirs

        self._r2_client = None

    @property
    def endpoint_url(self) -> str:
        return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"

    @property
    def r2_cache_root(self) -> str:
        # Keep s3:// style since existing helper utilities expect bucket/key semantics.
        return f"s3://{self.r2_bucket}/{self.r2_prefix}"

    @property
    def s3_cache_root(self) -> str:
        # Alias for compatibility with existing helper naming.
        return self.r2_cache_root

    @property
    def r2_client(self):
        if self._r2_client is None:
            import boto3

            self._r2_client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.r2_access_key_id,
                aws_secret_access_key=self.r2_secret_access_key,
                region_name="auto",
            )
        return self._r2_client

    # --------------------
    # Properties / mapping
    # --------------------
    def local_to_r2_path(self, path_local: str | Path) -> str:
        s3_path_loc = self.method_metadata.to_s3_cache_loc(
            path=Path(path_local),
            s3_cache_root=self.r2_cache_root,
        )
        _, r2_key = s3_path_to_bucket_prefix(s3_path_loc)
        return r2_key

    def local_to_s3_path(self, path_local: str | Path) -> str:
        return self.local_to_r2_path(path_local=path_local)

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
        r2_key = self.local_to_r2_path(path_local=path_local)
        self._download_to_local_if_exists(r2_key=r2_key, path_local=path_local)

    def download_raw(self):
        dest_dir = Path(self.method_metadata.path_raw)
        r2_key = (self.prefix / "raw.zip").as_posix()
        self._download_and_unzip_if_exists(r2_key=r2_key, dest_dir=dest_dir, clear_dir=self.clear_dirs)

    def download_processed(self, holdout: bool = False):
        if holdout:
            dest_dir = Path(self.method_metadata.path_processed_holdout)
            r2_key = (self.prefix / "processed_holdout.zip").as_posix()
        else:
            dest_dir = Path(self.method_metadata.path_processed)
            r2_key = (self.prefix / "processed.zip").as_posix()

        self._download_and_unzip_if_exists(r2_key=r2_key, dest_dir=dest_dir, clear_dir=self.clear_dirs)

    def download_configs_hyperparameters(self, holdout: bool = False):
        path_local = Path(self.method_metadata.path_configs_hyperparameters(holdout=holdout))
        r2_key = self.local_to_r2_path(path_local=path_local)
        self._download_to_local_if_exists(r2_key=r2_key, path_local=path_local)

    def download_results(self, holdout: bool = False):
        file_names: Iterable[Path | str] = self.method_metadata.path_results_files(holdout=holdout)
        for path_local in file_names:
            path_local = Path(path_local)
            r2_key = self.local_to_r2_path(path_local=path_local)
            self._download_to_local_if_exists(r2_key=r2_key, path_local=path_local)

    def download_results_hpo_trajectories(self, holdout: bool = False):
        path_local = self.method_metadata.path_results_hpo_trajectories(holdout=holdout)
        r2_key = self.local_to_r2_path(path_local=path_local)
        self._download_to_local_if_exists(r2_key=r2_key, path_local=Path(path_local))

    # --------------
    # Helper methods
    # --------------
    def _is_missing_error(self, e: Exception) -> bool:
        from botocore.exceptions import ClientError

        if not isinstance(e, ClientError):
            return False
        code = e.response.get("Error", {}).get("Code")
        return code in ("404", "NoSuchKey", "NotFound")

    def _download_to_local_if_exists(self, r2_key: str | Path, path_local: Path):
        """
        Attempts to download a single file to `path_local`. Skips quietly if not found.
        """
        from botocore.exceptions import ClientError

        if isinstance(r2_key, Path):
            r2_key = r2_key.as_posix()

        try:
            self.r2_client.head_object(Bucket=self.r2_bucket, Key=r2_key)
        except ClientError as e:
            if self._is_missing_error(e):
                if self.verbose:
                    print(f"[WARN] Missing on R2, skipping: r2://{self.r2_bucket}/{r2_key}")
                return
            raise

        path_local.parent.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print(f"[INFO] Downloading r2://{self.r2_bucket}/{r2_key} -> {path_local}")

        self.r2_client.download_file(
            Bucket=self.r2_bucket,
            Key=r2_key,
            Filename=str(path_local),
        )

    def _download_and_unzip_if_exists(self, r2_key: str | Path, dest_dir: Path, clear_dir: bool = True):
        """
        Downloads a zip from R2 into memory and extracts into `dest_dir`.
        Skips if the object does not exist.
        """
        from botocore.exceptions import ClientError

        if isinstance(r2_key, Path):
            r2_key = r2_key.as_posix()

        try:
            self.r2_client.head_object(Bucket=self.r2_bucket, Key=r2_key)
        except ClientError as e:
            if self._is_missing_error(e):
                if self.verbose:
                    print(f"[WARN] Missing on R2, skipping unzip: r2://{self.r2_bucket}/{r2_key}")
                return
            raise

        if self.verbose:
            print(f"[INFO] Downloading r2://{self.r2_bucket}/{r2_key} -> extracting to {dest_dir}")

        obj = self.r2_client.get_object(Bucket=self.r2_bucket, Key=r2_key)

        body = obj["Body"].read()
        buf = io.BytesIO(body)

        if clear_dir and dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(buf, "r") as zf:
            zf.extractall(path=dest_dir)