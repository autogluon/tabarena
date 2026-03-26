from __future__ import annotations

import io
from pathlib import Path

from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix

from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata
from tabarena.nips2025_utils.artifacts.method_uploader_utils import zip_in_memory


class MethodUploaderR2:
    def __init__(
        self,
        method_metadata: MethodMetadata,
        r2_account_id: str,
        r2_bucket: str,
        r2_access_key_id: str,
        r2_secret_access_key: str,
        r2_prefix: str = "cache",
    ):
        self.method_metadata = method_metadata
        self.method = method_metadata.method

        self.r2_account_id = r2_account_id
        self.r2_bucket = r2_bucket
        self.r2_access_key_id = r2_access_key_id
        self.r2_secret_access_key = r2_secret_access_key
        self.r2_prefix = r2_prefix

        self.prefix = Path(self.r2_prefix) / method_metadata.relative_to_cache_root(method_metadata.path)

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

    def upload_all(self):
        self.upload_metadata()
        self.upload_raw()
        self.upload_processed()
        self.upload_results()

    def upload_metadata(self):
        fileobj = self.method_metadata.to_yaml_fileobj()
        path_local = self.method_metadata.path_metadata
        r2_key = self.local_to_r2_path(path_local=path_local)
        self._upload_fileobj(fileobj=fileobj, r2_key=r2_key)

    def upload_raw(self):
        path_raw = Path(self.method_metadata.path_raw)

        print(f"Zipping raw files into memory under: {path_raw}")
        fileobj = zip_in_memory(path=path_raw)
        r2_key = self.prefix / "raw.zip"

        print(f"Uploading raw zipped files to: {r2_key}")
        self._upload_fileobj(fileobj=fileobj, r2_key=r2_key)

    def upload_processed(self, holdout: bool = False):
        if holdout:
            path_processed = self.method_metadata.path_processed_holdout
            processed_zip_name = "processed_holdout.zip"
        else:
            path_processed = self.method_metadata.path_processed
            processed_zip_name = "processed.zip"

        print(f"Zipping processed files into memory under: {path_processed}")
        fileobj = zip_in_memory(path=path_processed)
        r2_key = self.prefix / processed_zip_name

        print(f"Uploading processed zipped files to: {r2_key}")
        self._upload_fileobj(fileobj=fileobj, r2_key=r2_key)

        self.upload_configs_hyperparameters(holdout=holdout)

    def _upload_fileobj(self, fileobj: io.BytesIO, r2_key: str | Path):
        if isinstance(r2_key, Path):
            r2_key = r2_key.as_posix()

        self.r2_client.upload_fileobj(
            Fileobj=fileobj,
            Bucket=self.r2_bucket,
            Key=r2_key,
        )

    def _upload_file(self, path_local: str | Path, r2_key: str | Path | None = None):
        if r2_key is None:
            r2_key = self.local_to_r2_path(path_local=path_local)

        if isinstance(path_local, Path):
            path_local = str(path_local)
        if isinstance(r2_key, Path):
            r2_key = r2_key.as_posix()

        self.r2_client.upload_file(
            Filename=path_local,
            Bucket=self.r2_bucket,
            Key=r2_key,
        )

    def upload_configs_hyperparameters(self, holdout: bool = False):
        path_local = self.method_metadata.path_configs_hyperparameters(holdout=holdout)
        self._upload_file(path_local=path_local)

    def local_to_r2_path(self, path_local: str | Path) -> str:
        s3_path_loc = self.method_metadata.to_s3_cache_loc(
            path=Path(path_local),
            s3_cache_root=self.r2_cache_root,
        )
        _, r2_key = s3_path_to_bucket_prefix(s3_path_loc)
        return r2_key

    def local_to_s3_path(self, path_local: str | Path) -> str:
        return self.local_to_r2_path(path_local=path_local)

    def upload_results(self, holdout: bool = False):
        file_names = self.method_metadata.path_results_files(holdout=holdout)
        for path_local in file_names:
            self._upload_file(path_local=path_local)

    def upload_results_hpo_trajectories(self, holdout: bool = False):
        path_local = self.method_metadata.path_results_hpo_trajectories(holdout=holdout)
        self._upload_file(path_local=path_local)
