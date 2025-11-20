from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix

from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata


class MethodUploaderS3:
    def __init__(
        self,
        method_metadata: MethodMetadata,
        s3_bucket: str,
        s3_prefix: str = "cache",
        upload_as_public: bool = False,
    ):
        self.method_metadata = method_metadata
        self.method = method_metadata.method
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.prefix = Path(self.s3_prefix) / method_metadata.relative_to_cache_root(method_metadata.path)
        self.upload_as_public = upload_as_public

    @property
    def s3_cache_root(self) -> str:
        return f"s3://{self.s3_bucket}/{self.s3_prefix}"

    def upload_all(self):
        self.upload_metadata()
        self.upload_raw()
        self.upload_processed()
        self.upload_results()

    def upload_metadata(self):
        fileobj = self.method_metadata.to_yaml_fileobj()
        path_local = self.method_metadata.path_metadata
        s3_key = self.local_to_s3_path(path_local=path_local)
        self._upload_fileobj(fileobj=fileobj, s3_key=s3_key)

    def upload_raw(self):
        path_raw = Path(self.method_metadata.path_raw)

        print(f"Zipping raw files into memory under: {path_raw}")
        fileobj = self._zip(path=path_raw)
        s3_key = self.prefix / "raw.zip"

        # Upload to S3 directly from memory
        print(f"Uploading raw zipped files to: {s3_key}")
        self._upload_fileobj(fileobj=fileobj, s3_key=s3_key)

    def upload_processed(self, holdout: bool = False):
        if holdout:
            path_processed = self.method_metadata.path_processed_holdout
            processed_zip_name = "processed_holdout.zip"
        else:
            path_processed = self.method_metadata.path_processed
            processed_zip_name = "processed.zip"

        print(f"Zipping processed files into memory under: {path_processed}")
        fileobj = self._zip(path=path_processed)
        s3_key = self.prefix / processed_zip_name

        # Upload to S3 directly from memory
        print(f"Uploading processed zipped files to: {s3_key}")
        self._upload_fileobj(fileobj=fileobj, s3_key=s3_key)

        # Upload configs_hyperparameters as a standalone file for fast access
        self.upload_configs_hyperparameters(holdout=holdout)

    def _upload_fileobj(self, fileobj: io.BytesIO, s3_key: str | Path):
        import boto3

        if isinstance(s3_key, Path):
            s3_key = s3_key.as_posix()

        kwargs = {}
        if self.upload_as_public:
            kwargs = {"ExtraArgs": {"ACL": "public-read"}}

        s3_client = boto3.client("s3")
        s3_client.upload_fileobj(Fileobj=fileobj, Bucket=self.s3_bucket, Key=s3_key, **kwargs)

    def _upload_to_s3(self, path_local: str | Path, s3_key: str | Path | None = None):
        import boto3

        if s3_key is None:
            s3_key = self.local_to_s3_path(path_local=path_local)

        if isinstance(path_local, Path):
            path_local = str(path_local)
        if isinstance(s3_key, Path):
            s3_key = s3_key.as_posix()

        kwargs = {}
        if self.upload_as_public:
            kwargs = {"ExtraArgs": {"ACL": "public-read"}}

        # Upload the file
        s3_client = boto3.client("s3")
        s3_client.upload_file(Filename=path_local, Bucket=self.s3_bucket, Key=s3_key, **kwargs)

    def upload_configs_hyperparameters(self, holdout: bool = False):
        path_local = self.method_metadata.path_configs_hyperparameters(holdout=holdout)
        self._upload_to_s3(path_local=path_local)

    def local_to_s3_path(self, path_local: str | Path) -> str:
        s3_path_loc = self.method_metadata.to_s3_cache_loc(path=Path(path_local), s3_cache_root=self.s3_cache_root)
        _, s3_key = s3_path_to_bucket_prefix(s3_path_loc)
        return s3_key

    def upload_results(self, holdout: bool = False):
        file_names = self.method_metadata.path_results_files(holdout=holdout)
        for path_local in file_names:
            self._upload_to_s3(path_local=path_local)

    def upload_results_hpo_trajectories(self, holdout: bool = False):
        path_local = self.method_metadata.path_results_hpo_trajectories(holdout=holdout)
        self._upload_to_s3(path_local=path_local)

    # TODO: Move to util file
    def _zip(self, path: Path) -> io.BytesIO:
        """
        Create an in-memory ZIP archive of a directory.

        This method recursively traverses the given directory, compresses
        all contained files into a ZIP archive, and returns the archive
        as an in-memory `io.BytesIO` buffer. Paths inside the archive
        are stored relative to the provided root directory.

        Parameters
        ----------
        path : Path
            The root directory whose contents will be compressed into
            the ZIP archive.

        Returns
        -------
        io.BytesIO
            A binary buffer positioned at the beginning, containing the
            ZIP archive data. The caller can read from this buffer or
            upload it directly (e.g., to S3) without writing a local file.

        Raises
        ------
        FileNotFoundError
        If no files exist in the given directory to be zipped.
        """
        # Create an in-memory buffer
        buffer = io.BytesIO()
        file_count = 0

        # Write the zip archive into the buffer
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(path):
                for file in files:
                    file_count += 1
                    file_path = Path(root) / file
                    # Store relative path inside the zip
                    arcname = file_path.relative_to(path)
                    zf.write(file_path, arcname=arcname)

        if file_count == 0:
            raise FileNotFoundError(f"No files found to zip in directory: {path}")

        # Reset buffer position for reading
        buffer.seek(0)
        return buffer
