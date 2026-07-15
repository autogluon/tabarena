from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from tabarena.models._artifacts.uploader_utils import zip_dir_stream

if TYPE_CHECKING:
    import io

    from tabarena.models._method_metadata import MethodMetadata


@lru_cache(maxsize=1)
def _transfer_config():
    """Multipart transfer settings for artifact uploads (boto3 imported lazily).

    64 MiB parts, 8 concurrent: parallel part uploads saturate the link, buffered memory for
    the non-seekable zip streams stays bounded (~parts x concurrency = 512 MiB), and even
    multi-TiB archives stay far below the S3/R2 10k-part limit.
    """
    from boto3.s3.transfer import TransferConfig

    part_size = 64 * 1024 * 1024
    return TransferConfig(
        multipart_threshold=part_size,
        multipart_chunksize=part_size,
        max_concurrency=8,
    )


class MethodUploader(ABC):
    """Uploads a method's cached artifacts to an S3-compatible object store.

    Holds all the backend-independent logic: which artifacts to upload (metadata, raw, processed,
    results, HPO trajectories), how local paths map to remote keys, and the low-level put calls.
    Concrete subclasses (:class:`~tabarena.models._artifacts.uploader_s3.MethodUploaderS3`,
    :class:`~tabarena.models._artifacts.uploader_r2.MethodUploaderR2`) supply only the backend
    specifics — the boto3 client via :meth:`_make_client`, and any per-upload extra args via
    :meth:`_extra_args` (e.g. a public-read ACL).
    """

    def __init__(self, method_metadata: MethodMetadata, *, bucket: str, prefix: str = "cache"):
        self.method_metadata = method_metadata
        self.method = method_metadata.method
        self.bucket = bucket
        # ``prefix`` is the cache-root prefix only (e.g. "cache"); ``key_prefix`` is this method's
        # *full* key prefix within the bucket, e.g. "cache/artifacts/<suite>/methods/<method>".
        # NOTE: this differs from the pre-refactor subclasses, where ``self.prefix`` held the full
        # per-method key path now stored in ``key_prefix`` — use ``key_prefix`` for object keys.
        self.prefix = prefix
        self.key_prefix = Path(prefix) / method_metadata.relative_to_cache_root(method_metadata.path)
        self._client = None

    # -- backend-specific hooks ---------------------------------------------------------------
    @abstractmethod
    def _make_client(self):
        """Create and return the boto3 (S3-compatible) client used for uploads."""

    def _extra_args(self) -> dict:
        """Extra args passed to ``upload_file`` / ``upload_fileobj`` (e.g. an ACL). Default: none."""
        return {}

    @property
    def client(self):
        """The lazily-created, cached boto3 client (see :meth:`_make_client`)."""
        if self._client is None:
            self._client = self._make_client()
        return self._client

    # -- remote location ----------------------------------------------------------------------
    def local_to_key(self, path_local: str | Path) -> str:
        """Remote object key for a local cache path: ``prefix`` joined with the path's location
        relative to the cache root.

        Computed directly (no ``s3://`` URI is built or parsed), the same way :attr:`key_prefix`
        is — keeping it consistent with the raw/processed zip keys.
        """
        rel = self.method_metadata.relative_to_cache_root(Path(path_local))
        return (Path(self.prefix) / rel).as_posix()

    # -- artifact uploads ---------------------------------------------------------------------
    def upload_all(self):
        self.upload_metadata()
        self.upload_raw()
        self.upload_processed()
        self.upload_results()
        if self.method_metadata.method_type == "config":
            self.upload_results_hpo_trajectories()

    def upload_metadata(self):
        fileobj = self.method_metadata.to_yaml_fileobj()
        key = self.local_to_key(path_local=self.method_metadata.path_metadata)
        self._upload_fileobj(fileobj=fileobj, key=key)

    def upload_raw(self):
        self._upload_dir_zipped(path_local=self.method_metadata.path_raw, key=self.key_prefix / "raw.zip")

    def upload_processed(self):
        self._upload_dir_zipped(
            path_local=self.method_metadata.path_processed,
            key=self.key_prefix / "processed.zip",
        )

        # Upload configs_hyperparameters as a standalone file for fast access
        self.upload_configs_hyperparameters()

    def _upload_dir_zipped(self, path_local: str | Path, key: str | Path):
        """Upload a directory as one ZIP object, zipping and uploading concurrently.

        The archive is streamed (see :func:`zip_dir_stream`): compression runs in a background
        thread while the multipart upload consumes the stream, so wall time is ~max(zip, upload)
        rather than their sum, and the archive is never held in memory or written to disk.
        """
        print(f"Uploading {path_local} as a streamed zip to: {key}")
        progress_desc = f"Uploading {Path(key).name} ({self.method})"
        with zip_dir_stream(path=path_local, progress_desc=progress_desc) as fileobj:
            self._upload_fileobj(fileobj=fileobj, key=key)

    def upload_configs_hyperparameters(self):
        self._upload_file(path_local=self.method_metadata.path_configs_hyperparameters())

    def upload_results(self):
        for path_local in self.method_metadata.path_results_files():
            self._upload_file(path_local=path_local)

    def upload_results_hpo_trajectories(self):
        self._upload_file(path_local=self.method_metadata.path_results_hpo_trajectories())

    # -- low-level put ------------------------------------------------------------------------
    def _upload_fileobj(self, fileobj: io.IOBase, key: str | Path):
        if isinstance(key, Path):
            key = key.as_posix()
        extra_args = self._extra_args()
        kwargs = {"ExtraArgs": extra_args} if extra_args else {}
        self.client.upload_fileobj(Fileobj=fileobj, Bucket=self.bucket, Key=key, Config=_transfer_config(), **kwargs)

    def _upload_file(self, path_local: str | Path, key: str | Path | None = None):
        if key is None:
            key = self.local_to_key(path_local=path_local)
        if isinstance(key, Path):
            key = key.as_posix()
        extra_args = self._extra_args()
        kwargs = {"ExtraArgs": extra_args} if extra_args else {}
        self.client.upload_file(
            Filename=str(path_local), Bucket=self.bucket, Key=key, Config=_transfer_config(), **kwargs
        )
