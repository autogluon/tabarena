from __future__ import annotations

import shutil
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from tabarena.models._method_metadata import MethodMetadata


class MethodDownloader(ABC):
    """Downloads a method's cached artifacts from an S3-compatible store and restores the local
    layout produced by :class:`~tabarena.models._artifacts.uploader.MethodUploader`.

    Holds all the backend-independent logic: which artifacts to fetch (metadata, raw, processed,
    configs_hyperparameters, results, HPO trajectories) and how local paths map to remote keys.
    Concrete subclasses supply only the backend specifics — how to fetch a single object
    (:meth:`_download_to_local_if_exists`) and a zip archive (:meth:`_download_and_unzip_if_exists`),
    each a no-op (skip with a warning) when the object is missing. Subclasses can reuse
    :meth:`_extract_zip` for the shared clear/extract step and :meth:`_log` for verbose output.
    """

    def __init__(
        self,
        method_metadata: MethodMetadata,
        *,
        bucket: str,
        prefix: str = "cache",
        verbose: bool = True,
        clear_dirs: bool = True,
    ):
        self.method_metadata = method_metadata
        self.method = method_metadata.method
        self.bucket = bucket
        # ``prefix`` is the cache-root prefix only (e.g. "cache"); ``key_prefix`` is this method's
        # *full* key prefix within the bucket, e.g. "cache/artifacts/<suite>/methods/<method>".
        # NOTE: this differs from the pre-refactor subclasses, where ``self.prefix`` held the full
        # per-method key path now stored in ``key_prefix`` — use ``key_prefix`` for object keys.
        self.prefix = prefix
        self.key_prefix = Path(prefix) / method_metadata.relative_to_cache_root(method_metadata.path)
        self.verbose = verbose
        self.clear_dirs = clear_dirs

    # -- remote location ----------------------------------------------------------------------
    def local_to_key(self, path_local: str | Path) -> str:
        """Remote object key for a local cache path: ``prefix`` joined with the path's location
        relative to the cache root.

        Computed directly (no ``s3://`` URI is built or parsed), the same way :attr:`key_prefix`
        is — keeping it consistent with the raw/processed zip keys.
        """
        rel = self.method_metadata.relative_to_cache_root(Path(path_local))
        return (Path(self.prefix) / rel).as_posix()

    # -- artifact downloads -------------------------------------------------------------------
    def download_all(self):
        self.download_metadata()
        self.download_raw()
        self.download_processed()
        self.download_results()

    def download_metadata(self):
        path_local = Path(self.method_metadata.path_metadata)
        self._download_to_local_if_exists(key=self.local_to_key(path_local), path_local=path_local)

    def download_raw(self):
        dest_dir = Path(self.method_metadata.path_raw)
        key = (self.key_prefix / "raw.zip").as_posix()
        self._download_and_unzip_if_exists(key=key, dest_dir=dest_dir, clear_dir=self.clear_dirs)

    def download_processed(self):
        dest_dir = Path(self.method_metadata.path_processed)
        key = (self.key_prefix / "processed.zip").as_posix()
        self._download_and_unzip_if_exists(key=key, dest_dir=dest_dir, clear_dir=self.clear_dirs)

    def download_configs_hyperparameters(self):
        path_local = Path(self.method_metadata.path_configs_hyperparameters())
        self._download_to_local_if_exists(key=self.local_to_key(path_local), path_local=path_local)

    def download_results(self):
        for path_local in self.method_metadata.path_results_files():
            path_local = Path(path_local)
            self._download_to_local_if_exists(key=self.local_to_key(path_local), path_local=path_local)

    def download_results_hpo_trajectories(self):
        path_local = Path(self.method_metadata.path_results_hpo_trajectories())
        self._download_to_local_if_exists(key=self.local_to_key(path_local), path_local=path_local)

    # -- shared helpers -----------------------------------------------------------------------
    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _extract_zip(self, zip_source: str | Path | IO[bytes], dest_dir: Path, clear_dir: bool):
        """Extract a zip (a path or an in-memory/file-like object) into ``dest_dir``."""
        if clear_dir and dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_source, "r") as zf:
            zf.extractall(path=dest_dir)

    # -- backend-specific fetch ---------------------------------------------------------------
    @abstractmethod
    def _download_to_local_if_exists(self, key: str, path_local: Path):
        """Download the object at ``key`` to ``path_local``; skip quietly if it is missing."""

    @abstractmethod
    def _download_and_unzip_if_exists(self, key: str, dest_dir: Path, clear_dir: bool = True):
        """Download the zip at ``key`` and extract into ``dest_dir``; skip quietly if missing."""
