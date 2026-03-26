from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, urljoin

import requests

from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix

from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata


class MethodDownloaderPublicR2:
    """
    Download a method's cached artifacts from a public Cloudflare R2 bucket exposed
    via a custom domain, and restore the original local layout expected by
    MethodUploaderR2 / MethodMetadata.

    Artifacts handled:
      - metadata YAML
      - raw.zip  -> extracted into `method_metadata.path_raw`
      - processed.zip -> extracted into `method_metadata.path_processed`
      - configs_hyperparameters (standalone YAML/JSON/etc. per your metadata)
      - results files (the set returned by `method_metadata.path_results_files()`)

    Notes
    -----
    - Object keys are reconstructed from the local target paths via
      MethodMetadata.to_s3_cache_loc so the local <-> public R2 mapping stays
      symmetric with the uploader.
    - This class assumes the custom domain serves objects directly by key, e.g.
      https://data.tabarena.ai/cache/.../raw.zip
    - If an optional artifact is missing remotely, it is skipped with a warning.
    """

    def __init__(
        self,
        method_metadata: MethodMetadata,
        base_url: str,
        r2_prefix: str = "cache",
        verbose: bool = True,
        clear_dirs: bool = True,
        timeout: float = 60.0,
        chunk_size: int = 1024 * 1024,
        session: requests.Session | None = None,
    ):
        self.method_metadata = method_metadata
        self.method = method_metadata.method

        self.base_url = base_url.rstrip("/") + "/"
        self.r2_prefix = r2_prefix.strip("/")
        self.verbose = verbose
        self.clear_dirs = clear_dirs
        self.timeout = timeout
        self.chunk_size = chunk_size

        self.prefix = Path(self.r2_prefix) / method_metadata.relative_to_cache_root(method_metadata.path)

        self.session = session if session is not None else requests.Session()

    @property
    def r2_cache_root(self) -> str:
        # Keep s3:// style since existing helper utilities expect bucket/key semantics.
        # The bucket name itself is irrelevant here; only the key extracted from the
        # resulting s3 path is used.
        return f"s3://public/{self.r2_prefix}"

    @property
    def s3_cache_root(self) -> str:
        # Alias for compatibility with existing helper naming.
        return self.r2_cache_root

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

    def r2_key_to_url(self, r2_key: str | Path) -> str:
        if isinstance(r2_key, Path):
            r2_key = r2_key.as_posix()
        # Quote each path segment but preserve '/'
        quoted_key = quote(r2_key, safe="/")
        return urljoin(self.base_url, quoted_key)

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
        self._download_and_unzip_if_exists(
            r2_key=r2_key,
            dest_dir=dest_dir,
            clear_dir=self.clear_dirs,
        )

    def download_processed(self, holdout: bool = False):
        if holdout:
            dest_dir = Path(self.method_metadata.path_processed_holdout)
            r2_key = (self.prefix / "processed_holdout.zip").as_posix()
        else:
            dest_dir = Path(self.method_metadata.path_processed)
            r2_key = (self.prefix / "processed.zip").as_posix()

        self._download_and_unzip_if_exists(
            r2_key=r2_key,
            dest_dir=dest_dir,
            clear_dir=self.clear_dirs,
        )

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
        path_local = Path(self.method_metadata.path_results_hpo_trajectories(holdout=holdout))
        r2_key = self.local_to_r2_path(path_local=path_local)
        self._download_to_local_if_exists(r2_key=r2_key, path_local=Path(path_local))

    # --------------
    # Helper methods
    # --------------
    def _head_exists(self, url: str) -> bool:
        try:
            response = self.session.head(
                url,
                allow_redirects=True,
                timeout=self.timeout,
            )
            if response.status_code == 404:
                return False
            response.raise_for_status()
            return True
        except requests.HTTPError:
            return False
        except requests.RequestException:
            # Some public object endpoints may not support HEAD correctly.
            # Fall back to a streamed GET and close immediately.
            try:
                response = self.session.get(
                    url,
                    stream=True,
                    timeout=self.timeout,
                )
                if response.status_code == 404:
                    response.close()
                    return False
                response.raise_for_status()
                response.close()
                return True
            except requests.RequestException:
                return False

    def _download_to_local_if_exists(self, r2_key: str | Path, path_local: Path):
        """
        Attempts to download a single file to `path_local`. Skips quietly if not found.
        """
        url = self.r2_key_to_url(r2_key)

        if not self._head_exists(url):
            if self.verbose:
                print(f"[WARN] Missing on public R2, skipping: {url}")
            return

        path_local.parent.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print(f"[INFO] Downloading {url} -> {path_local}")

        with self.session.get(url, stream=True, timeout=self.timeout) as response:
            response.raise_for_status()
            with path_local.open("wb") as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)

    def _download_and_unzip_if_exists(
        self,
        r2_key: str | Path,
        dest_dir: Path,
        clear_dir: bool = True,
    ):
        """
        Downloads a zip from public R2 to a temporary file and extracts it into
        `dest_dir`. Skips if the object does not exist.

        This avoids loading the full zip into memory, which is important for large
        archives.
        """
        url = self.r2_key_to_url(r2_key)

        if not self._head_exists(url):
            if self.verbose:
                print(f"[WARN] Missing on public R2, skipping unzip: {url}")
            return

        if self.verbose:
            print(f"[INFO] Downloading {url} -> extracting to {dest_dir}")

        if clear_dir and dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
            with self.session.get(url, stream=True, timeout=self.timeout) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        tmp.write(chunk)

            tmp.flush()

            with zipfile.ZipFile(tmp.name, "r") as zf:
                zf.extractall(path=dest_dir)
