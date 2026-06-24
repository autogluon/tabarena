from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote, urljoin

import requests

from tabarena.models._artifacts.downloader import MethodDownloader

if TYPE_CHECKING:
    from tabarena.models._method_metadata import MethodMetadata


class MethodDownloaderPublicR2(MethodDownloader):
    """Download a method's cached artifacts from a public Cloudflare R2 bucket exposed via a
    custom domain (e.g. ``https://data.tabarena.ai/cache/.../raw.zip``).

    Backend specifics over :class:`MethodDownloader`: anonymous HTTP(S) GETs against the custom
    domain (no credentials, no boto3), streaming large archives to a temp file before extracting.
    The bucket name is irrelevant — objects are addressed by key relative to ``base_url`` — so a
    ``"public"`` placeholder bucket is used purely to make the key mapping resolve.
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
        super().__init__(
            method_metadata,
            bucket="public",
            prefix=r2_prefix.strip("/"),
            verbose=verbose,
            clear_dirs=clear_dirs,
        )
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.session = session if session is not None else requests.Session()

    def key_to_url(self, key: str | Path) -> str:
        if isinstance(key, Path):
            key = key.as_posix()
        # Quote each path segment but preserve '/'
        quoted_key = quote(key, safe="/")
        return urljoin(self.base_url, quoted_key)

    def _head_exists(self, url: str) -> bool:
        try:
            response = self.session.head(url, allow_redirects=True, timeout=self.timeout)
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
                response = self.session.get(url, stream=True, timeout=self.timeout)
                if response.status_code == 404:
                    response.close()
                    return False
                response.raise_for_status()
                response.close()
                return True
            except requests.RequestException:
                return False

    def _download_to_local_if_exists(self, key: str, path_local: Path):
        """Attempts to download a single file to `path_local`. Skips quietly if not found."""
        url = self.key_to_url(key)

        if not self._head_exists(url):
            self._log(f"[WARN] Missing on public R2, skipping: {url}")
            return

        path_local.parent.mkdir(parents=True, exist_ok=True)
        self._log(f"[INFO] Downloading {url} -> {path_local}")

        with self.session.get(url, stream=True, timeout=self.timeout) as response:
            response.raise_for_status()
            with path_local.open("wb") as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)

    def _download_and_unzip_if_exists(self, key: str, dest_dir: Path, clear_dir: bool = True):
        """Downloads a zip from public R2 to a temporary file and extracts it into `dest_dir`.
        Skips if the object does not exist.

        This avoids loading the full zip into memory, which is important for large archives.
        """
        url = self.key_to_url(key)

        if not self._head_exists(url):
            self._log(f"[WARN] Missing on public R2, skipping unzip: {url}")
            return

        self._log(f"[INFO] Downloading {url} -> extracting to {dest_dir}")

        with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
            with self.session.get(url, stream=True, timeout=self.timeout) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        tmp.write(chunk)
            tmp.flush()
            self._extract_zip(tmp.name, dest_dir=dest_dir, clear_dir=clear_dir)
