from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing_extensions import Self

from tabarena.nips2025_utils.artifacts.download_utils import download_and_extract_zip
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata
from tabarena.nips2025_utils.artifacts.method_uploader import MethodUploaderS3
from tabarena.nips2025_utils.end_to_end import EndToEndSingle


@dataclass(slots=True)
class MethodArtifactManager:
    """
    Orchestrates download → local cache → upload-to-S3 for a single method.
    """
    name: str               # method name, e.g., "xRFM_GPU" <- Method identifier. (name, artifact_name) must be unique.
    model_key: str          # e.g., "XRFM_GPU" <- Model key to use for differentiating model families.
    artifact_name: str      # e.g., "tabarena-2025-09-03" <- Differentiates methods with the same name by origin.
    path_suffix: Path       # e.g., Path("leaderboard_submissions/data_xRFM_11092025.zip")
    download_prefix: str    # e.g., "https://data.lennart-purucker.com/tabarena/"
    local_prefix: Path      # e.g., Path("local_data") <- Local dir to download raw files. Can delete after cache.
    s3_bucket: str          # e.g., "my-bucket"
    s3_prefix: str          # e.g., "cache"
    upload_as_public: bool = False  # Whether the s3 upload will make files public readable
    method_metadata: MethodMetadata | None = None

    def __post_init__(self):
        if not isinstance(self.path_suffix, Path):
            self.path_suffix = Path(self.path_suffix)
        if self.path_suffix.suffix != ".zip":
            raise ValueError(f"path_suffix should be a .zip, got {self.path_suffix}")
        if not self.download_prefix:
            raise ValueError("download_prefix must be non-empty")
        if not isinstance(self.local_prefix, Path):
            self.local_prefix = Path(self.local_prefix)
        # Normalize s3_prefix to avoid accidental '//' in keys
        self.s3_prefix = self.s3_prefix.strip("/")

    @classmethod
    def from_method_metadata(
        cls,
        method_metadata: MethodMetadata,
        path_suffix: Path,
        download_prefix: str,
        local_prefix: Path,
    ) -> Self:
        return cls(
            name=method_metadata.method,
            artifact_name=method_metadata.artifact_name,
            model_key=method_metadata.model_key,
            s3_bucket=method_metadata.s3_bucket,
            s3_prefix=method_metadata.s3_prefix,
            upload_as_public=method_metadata.upload_as_public,
            method_metadata=method_metadata,
            path_suffix=path_suffix,
            download_prefix=download_prefix,
            local_prefix=local_prefix,
        )

    @property
    def path_raw(self) -> Path:
        return (self.local_prefix / self.path_suffix).with_suffix("")

    @property
    def url(self) -> str:
        return f"{self.download_prefix}{Path(self.path_suffix).as_posix()}"

    def download_raw(self) -> None:
        """
        Download the zip from `self.url` and extract into `self.path_raw`.
        """
        download_and_extract_zip(url=self.url, path_local=self.path_raw)

    def cache(self) -> EndToEndSingle:
        """
        Requires first having the raw files locally by calling `self.download_raw()`

        Cached files will be saved under ~/.cache/tabarena/artifacts/{self.artifact_name}/methods/{self.name}/

        Run logic end-to-end and cache all results:
        1. load raw artifacts
            path_raw should be a directory containing `results.pkl` files for each run.
        2. infer method_metadata
        3. cache method_metadata
        4. cache raw artifacts
        5. infer task_metadata
        5. generate processsed
        6. cache processed
        7. generate results
        8. cache results

        Once this is executed once, it does not need to be run again.
        """
        e2e = EndToEndSingle.from_path_raw(
            path_raw=self.path_raw,
            method_metadata=self.method_metadata,
            name=self.name,
            model_key=self.model_key,
            artifact_name=self.artifact_name,
            cache=True,
            cache_raw=True,
        )
        return e2e

    def upload_to_s3(self) -> None:
        """
        Upload the local cache for this method to S3 under:
        s3://{self.s3_bucket}/{self.s3_prefix}/artifacts/{self.artifact_name}/methods/{self.name}/
        """
        method_metadata = self.load_metadata()
        uploader = MethodUploaderS3(
            method_metadata=method_metadata,
            s3_bucket=self.s3_bucket,
            s3_prefix=self.s3_prefix,
            upload_as_public=self.upload_as_public,
        )
        uploader.upload_all()

    def load_metadata(self) -> MethodMetadata:
        """
        Load MethodMetadata from the local cache for this method.
        """
        return MethodMetadata.from_yaml(method=self.name, artifact_name=self.artifact_name)
