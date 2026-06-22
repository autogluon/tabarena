from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.models._artifacts.uploader import MethodUploader

if TYPE_CHECKING:
    from tabarena.models._method_metadata import MethodMetadata


class MethodUploaderR2(MethodUploader):
    """Uploads method artifacts to Cloudflare R2 (S3-compatible) using explicit credentials.

    Backend specifics over :class:`MethodUploader`: a ``boto3`` client pointed at the account's R2
    endpoint with explicit access keys. R2 uses ``s3://`` bucket/key semantics, so all of the
    generic key-mapping logic is inherited unchanged.
    """

    def __init__(
        self,
        method_metadata: MethodMetadata,
        r2_account_id: str,
        r2_bucket: str,
        r2_access_key_id: str,
        r2_secret_access_key: str,
        r2_prefix: str = "cache",
    ):
        super().__init__(method_metadata, bucket=r2_bucket, prefix=r2_prefix)
        self.r2_account_id = r2_account_id
        self.r2_access_key_id = r2_access_key_id
        self.r2_secret_access_key = r2_secret_access_key

    def _make_client(self):
        import boto3

        return boto3.client(
            "s3",
            endpoint_url=f"https://{self.r2_account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=self.r2_access_key_id,
            aws_secret_access_key=self.r2_secret_access_key,
            region_name="auto",
        )
