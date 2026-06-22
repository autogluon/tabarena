from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.models._artifacts.uploader import MethodUploader

if TYPE_CHECKING:
    from tabarena.models._method_metadata import MethodMetadata


class MethodUploaderS3(MethodUploader):
    """Uploads method artifacts to AWS S3 using ambient AWS credentials.

    Backend specifics over :class:`MethodUploader`: a default ``boto3`` S3 client, and an optional
    ``public-read`` ACL on every uploaded object when ``upload_as_public`` is set.
    """

    def __init__(
        self,
        method_metadata: MethodMetadata,
        s3_bucket: str,
        s3_prefix: str = "cache",
        upload_as_public: bool = False,
    ):
        super().__init__(method_metadata, bucket=s3_bucket, prefix=s3_prefix)
        self.upload_as_public = upload_as_public

    def _make_client(self):
        import boto3

        return boto3.client("s3")

    def _extra_args(self) -> dict:
        return {"ACL": "public-read"} if self.upload_as_public else {}
