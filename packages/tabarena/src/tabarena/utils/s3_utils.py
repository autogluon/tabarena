from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def s3_get_object(Bucket: str, Key: str, s3=None, **kwargs) -> dict:
    """Get an S3 object, automatically falling back to an anonymous (unsigned) GET
    when a signed request is not possible or is denied.

    This helper first attempts a standard **signed** `GetObject` using the
    provided (or default) boto3 S3 client. If credentials are missing or the
    signed request fails with a credential-related error, it retries with an
    **unsigned** client (`signature_version=UNSIGNED`) so publicly readable
    objects can still be fetched—equivalent to downloading via the HTTPS URL.

    Parameters
    ----------
    Bucket : str
        Name of the S3 bucket.
    Key : str
        Object key within the bucket.
    s3 : botocore.client.S3, optional
        An existing S3 client to use for the initial signed attempt. If not
        provided, a new default client is created.
    **kwargs
        Additional keyword arguments forwarded to `boto3.client('s3').get_object`,
        e.g. `Range`, `VersionId`, `IfMatch`, `IfModifiedSince`, `IfNoneMatch`,
        `IfUnmodifiedSince`, `RequestPayer='requester'`, etc.

    Returns:
    -------
    dict
        The standard `GetObject` response dict from boto3. The payload stream is
        available as `response['Body']` (a `botocore.response.StreamingBody`).

    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

    if s3 is None:
        s3 = boto3.client("s3")
    try:
        return s3.get_object(Bucket=Bucket, Key=Key, **kwargs)  # returns a StreamingBody in ['Body']
    except (NoCredentialsError, PartialCredentialsError, ClientError) as e:
        # If creds are missing or access is denied, try anonymous (unsigned) request.
        # Note: even if you *have* creds, a signed request can be denied while the
        # object is still publicly readable; unsigned can succeed in that case.
        if isinstance(e, ClientError) and e.response["Error"]["Code"] not in {
            "AccessDenied",
            "InvalidAccessKeyId",
            "SignatureDoesNotMatch",
        }:
            raise

        s3_unsigned = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        return s3_unsigned.get_object(Bucket=Bucket, Key=Key, **kwargs)
