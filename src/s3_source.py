"""Read source documents out of S3.

Thin on purpose: list what is there, hand back bytes. Everything downstream
works on bytes and does not care where they came from, which is what makes the
system document-source-agnostic -- swapping S3 for a local folder would mean
replacing this file and nothing else.
"""

from typing import Iterator

import boto3

import config


def make_client():
    return boto3.client("s3", region_name=config.AWS_REGION)


def list_pdf_keys(client) -> Iterator[str]:
    """Yield the S3 key of every .pdf under the configured prefix.

    A paginator rather than a plain list_objects_v2 call: that API returns at
    most 1000 keys per response, so a direct call silently truncates on a large
    bucket. The paginator follows the continuation tokens for us. Ours is small
    today, but a silent cap is a bad failure mode to design in.
    """
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=config.S3_BUCKET, Prefix=config.S3_PREFIX):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".pdf"):
                yield obj["Key"]


def download(client, key: str) -> bytes:
    """Fetch one object's bytes into memory.

    Fine for whitepapers (single-digit MB). A production version would stream
    to a temp file so a 500MB document could not exhaust memory.
    """
    return client.get_object(Bucket=config.S3_BUCKET, Key=key)["Body"].read()
