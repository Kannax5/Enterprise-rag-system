"""
s3_loader.py — Fetch documents from AWS S3.
"""

import logging
from pathlib import Path
from typing import Iterator, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".txt"}


class S3Loader:
    """Downloads documents from an S3 bucket to a local cache directory."""

    def __init__(
        self,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
        local_dir: str = "data/raw",
    ):
        self.bucket = bucket or settings.S3_BUCKET_NAME
        self.prefix = prefix or settings.S3_PREFIX
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self._client = boto3.client(
            "s3",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def list_objects(self) -> List[str]:
        """Return all S3 keys under the configured prefix."""
        keys: List[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
                for obj in page.get("Contents", []):
                    key: str = obj["Key"]
                    if Path(key).suffix.lower() in SUPPORTED_EXTENSIONS:
                        keys.append(key)
        except (BotoCoreError, ClientError) as exc:
            logger.error("Failed to list S3 objects: %s", exc)
            raise
        return keys

    def download_file(self, s3_key: str) -> Path:
        """Download a single S3 object and return its local path."""
        local_path = self.local_dir / Path(s3_key).name
        if local_path.exists():
            logger.debug("Cache hit — skipping download: %s", local_path)
            return local_path
        try:
            logger.info("Downloading s3://%s/%s → %s", self.bucket, s3_key, local_path)
            self._client.download_file(self.bucket, s3_key, str(local_path))
        except (BotoCoreError, ClientError) as exc:
            logger.error("Download failed for key %s: %s", s3_key, exc)
            raise
        return local_path

    def download_all(self) -> Iterator[Path]:
        """Download every supported document and yield local paths."""
        keys = self.list_objects()
        logger.info("Found %d document(s) in s3://%s/%s", len(keys), self.bucket, self.prefix)
        for key in keys:
            yield self.download_file(key)
