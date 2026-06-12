"""
metadata_tagger.py — Attach source, date, and doc_type metadata to chunks.
"""

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Map file extension → doc_type label
_EXT_TO_DOC_TYPE = {
    ".pdf": "pdf",
    ".csv": "csv",
    ".txt": "text",
    ".md": "markdown",
}


class MetadataTagger:
    """
    Enriches a list of chunk dicts with standardised metadata fields:
        - source      : file path / URL
        - doc_type    : inferred from file extension
        - ingested_at : ISO-8601 UTC timestamp
        - chunk_id    : deterministic SHA-256 hash of (source + chunk_index)
    """

    def __init__(self, default_doc_type: str = "unknown"):
        self.default_doc_type = default_doc_type

    # ── Public API ────────────────────────────────────────────────────────────

    def tag(
        self,
        chunks: List[dict],
        source: Optional[str] = None,
        doc_type: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> List[dict]:
        """
        Tag each chunk dict in-place and return the list.

        Args:
            chunks:   List of chunk dicts (must have at least "chunk_index").
            source:   Override source string; falls back to chunk["source"].
            doc_type: Override doc_type; auto-detected from source if omitted.
            extra:    Any additional key-value pairs to merge into every chunk.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        extra = extra or {}

        for chunk in chunks:
            resolved_source = source or chunk.get("source", "unknown")
            resolved_doc_type = doc_type or self._infer_doc_type(resolved_source)

            chunk_index = chunk.get("chunk_index", 0)
            chunk_id = self._make_chunk_id(resolved_source, chunk_index)

            chunk.update(
                {
                    "source": resolved_source,
                    "doc_type": resolved_doc_type,
                    "ingested_at": now_iso,
                    "chunk_id": chunk_id,
                    **extra,
                }
            )

        logger.debug("Tagged %d chunk(s).", len(chunks))
        return chunks

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _infer_doc_type(self, source: str) -> str:
        ext = Path(source).suffix.lower()
        return _EXT_TO_DOC_TYPE.get(ext, self.default_doc_type)

    @staticmethod
    def _make_chunk_id(source: str, chunk_index: int) -> str:
        raw = f"{source}::{chunk_index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
