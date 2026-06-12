"""
index_builder.py — End-to-end pipeline: chunks → embeddings → FAISS index.
"""

import logging
from typing import List

from app.embeddings.encoder import Encoder
from app.embeddings.faiss_store import FaissStore

logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Orchestrates the full pipeline:
        1. Receive a list of tagged chunk dicts.
        2. Extract text fields and encode them with the Encoder.
        3. Store the resulting vectors + metadata in a FaissStore.
        4. Persist the index to disk.
    """

    def __init__(
        self,
        encoder: Encoder | None = None,
        faiss_store: FaissStore | None = None,
    ):
        self.encoder = encoder or Encoder()
        self.store = faiss_store or FaissStore(embedding_dim=self.encoder.embedding_dim)

    def build(self, chunks: List[dict], save: bool = True) -> FaissStore:
        """
        Build (and optionally save) the FAISS index from *chunks*.

        Args:
            chunks: List of dicts, each containing at least a "text" key.
            save:   If True, persist the index after building.

        Returns:
            The populated FaissStore instance.
        """
        if not chunks:
            raise ValueError("Cannot build index from an empty chunk list.")

        texts = [c["text"] for c in chunks]
        logger.info("Encoding %d chunk(s)…", len(texts))
        embeddings = self.encoder.encode(texts, show_progress=True)

        logger.info("Building FAISS index…")
        self.store.build(embeddings, metadata=chunks)

        if save:
            self.store.save()
            logger.info("Index saved successfully.")

        return self.store
