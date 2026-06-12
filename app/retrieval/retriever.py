"""
retriever.py — Top-k similarity search via FAISS.
"""

import logging
from typing import List, Tuple

import numpy as np

from app.config import settings
from app.embeddings.encoder import Encoder
from app.embeddings.faiss_store import FaissStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Encodes a natural-language query and retrieves the most relevant chunks
    from the FAISS index.
    """

    def __init__(
        self,
        encoder: Encoder | None = None,
        faiss_store: FaissStore | None = None,
        top_k: int | None = None,
    ):
        self.encoder = encoder or Encoder()
        self.store = faiss_store or FaissStore(embedding_dim=self.encoder.embedding_dim)
        self.top_k = top_k or settings.FAISS_TOP_K

        # Lazy-load the FAISS index if it exists on disk
        try:
            self.store.load()
        except FileNotFoundError:
            logger.warning("FAISS index not found — call IndexBuilder.build() first.")

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> List[Tuple[dict, float]]:
        """
        Encode *query* and return the top-k (chunk_dict, score) pairs.

        Args:
            query:  Natural-language search query.
            top_k:  Override the default top-k value.

        Returns:
            List of (metadata_dict, similarity_score) sorted by descending score.
        """
        k = top_k or self.top_k
        query_vec: np.ndarray = self.encoder.encode_query(query)
        results = self.store.query(query_vec, top_k=k)
        logger.debug("Query '%s' → %d result(s).", query[:80], len(results))
        return results
