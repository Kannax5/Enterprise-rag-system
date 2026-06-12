"""
encoder.py — SentenceTransformer wrapper for generating dense embeddings.
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class Encoder:
    """Thin wrapper around SentenceTransformer for batch encoding."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = device or settings.EMBEDDING_DEVICE
        logger.info("Loading embedding model '%s' on device '%s'.", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of strings into L2-normalised dense vectors.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), dtype float32.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        logger.debug("Encoded %d text(s) → shape %s.", len(texts), embeddings.shape)
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a 1-D float32 vector."""
        return self.encode([query])[0]
