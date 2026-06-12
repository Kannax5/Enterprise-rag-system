"""
faiss_store.py — FAISS index: build, save, load, and query.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class FaissStore:
    """
    Manages a FAISS IndexFlatIP (inner-product / cosine for L2-normalised vectors).

    Metadata (chunk dicts) is stored alongside the index as a pickle file
    so that each numeric result can be mapped back to the originating chunk.
    """

    def __init__(
        self,
        index_path: str | None = None,
        metadata_path: str | None = None,
        embedding_dim: int = 384,
    ):
        self.index_path = Path(index_path or settings.FAISS_INDEX_PATH)
        self.metadata_path = Path(metadata_path or settings.FAISS_METADATA_PATH)
        self.embedding_dim = embedding_dim

        self._index: faiss.Index | None = None
        self._metadata: List[dict] = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, metadata: List[dict]) -> None:
        """Create a new index from *embeddings* and associated *metadata*."""
        if embeddings.shape[0] != len(metadata):
            raise ValueError("embeddings and metadata must have the same length.")

        self.embedding_dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        self._index.add(embeddings.astype(np.float32))
        self._metadata = metadata
        logger.info("Built FAISS index with %d vector(s).", self._index.ntotal)

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Write index and metadata to disk."""
        if self._index is None:
            raise RuntimeError("No index to save. Call build() first.")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(self.index_path))
        with open(self.metadata_path, "wb") as fh:
            pickle.dump(self._metadata, fh)
        logger.info("Saved FAISS index → %s", self.index_path)

    def load(self) -> None:
        """Load index and metadata from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self._index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as fh:
            self._metadata = pickle.load(fh)
        logger.info("Loaded FAISS index with %d vector(s) from %s.", self._index.ntotal, self.index_path)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_vector: np.ndarray,
        top_k: int | None = None,
    ) -> List[Tuple[dict, float]]:
        """
        Find the *top_k* most similar chunks to *query_vector*.

        Returns:
            List of (chunk_metadata_dict, score) sorted by descending score.
        """
        if self._index is None:
            raise RuntimeError("Index is not loaded. Call load() or build() first.")

        top_k = top_k or settings.FAISS_TOP_K
        q = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(q, top_k)

        results: List[Tuple[dict, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._metadata[idx], float(score)))

        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0
