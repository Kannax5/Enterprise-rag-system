"""
reranker.py — Score-based re-ranking and MMR (Maximal Marginal Relevance) filtering.
"""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Reranker:
    """
    Post-processes FAISS retrieval results with two optional strategies:

    1. **Score threshold** — drop chunks whose similarity score is below a minimum.
    2. **MMR** — trade off relevance vs. diversity to reduce redundant results.
    """

    def __init__(
        self,
        min_score: float = 0.0,
        mmr_lambda: float = 0.5,
    ):
        """
        Args:
            min_score:  Minimum cosine similarity required to keep a chunk.
            mmr_lambda: MMR trade-off (1.0 = pure relevance, 0.0 = pure diversity).
        """
        self.min_score = min_score
        self.mmr_lambda = mmr_lambda

    # ── Public API ────────────────────────────────────────────────────────────

    def rerank(
        self,
        results: List[Tuple[dict, float]],
        top_k: int | None = None,
    ) -> List[Tuple[dict, float]]:
        """
        Filter by score threshold, then return up to *top_k* results.
        """
        filtered = [(chunk, score) for chunk, score in results if score >= self.min_score]
        filtered.sort(key=lambda x: x[1], reverse=True)
        if top_k:
            filtered = filtered[:top_k]
        logger.debug("Reranked: %d → %d result(s).", len(results), len(filtered))
        return filtered

    def mmr_rerank(
        self,
        results: List[Tuple[dict, float]],
        embeddings: List[np.ndarray],
        top_k: int = 5,
    ) -> List[Tuple[dict, float]]:
        """
        Maximal Marginal Relevance re-ranking.

        Args:
            results:    List of (chunk, score) from the retriever.
            embeddings: Corresponding embedding vectors (same order as *results*).
            top_k:      Number of diverse results to return.

        Returns:
            Subset of *results* selected for relevance + diversity.
        """
        if not results:
            return []

        scores = np.array([s for _, s in results])
        emb_matrix = np.vstack(embeddings)

        selected_indices: List[int] = []
        remaining = list(range(len(results)))

        while remaining and len(selected_indices) < top_k:
            if not selected_indices:
                # Seed with highest-scoring chunk
                best = int(np.argmax(scores[remaining]))
                idx = remaining[best]
            else:
                # MMR: maximise λ·relevance − (1−λ)·max_similarity_to_selected
                sel_embs = emb_matrix[selected_indices]
                mmr_scores = []
                for i in remaining:
                    rel = scores[i]
                    sim_to_sel = float(np.max(emb_matrix[i] @ sel_embs.T))
                    mmr = self.mmr_lambda * rel - (1 - self.mmr_lambda) * sim_to_sel
                    mmr_scores.append(mmr)
                best = int(np.argmax(mmr_scores))
                idx = remaining[best]

            selected_indices.append(idx)
            remaining.remove(idx)

        return [results[i] for i in selected_indices]
