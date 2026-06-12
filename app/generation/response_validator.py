"""
response_validator.py — Factual grounding checks on LLM output.

Strategy (Hybrid):
    - Semantic similarity: cosine similarity between the generated answer
      and each retrieved chunk embedding (using the shared Encoder).
    - Lexical overlap: unigram overlap ratio between answer tokens and
      context tokens (tolerant of paraphrasing).
    - Hybrid score = 0.60 * semantic_sim + 0.40 * lexical_overlap
    - Grounded if hybrid_score >= 0.35  OR  semantic_sim >= 0.55
"""

import logging
import re
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from app.embeddings.encoder import Encoder

logger = logging.getLogger(__name__)


class ResponseValidator:
    """
    Checks whether the generated response is factually grounded in the
    retrieved context chunks.

    Strategy:
        Hybrid = 60% semantic cosine similarity (embedding-based)
               + 40% unigram lexical overlap.

    Grounded if:
        hybrid_score >= hybrid_threshold  OR
        semantic_similarity >= semantic_threshold (strong semantic match alone)
    """

    def __init__(
        self,
        encoder: Optional["Encoder"] = None,
        hybrid_threshold: float = 0.35,
        semantic_threshold: float = 0.55,
        lexical_threshold: float = 0.10,
        semantic_weight: float = 0.60,
        lexical_weight: float = 0.40,
        raise_on_fail: bool = False,
    ):
        """
        Args:
            encoder:            Shared Encoder instance for semantic similarity.
                                If None, falls back to lexical-only mode.
            hybrid_threshold:   Minimum hybrid score to consider grounded.
            semantic_threshold: Semantic similarity alone sufficient for grounding.
            lexical_threshold:  Minimum lexical overlap (informational, not gating).
            semantic_weight:    Weight of semantic similarity in hybrid score.
            lexical_weight:     Weight of lexical overlap in hybrid score.
            raise_on_fail:      If True, raise ValueError on ungrounded response.
        """
        self._encoder = encoder
        self.hybrid_threshold = hybrid_threshold
        self.semantic_threshold = semantic_threshold
        self.lexical_threshold = lexical_threshold
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.raise_on_fail = raise_on_fail

    # ── Public API ─────────────────────────────────────────────────────────────

    def validate(
        self,
        response: str,
        context_chunks: List[Tuple[dict, float]],
    ) -> dict:
        """
        Assess whether *response* is grounded in *context_chunks*.

        Returns:
            {
                "grounded": bool,
                "overlap_score": float,          # hybrid score (backwards-compat key)
                "semantic_similarity": float,
                "lexical_overlap": float,
                "hybrid_score": float,
                "warning": str | None,
            }
        """
        print("\n---------- Grounding Validation ----------")
        print(f"Answer: {response[:200]}")

        # ── Lexical overlap (unigram) against all context ──────────────────────
        context_text = " ".join(chunk.get("text", "") for chunk, _ in context_chunks)
        lexical = self._unigram_overlap(response, context_text)
        print(f"Lexical Overlap (unigram): {lexical:.4f}")

        # ── Semantic similarity against each chunk ─────────────────────────────
        semantic = self._max_semantic_similarity(response, context_chunks)
        print(f"Semantic Similarity (max cosine): {semantic:.4f}")

        # ── Hybrid score ────────────────────────────────────────────────────────
        hybrid = self.semantic_weight * semantic + self.lexical_weight * lexical
        print(f"Hybrid Score: {hybrid:.4f}  "
              f"(= {self.semantic_weight}×{semantic:.3f} + "
              f"{self.lexical_weight}×{lexical:.3f})")

        grounded = hybrid >= self.hybrid_threshold or semantic >= self.semantic_threshold
        print(f"Grounded: {grounded}  "
              f"(hybrid≥{self.hybrid_threshold} or semantic≥{self.semantic_threshold})")
        print("------------------------------------------")

        warning: str | None = None
        if not grounded:
            warning = (
                f"Response may not be grounded in retrieved context "
                f"(hybrid={hybrid:.3f}, semantic={semantic:.3f}, "
                f"lexical={lexical:.3f})."
            )
            logger.warning(warning)
            if self.raise_on_fail:
                raise ValueError(warning)

        return {
            "grounded": grounded,
            "overlap_score": round(hybrid, 4),   # kept for API backwards-compat
            "semantic_similarity": round(semantic, 4),
            "lexical_overlap": round(lexical, 4),
            "hybrid_score": round(hybrid, 4),
            "warning": warning,
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _tokenise(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _unigram_overlap(self, response: str, context: str) -> float:
        """Fraction of unique answer tokens that appear in the context."""
        resp_tokens = set(self._tokenise(response))
        ctx_tokens = set(self._tokenise(context))
        if not resp_tokens:
            return 0.0
        return len(resp_tokens & ctx_tokens) / len(resp_tokens)

    def _max_semantic_similarity(
        self,
        response: str,
        context_chunks: List[Tuple[dict, float]],
    ) -> float:
        """
        Compute cosine similarity between the answer and each chunk,
        return the maximum.  Falls back to 0.0 if encoder is unavailable.
        """
        if self._encoder is None or not context_chunks:
            logger.debug("Encoder not available — skipping semantic similarity.")
            return 0.0

        try:
            answer_emb = self._encoder.encode([response], normalize=True)[0]  # shape (dim,)

            chunk_texts = [chunk.get("text", "") for chunk, _ in context_chunks]
            chunk_embs = self._encoder.encode(chunk_texts, normalize=True)   # shape (N, dim)

            # Cosine similarity = dot product (vectors already L2-normalised)
            similarities = chunk_embs @ answer_emb                            # shape (N,)

            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            best_source = context_chunks[best_idx][0].get("source", "unknown")

            print(f"  Per-chunk semantic similarities:")
            for i, (sim, (chunk, _)) in enumerate(zip(similarities, context_chunks)):
                src = chunk.get("source", "?").split("\\")[-1]
                snippet = chunk.get("text", "")[:80].replace("\n", " ")
                marker = " ← best" if i == best_idx else ""
                print(f"    [{i}] {sim:.4f}  [{src}]  {snippet}...{marker}")

            return max(0.0, best_score)

        except Exception as exc:  # noqa: BLE001
            logger.error("Semantic similarity computation failed: %s", exc)
            return 0.0
