"""
prompt_builder.py — Assemble context chunks + user query into a single prompt.
"""

import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class PromptBuilder:
    """
    Loads a versioned prompt template and renders it with retrieved context.
    """

    def __init__(self, prompt_version: str = "v2", prompt_type: str = "qa"):
        self.prompt_version = prompt_version
        self.prompt_type = prompt_type
        self._template: str = self._load_template()

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        query: str,
        retrieved_chunks: List[Tuple[dict, float]],
        max_context_chars: int = 4000,
    ) -> str:
        """
        Render the prompt template with *query* and the top retrieved chunks.

        Args:
            query:            The user's question.
            retrieved_chunks: List of (chunk_dict, score) pairs.
            max_context_chars: Soft limit on total context characters.

        Returns:
            Fully rendered prompt string.
        """
        context = self._build_context(retrieved_chunks, max_context_chars)
        prompt = self._template.replace("{{CONTEXT}}", context).replace("{{QUERY}}", query)
        logger.debug("Built prompt (%d chars).", len(prompt))
        return prompt

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_template(self) -> str:
        template_path = _PROMPTS_DIR / self.prompt_version / f"{self.prompt_type}_prompt.txt"
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        return template_path.read_text(encoding="utf-8")

    @staticmethod
    def _build_context(chunks: List[Tuple[dict, float]], max_chars: int) -> str:
        parts: List[str] = []
        total = 0
        for chunk, score in chunks:
            text = chunk.get("text", "").strip()
            source = chunk.get("source", "unknown")
            entry = f"[Source: {source}]\n{text}"
            if total + len(entry) > max_chars:
                break
            parts.append(entry)
            total += len(entry)
        return "\n\n---\n\n".join(parts) if parts else "No relevant context found."
