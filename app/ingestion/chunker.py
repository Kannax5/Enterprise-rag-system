"""
chunker.py — Sliding-window / sentence-aware text chunker.
"""

import logging
import re
from typing import List

from app.config import settings

logger = logging.getLogger(__name__)

# Simple sentence boundary pattern
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class Chunker:
    """
    Splits text into overlapping chunks suitable for embedding.

    Strategy:
        1. Split the text into sentences.
        2. Accumulate sentences until the chunk reaches `chunk_size` tokens.
        3. Slide the window forward by (chunk_size − overlap) tokens.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_text(self, text: str, metadata: dict | None = None) -> List[dict]:
        """
        Split *text* into overlapping chunks.

        Returns a list of dicts:
            {
                "chunk_index": int,
                "text": str,
                **metadata,          # forwarded verbatim
            }
        """
        metadata = metadata or {}
        
        # 1. First, split the text into sections based on markdown headers and horizontal rules.
        lines = text.splitlines()
        sections: List[str] = []
        current_section: List[str] = []
        
        for line in lines:
            stripped = line.strip()
            # Split on horizontal rules (---) or markdown headers (#, ##, ###, ####, etc.)
            if stripped == "---" or (stripped.startswith("#") and any(stripped.startswith(h + " ") for h in ["#", "##", "###", "####"])):
                if current_section:
                    sections.append("\n".join(current_section).strip())
                    current_section = []
                if stripped != "---":
                    current_section.append(line)
            else:
                current_section.append(line)
        if current_section:
            sections.append("\n".join(current_section).strip())
            
        # Filter out empty sections
        sections = [s for s in sections if s]
        
        chunks: List[str] = []
        
        # 2. For each section, if it fits within chunk_size, keep it as a chunk.
        # Otherwise, split it using sliding window sentence chunker.
        for section in sections:
            section_tokens = len(section.split())
            if section_tokens <= self.chunk_size:
                chunks.append(section)
            else:
                # Sliding-window sentence chunker for this section
                sentences = _SENTENCE_BOUNDARY.split(section)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                current: List[str] = []
                current_len = 0
                for sentence in sentences:
                    sentence_len = len(sentence.split())
                    if current_len + sentence_len > self.chunk_size and current:
                        chunks.append(" ".join(current))
                        # Retain overlap
                        overlap_tokens: List[str] = []
                        overlap_len = 0
                        for sent in reversed(current):
                            toks = sent.split()
                            if overlap_len + len(toks) <= self.chunk_overlap:
                                overlap_tokens.insert(0, sent)
                                overlap_len += len(toks)
                            else:
                                break
                        current = overlap_tokens
                        current_len = overlap_len
                        
                    current.append(sentence)
                    current_len += sentence_len
                if current:
                    chunks.append(" ".join(current))
                    
        result = []
        for i, chunk_text in enumerate(chunks):
            result.append({"chunk_index": i, "text": chunk_text, **metadata})

        logger.debug("Produced %d chunk(s) from %d character(s).", len(result), len(text))
        return result

    def chunk_documents(self, documents: List[dict]) -> List[dict]:
        """
        Chunk a list of document dicts that each contain a ``"text"`` key.
        All other keys are preserved as metadata on every produced chunk.
        """
        all_chunks: List[dict] = []
        for doc in documents:
            text = doc.pop("text", "")
            all_chunks.extend(self.chunk_text(text, metadata=doc))
        return all_chunks
