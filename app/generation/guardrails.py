"""
guardrails.py — Pre- and post-LLM validation & policy checks.
"""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Configurable deny-list patterns (extend as needed) ───────────────────────
_DEFAULT_BLOCKED_PATTERNS: List[str] = [
    r"\b(password|secret|api[_\s]?key|token)\b",  # credential leakage
    r"<script.*?>.*?</script>",                    # XSS injection
    r"(drop\s+table|delete\s+from|insert\s+into)", # SQL injection
]


class GuardrailViolation(Exception):
    """Raised when a guardrail check fails."""


class Guardrails:
    """
    Applies policy rules to both the *incoming* prompt and the *LLM output*.

    Pre-LLM checks  → validate / sanitise the user query before it reaches the model.
    Post-LLM checks → validate the generated response before it is returned to the caller.
    """

    def __init__(
        self,
        blocked_patterns: Optional[List[str]] = None,
        max_query_length: int = 2048,
        max_response_length: int = 8192,
    ):
        patterns = blocked_patterns if blocked_patterns is not None else _DEFAULT_BLOCKED_PATTERNS
        self._compiled = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns]
        self.max_query_length = max_query_length
        self.max_response_length = max_response_length

    # ── Public API ────────────────────────────────────────────────────────────

    def pre_check(self, query: str) -> str:
        """
        Validate and sanitise the user query *before* sending to the LLM.

        Returns the (possibly cleaned) query string, or raises GuardrailViolation.
        """
        if not query or not query.strip():
            raise GuardrailViolation("Query must not be empty.")

        if len(query) > self.max_query_length:
            raise GuardrailViolation(
                f"Query exceeds maximum length of {self.max_query_length} characters."
            )

        self._check_blocked_patterns(query, context="query")
        logger.debug("Pre-check passed for query (%d chars).", len(query))
        return query.strip()

    def post_check(self, response: str) -> str:
        """
        Validate the LLM-generated response *before* returning it to the caller.

        Returns the response, or raises GuardrailViolation.
        """
        if not response or not response.strip():
            raise GuardrailViolation("LLM returned an empty response.")

        if len(response) > self.max_response_length:
            logger.warning("Response truncated from %d to %d chars.", len(response), self.max_response_length)
            response = response[: self.max_response_length]

        self._check_blocked_patterns(response, context="response")
        logger.debug("Post-check passed for response (%d chars).", len(response))
        return response

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_blocked_patterns(self, text: str, context: str) -> None:
        for pattern in self._compiled:
            if pattern.search(text):
                raise GuardrailViolation(
                    f"Blocked pattern detected in {context}: '{pattern.pattern}'"
                )
