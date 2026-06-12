"""
test_generation.py — Unit tests for the generation layer.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.generation.guardrails import Guardrails, GuardrailViolation
from app.generation.prompt_builder import PromptBuilder
from app.generation.response_validator import ResponseValidator


# ── Guardrails ────────────────────────────────────────────────────────────────

class TestGuardrails:
    def test_valid_query_passes(self):
        g = Guardrails()
        result = g.pre_check("What is the refund policy?")
        assert result == "What is the refund policy?"

    def test_empty_query_raises(self):
        g = Guardrails()
        with pytest.raises(GuardrailViolation):
            g.pre_check("")

    def test_too_long_query_raises(self):
        g = Guardrails(max_query_length=10)
        with pytest.raises(GuardrailViolation):
            g.pre_check("A" * 11)

    def test_blocked_pattern_raises(self):
        g = Guardrails()
        with pytest.raises(GuardrailViolation):
            g.pre_check("What is my password?")

    def test_post_check_passes_valid(self):
        g = Guardrails()
        result = g.post_check("The refund policy allows 30 days.")
        assert "refund" in result

    def test_post_check_empty_raises(self):
        g = Guardrails()
        with pytest.raises(GuardrailViolation):
            g.post_check("")


# ── PromptBuilder ─────────────────────────────────────────────────────────────

class TestPromptBuilder:
    def test_build_renders_query(self, tmp_path, monkeypatch):
        # Point prompts dir to a temp directory
        import app.generation.prompt_builder as pb_module
        prompts_dir = tmp_path / "prompts" / "v1"
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "qa_prompt.txt").write_text(
            "Context:\n{{CONTEXT}}\n\nQuestion: {{QUERY}}\n\nAnswer:", encoding="utf-8"
        )
        monkeypatch.setattr(pb_module, "_PROMPTS_DIR", tmp_path / "prompts")
        builder = PromptBuilder(prompt_version="v1", prompt_type="qa")
        chunks = [({"text": "Dogs are mammals.", "source": "bio.pdf"}, 0.9)]
        prompt = builder.build("What are dogs?", chunks)
        assert "What are dogs?" in prompt
        assert "Dogs are mammals." in prompt

    def test_missing_template_raises(self):
        with pytest.raises(FileNotFoundError):
            PromptBuilder(prompt_version="v99", prompt_type="nonexistent")


# ── ResponseValidator ─────────────────────────────────────────────────────────

class TestResponseValidator:
    def _chunks(self, text):
        return [({"text": text}, 0.9)]

    def test_grounded_response(self):
        validator = ResponseValidator(min_overlap=0.05)
        ctx = "The company was founded in 2010 in New York City."
        response = "The company was founded in 2010."
        result = validator.validate(response, self._chunks(ctx))
        assert result["grounded"] is True
        assert result["overlap_score"] > 0

    def test_ungrounded_response(self):
        validator = ResponseValidator(min_overlap=0.9, raise_on_fail=False)
        ctx = "cats and dogs"
        response = "quantum entanglement is a physics phenomenon"
        result = validator.validate(response, self._chunks(ctx))
        assert result["grounded"] is False
        assert result["warning"] is not None

    def test_raise_on_fail(self):
        validator = ResponseValidator(min_overlap=0.99, raise_on_fail=True)
        with pytest.raises(ValueError):
            validator.validate("completely unrelated text", self._chunks("something else"))
