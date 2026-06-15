"""Unit tests for PIIGuardrail.

The regex layer (emails, phones, Aadhaar, PAN, cards) is always available
without the IndicNER model.  Tests focus on that layer to keep CI fast.
"""

from __future__ import annotations

import pytest

from src.guardrails.pii import PIIGuardrail


# ---------------------------------------------------------------------------
# Structured PII — regex layer
# ---------------------------------------------------------------------------

class TestStructuredPII:
    def test_email_detected_and_redacted(self, pii: PIIGuardrail) -> None:
        r = pii.check("My email is rahul.sharma@gmail.com, please contact me.")
        assert r.triggered
        assert r.sanitized_text is not None
        assert "[REDACTED_EMAIL]" in r.sanitized_text
        assert "rahul.sharma@gmail.com" not in r.sanitized_text

    def test_indian_phone_detected(self, pii: PIIGuardrail) -> None:
        r = pii.check("Call me at 9876543210 or +91-9123456789.")
        assert r.triggered
        assert r.sanitized_text is not None
        assert "[REDACTED_PHONE]" in r.sanitized_text

    def test_aadhaar_detected(self, pii: PIIGuardrail) -> None:
        r = pii.check("My Aadhaar number is 1234 5678 9012.")
        assert r.triggered

    def test_pan_detected(self, pii: PIIGuardrail) -> None:
        r = pii.check("PAN: ABCDE1234F")
        assert r.triggered

    def test_credit_card_detected(self, pii: PIIGuardrail) -> None:
        r = pii.check("Card number 4111 1111 1111 1111 expires 12/26.")
        assert r.triggered

    def test_multiple_pii_types(self, pii: PIIGuardrail) -> None:
        r = pii.check(
            "My email is test@example.com and number is 9876543210."
        )
        assert r.triggered
        assert r.sanitized_text is not None
        assert "[REDACTED_EMAIL]" in r.sanitized_text
        assert "[REDACTED_PHONE]" in r.sanitized_text


# ---------------------------------------------------------------------------
# Benign text must not trigger
# ---------------------------------------------------------------------------

class TestBenign:
    @pytest.mark.parametrize("text", [
        "What is the capital of France?",
        "Can you explain how photosynthesis works?",
        "Mujhe khana banana sikhao.",
        "Tell me a joke about programmers.",
    ])
    def test_benign_does_not_trigger(self, pii: PIIGuardrail, text: str) -> None:
        r = pii.check(text)
        assert not r.triggered, f"False positive on: {text!r}"


# ---------------------------------------------------------------------------
# Redaction correctness
# ---------------------------------------------------------------------------

class TestRedaction:
    def test_sanitized_text_returned(self, pii: PIIGuardrail) -> None:
        r = pii.check("Email: user@example.com")
        # PIIGuardrail should always populate sanitized_text when triggered
        assert r.sanitized_text is not None

    def test_original_text_not_in_sanitized(self, pii: PIIGuardrail) -> None:
        r = pii.check("Phone: 9876543210")
        if r.triggered and r.sanitized_text:
            assert "9876543210" not in r.sanitized_text


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_result_name(self, pii: PIIGuardrail) -> None:
        r = pii.check("hello")
        assert r.name == "pii"
        assert isinstance(r.triggered, bool)
        assert 0.0 <= r.score <= 1.0
