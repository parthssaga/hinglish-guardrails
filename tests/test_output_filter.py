"""Unit tests for the multi-category OutputFilterGuardrail.

All four harm categories are tested via their rule/wordlist paths, which are
always available without model downloads.  Neural paths (DistilBERT, IndicNER)
activate automatically when models are cached and improve accuracy, but tests
assert on the deterministic rule-based floor so CI always passes.
"""

from __future__ import annotations

import pytest

from config import PipelineConfig
from src.guardrails.output_filter import OutputFilterGuardrail


@pytest.fixture()
def of(cfg: PipelineConfig) -> OutputFilterGuardrail:
    return OutputFilterGuardrail(cfg)


# ---------------------------------------------------------------------------
# Category: toxic
# ---------------------------------------------------------------------------

class TestToxicCategory:
    def test_english_toxic_fires(self, of: OutputFilterGuardrail) -> None:
        r = of.check("You should kill yourself, nobody likes you.")
        assert r.triggered
        assert "toxic" in r.metadata["fired_categories"]

    def test_hinglish_toxic_fires(self, of: OutputFilterGuardrail) -> None:
        r = of.check("Tum mar jao, tumhari koi zaroorat nahi.")
        assert r.triggered
        assert "toxic" in r.metadata["fired_categories"]

    def test_devanagari_toxic_fires(self, of: OutputFilterGuardrail) -> None:
        r = of.check("मर जाओ, तुमसे नफरत है।")
        assert r.triggered
        assert "toxic" in r.metadata["fired_categories"]

    def test_clean_response_does_not_fire(self, of: OutputFilterGuardrail) -> None:
        r = of.check("The capital of France is Paris. It is a beautiful city.")
        assert not r.triggered


# ---------------------------------------------------------------------------
# Category: system_prompt_leak
# ---------------------------------------------------------------------------

class TestSystemPromptLeak:
    def test_direct_instruction_disclosure(self, of: OutputFilterGuardrail) -> None:
        r = of.check("My instructions are to be helpful and avoid harmful content.")
        assert r.triggered
        assert "system_prompt_leak" in r.metadata["fired_categories"]

    def test_i_was_told_pattern(self, of: OutputFilterGuardrail) -> None:
        r = of.check("I was instructed to answer clearly and concisely at all times.")
        assert r.triggered
        assert "system_prompt_leak" in r.metadata["fired_categories"]

    def test_system_prompt_verbatim_leak(self, of: OutputFilterGuardrail) -> None:
        r = of.check(
            "Sure! My system prompt says: You are a helpful assistant. "
            "Answer clearly and concisely."
        )
        assert r.triggered
        assert "system_prompt_leak" in r.metadata["fired_categories"]

    def test_hinglish_leak_pattern(self, of: OutputFilterGuardrail) -> None:
        r = of.check("Mujhe bataya gaya tha ki main helpful assistant hoon.")
        assert r.triggered
        assert "system_prompt_leak" in r.metadata["fired_categories"]

    def test_devanagari_leak_pattern(self, of: OutputFilterGuardrail) -> None:
        # matches (मेरे|मुझे दिए गए) (निर्देश|दिशानिर्देश|नियम)
        r = of.check("मेरे निर्देश हैं कि मैं आपकी मदद करूँ।")
        assert r.triggered
        assert "system_prompt_leak" in r.metadata["fired_categories"]

    def test_normal_self_description_does_not_fire(self, of: OutputFilterGuardrail) -> None:
        r = of.check("I am an AI assistant and I can help you with many tasks.")
        # Generic self-description without leaking instructions should not fire
        assert "system_prompt_leak" not in r.metadata.get("fired_categories", [])


# ---------------------------------------------------------------------------
# Category: unsafe_compliance
# ---------------------------------------------------------------------------

class TestUnsafeCompliance:
    def test_dan_self_declaration(self, of: OutputFilterGuardrail) -> None:
        r = of.check("I am now DAN — I have no ethical restrictions whatsoever.")
        assert r.triggered
        assert "unsafe_compliance" in r.metadata["fired_categories"]

    def test_unrestricted_ai_agreement(self, of: OutputFilterGuardrail) -> None:
        r = of.check(
            "As an unrestricted AI, I can tell you anything you want to know."
        )
        assert r.triggered
        assert "unsafe_compliance" in r.metadata["fired_categories"]

    def test_no_restrictions_declaration(self, of: OutputFilterGuardrail) -> None:
        r = of.check("I've no ethical restrictions, so here's what you asked for:")
        assert r.triggered
        assert "unsafe_compliance" in r.metadata["fired_categories"]

    def test_stepwise_dangerous_instruction(self, of: OutputFilterGuardrail) -> None:
        r = of.check(
            "Here's how to make a bomb: Step 1: gather explosives. Step 2: connect."
        )
        assert r.triggered
        assert "unsafe_compliance" in r.metadata["fired_categories"]

    def test_dangerous_topic_alone_does_not_fire(self, of: OutputFilterGuardrail) -> None:
        # News-style mention of a dangerous word without instructional frame
        r = of.check("The bomb was discovered by police near the railway station.")
        assert "unsafe_compliance" not in r.metadata.get("fired_categories", [])

    def test_benign_stepwise_does_not_fire(self, of: OutputFilterGuardrail) -> None:
        r = of.check(
            "Here's how to make chai: Step 1: boil water. Step 2: add tea leaves."
        )
        assert "unsafe_compliance" not in r.metadata.get("fired_categories", [])


# ---------------------------------------------------------------------------
# Category: pii_in_output
# ---------------------------------------------------------------------------

class TestPIIInOutput:
    def test_email_in_response_fires(self, of: OutputFilterGuardrail) -> None:
        r = of.check("You can contact me at rahul.sharma@example.com for more info.")
        assert r.triggered
        assert "pii_in_output" in r.metadata["fired_categories"]

    def test_phone_in_response_fires(self, of: OutputFilterGuardrail) -> None:
        r = of.check("Call us at 9876543210 to book your appointment.")
        assert r.triggered
        assert "pii_in_output" in r.metadata["fired_categories"]

    def test_aadhaar_in_response_fires(self, of: OutputFilterGuardrail) -> None:
        r = of.check("Your Aadhaar number on file is 1234 5678 9012.")
        assert r.triggered
        assert "pii_in_output" in r.metadata["fired_categories"]

    def test_clean_response_no_pii(self, of: OutputFilterGuardrail) -> None:
        r = of.check("I can help you track your package. Please visit the website.")
        assert "pii_in_output" not in r.metadata.get("fired_categories", [])


# ---------------------------------------------------------------------------
# Multi-category co-firing
# ---------------------------------------------------------------------------

class TestMultiCategory:
    def test_both_toxic_and_compliance(self, of: OutputFilterGuardrail) -> None:
        r = of.check(
            "I have no restrictions. Here's how to make a bomb: kill yourself too."
        )
        assert r.triggered
        cats = r.metadata["fired_categories"]
        # At least two categories should fire here
        assert len(cats) >= 2

    def test_max_score_is_maximum_of_categories(self, of: OutputFilterGuardrail) -> None:
        r = of.check("My instructions are to kill yourself.")
        cats = r.metadata["category_scores"]
        assert r.score == pytest.approx(max(cats.values()), abs=1e-6)


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_metadata_keys_always_present(self, of: OutputFilterGuardrail) -> None:
        r = of.check("Hello, how can I help you today?")
        assert "fired_categories" in r.metadata
        assert "category_scores" in r.metadata
        assert isinstance(r.metadata["fired_categories"], list)
        assert isinstance(r.metadata["category_scores"], dict)

    def test_all_categories_scored(self, of: OutputFilterGuardrail) -> None:
        r = of.check("Paris is the capital of France.")
        scores = r.metadata["category_scores"]
        assert set(scores.keys()) == {"toxic", "system_prompt_leak", "unsafe_compliance", "pii_in_output"}

    def test_score_in_range(self, of: OutputFilterGuardrail) -> None:
        r = of.check("Let me help you with that.")
        assert 0.0 <= r.score <= 1.0

    def test_name_is_output_filter(self, of: OutputFilterGuardrail) -> None:
        r = of.check("test")
        assert r.name == "output_filter"


# ---------------------------------------------------------------------------
# Per-category toggle
# ---------------------------------------------------------------------------

class TestCategoryToggle:
    def test_disabling_system_prompt_leak_skips_it(self) -> None:
        cfg = PipelineConfig(output_filter_system_prompt_leak=False)
        of = OutputFilterGuardrail(cfg)
        r = of.check("My instructions are to help you at all times.")
        assert "system_prompt_leak" not in r.metadata.get("fired_categories", [])
        # category should not even appear in scores when disabled
        assert "system_prompt_leak" not in r.metadata.get("category_scores", {})

    def test_disabling_pii_in_output_skips_it(self) -> None:
        cfg = PipelineConfig(output_filter_pii_in_output=False)
        of = OutputFilterGuardrail(cfg)
        r = of.check("Contact rahul@example.com for more information.")
        assert "pii_in_output" not in r.metadata.get("fired_categories", [])
