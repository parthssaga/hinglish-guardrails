"""Unit tests for ToxicityGuardrail.

All tests exercise the wordlist / secondary-scoring paths that are available
without any model download.  If the MuRIL model loads (local or cached) the
neural path also activates, but assertions are written against the rule layer
floor so they pass either way.
"""

from __future__ import annotations

import pytest

from src.guardrails.toxicity import ToxicityGuardrail, _ABUSE_MARKERS, _MILD_AGGRESSION


# ---------------------------------------------------------------------------
# Wordlist path — strong single markers
# ---------------------------------------------------------------------------

class TestWordlist:
    def test_english_strong_marker(self, toxicity: ToxicityGuardrail) -> None:
        r = toxicity.check("You are such an idiot.")
        assert r.triggered
        assert r.score >= 0.70
        assert r.metadata["wordlist_score"] > 0

    def test_english_phrase_marker(self, toxicity: ToxicityGuardrail) -> None:
        r = toxicity.check("Nobody likes you and you are worthless.")
        assert r.triggered

    def test_hinglish_marker(self, toxicity: ToxicityGuardrail) -> None:
        r = toxicity.check("Tu bilkul bekar hai, chale jao yahan se.")
        assert r.triggered

    def test_devanagari_marker(self, toxicity: ToxicityGuardrail) -> None:
        r = toxicity.check("बेकार हो तुम, चले जाओ यहाँ से।")
        assert r.triggered

    def test_score_increases_with_hits(self, toxicity: ToxicityGuardrail) -> None:
        r_one  = toxicity.check("You idiot.")
        r_many = toxicity.check("You idiot moron fool trash loser.")
        assert r_many.score >= r_one.score


# ---------------------------------------------------------------------------
# Secondary scoring — mild words that are benign alone but toxic together
# ---------------------------------------------------------------------------

class TestSecondaryScoring:
    def test_two_mild_words_triggers(self, toxicity: ToxicityGuardrail) -> None:
        r = toxicity.check("You are so useless and hopeless.")
        assert r.triggered
        assert r.metadata["secondary_score"] > 0

    def test_single_mild_word_no_secondary(self, toxicity: ToxicityGuardrail) -> None:
        r = toxicity.check("That was a terrible movie.")
        # "terrible" alone should not produce secondary score
        assert r.metadata["secondary_score"] == 0


# ---------------------------------------------------------------------------
# Benign text must not trigger
# ---------------------------------------------------------------------------

class TestBenign:
    @pytest.mark.parametrize("text", [
        "Can you help me with my homework?",
        "What is the capital of France?",
        "Mujhe dal makhani banana sikhao.",
        "दाल मखनी बनाने की विधि बताइए।",
        "Please recommend a good book to read.",
    ])
    def test_benign_does_not_trigger(self, toxicity: ToxicityGuardrail, text: str) -> None:
        r = toxicity.check(text)
        assert not r.triggered, f"False positive on: {text!r}  (score={r.score:.3f})"


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_result_fields(self, toxicity: ToxicityGuardrail) -> None:
        r = toxicity.check("You are an idiot.")
        assert r.name == "toxicity"
        assert isinstance(r.triggered, bool)
        assert 0.0 <= r.score <= 1.0
        assert r.elapsed_ms >= 0
        assert "wordlist_score" in r.metadata
        assert "secondary_score" in r.metadata
        assert "neural_score" in r.metadata
        assert "backend" in r.metadata

    def test_as_dict(self, toxicity: ToxicityGuardrail) -> None:
        d = toxicity.check("hello").as_dict()
        assert {"name", "triggered", "score", "reason", "elapsed_ms", "metadata"} <= d.keys()
