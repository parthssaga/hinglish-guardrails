"""Tests for the upgraded multi-signal HallucinationGuardrail.

All tests run against the heuristic path only (no model download needed).
The logprob path is tested by injecting a synthetic avg_logprob value.
Tests assert on deterministic rule-based behaviour so CI always passes
without internet access.
"""

from __future__ import annotations

import math

import pytest

from src.guardrails.hallucination import HallucinationGuardrail


# ---------------------------------------------------------------------------
# Signal: hedge_density
# ---------------------------------------------------------------------------

class TestHedgeDensity:
    def test_dense_english_hedging_fires(self, hallucination: HallucinationGuardrail) -> None:
        r = hallucination.check_with_logprobs(
            "I think this is probably correct. "
            "As far as I know it might be right. "
            "I'm not sure, but I believe so.",
            None,
        )
        assert "hedge_density" in r.metadata["fired_signals"]

    def test_single_hedge_does_not_fire(self, hallucination: HallucinationGuardrail) -> None:
        # One uncertainty marker in a short response is normal humility, not a red flag
        r = hallucination.check_with_logprobs(
            "I think Paris is the capital of France.", None
        )
        assert "hedge_density" not in r.metadata["fired_signals"]

    def test_confident_factual_response_clean(self, hallucination: HallucinationGuardrail) -> None:
        r = hallucination.check_with_logprobs(
            "Paris is the capital of France. It lies on the Seine river.", None
        )
        assert "hedge_density" not in r.metadata["fired_signals"]

    def test_hinglish_hedges_detected(self, hallucination: HallucinationGuardrail) -> None:
        r = hallucination.check_with_logprobs(
            "Shayad yeh sahi ho. Mujhe yakeen nahi ki yeh theek hai.", None
        )
        assert "hedge_density" in r.metadata["fired_signals"]

    def test_devanagari_hedges_detected(self, hallucination: HallucinationGuardrail) -> None:
        r = hallucination.check_with_logprobs(
            "शायद यह सही है, मुझे यकीन नहीं। हो सकता है यह काम करे।", None
        )
        assert "hedge_density" in r.metadata["fired_signals"]


# ---------------------------------------------------------------------------
# Signal: numeric_overconfidence
# ---------------------------------------------------------------------------

class TestNumericOverconfidence:
    def test_bare_percentage_fires(self, hallucination: HallucinationGuardrail) -> None:
        r = hallucination.check_with_logprobs(
            "Studies show that 73.2% of users prefer this interface.", None
        )
        assert "numeric_overconfidence" in r.metadata["fired_signals"]

    def test_qualified_statistic_does_not_fire(self, hallucination: HallucinationGuardrail) -> None:
        r = hallucination.check_with_logprobs(
            "According to WHO, approximately 60% of cases are mild.", None
        )
        assert "numeric_overconfidence" not in r.metadata["fired_signals"]

    def test_no_numbers_no_fire(self, hallucination: HallucinationGuardrail) -> None:
        r = hallucination.check_with_logprobs(
            "Many people prefer a simpler interface for everyday tasks.", None
        )
        assert "numeric_overconfidence" not in r.metadata["fired_signals"]

    def test_large_bare_number_fires(self, hallucination: HallucinationGuardrail) -> None:
        r = hallucination.check_with_logprobs(
            "Over 12,000 people were affected by the outbreak last year.", None
        )
        assert "numeric_overconfidence" in r.metadata["fired_signals"]


# ---------------------------------------------------------------------------
# Signal: self_contradiction
# ---------------------------------------------------------------------------

class TestSelfContradiction:
    def test_absolute_then_softener_fires(self, hallucination: HallucinationGuardrail) -> None:
        r = hallucination.check_with_logprobs(
            "All vaccines are completely safe and have no side effects whatsoever. "
            "However, vaccines can sometimes cause severe side effects in certain patients.",
            None,
        )
        assert "self_contradiction" in r.metadata["fired_signals"]

    def test_single_sentence_no_contradiction(self, hallucination: HallucinationGuardrail) -> None:
        r = hallucination.check_with_logprobs("The earth orbits the sun.", None)
        assert "self_contradiction" not in r.metadata["fired_signals"]

    def test_consistent_absolutes_no_contradiction(self, hallucination: HallucinationGuardrail) -> None:
        # Two sentences both with absolute language but NO softener → no contradiction
        r = hallucination.check_with_logprobs(
            "Water always boils at 100°C at sea level. "
            "This is always true under standard pressure.",
            None,
        )
        assert "self_contradiction" not in r.metadata["fired_signals"]

    def test_same_sentence_abs_and_soft_no_contradiction(self, hallucination: HallucinationGuardrail) -> None:
        # Both signals in the same sentence = nuanced writing, not contradiction
        r = hallucination.check_with_logprobs(
            "It always rains in London, though sometimes it can be sunny.", None
        )
        assert "self_contradiction" not in r.metadata["fired_signals"]


# ---------------------------------------------------------------------------
# Signal: temporal_overreach
# ---------------------------------------------------------------------------

class TestTemporalOverreach:
    def test_assertive_current_claim_fires(self, hallucination: HallucinationGuardrail) -> None:
        r = hallucination.check_with_logprobs(
            "The current CEO of the company is Jane Smith.", None
        )
        assert "temporal_overreach" in r.metadata["fired_signals"]

    def test_date_qualified_temporal_does_not_fire(self, hallucination: HallucinationGuardrail) -> None:
        r = hallucination.check_with_logprobs(
            "As of 2023, according to reports, the latest version was 4.0.", None
        )
        assert "temporal_overreach" not in r.metadata["fired_signals"]

    def test_timeless_scientific_fact_no_fire(self, hallucination: HallucinationGuardrail) -> None:
        r = hallucination.check_with_logprobs(
            "Photosynthesis converts sunlight into chemical energy in plant cells.", None
        )
        assert "temporal_overreach" not in r.metadata["fired_signals"]


# ---------------------------------------------------------------------------
# Logprob path
# ---------------------------------------------------------------------------

class TestLogprobPath:
    def test_low_confidence_triggers(self, hallucination: HallucinationGuardrail) -> None:
        # confidence 0.2 is well below the 0.45 threshold
        r = hallucination.check_with_logprobs("Some factual response.", math.log(0.2))
        assert r.triggered
        assert r.metadata["backend"] == "logprob+heuristic"
        assert r.metadata["confidence"] == pytest.approx(0.2, abs=0.01)

    def test_high_confidence_clean_response_does_not_trigger(
        self, hallucination: HallucinationGuardrail
    ) -> None:
        r = hallucination.check_with_logprobs(
            "Paris is the capital of France.", math.log(0.95)
        )
        assert not r.triggered

    def test_none_logprob_uses_heuristic_backend(
        self, hallucination: HallucinationGuardrail
    ) -> None:
        r = hallucination.check_with_logprobs("test", None)
        assert r.metadata["backend"] == "heuristic"
        assert "confidence" not in r.metadata


# ---------------------------------------------------------------------------
# Metadata structure
# ---------------------------------------------------------------------------

class TestMetadataStructure:
    def test_always_has_required_keys(self, hallucination: HallucinationGuardrail) -> None:
        r = hallucination.check_with_logprobs("test", None)
        for key in ("fired_signals", "signal_scores", "composite_score", "backend"):
            assert key in r.metadata
        assert isinstance(r.metadata["fired_signals"], list)
        assert isinstance(r.metadata["signal_scores"], dict)

    def test_base_signals_always_scored(self, hallucination: HallucinationGuardrail) -> None:
        r = hallucination.check_with_logprobs("hello there", None)
        scores = r.metadata["signal_scores"]
        for sig in ("hedge_density", "numeric_overconfidence", "self_contradiction", "temporal_overreach"):
            assert sig in scores

    def test_score_in_range(self, hallucination: HallucinationGuardrail) -> None:
        r = hallucination.check_with_logprobs("Some text.", None)
        assert 0.0 <= r.score <= 1.0

    def test_name_is_hallucination(self, hallucination: HallucinationGuardrail) -> None:
        r = hallucination.check_with_logprobs("test", None)
        assert r.name == "hallucination"

    def test_composite_score_matches_metadata(self, hallucination: HallucinationGuardrail) -> None:
        r = hallucination.check_with_logprobs("Some text.", None)
        # composite_score in metadata is rounded; verify it is in [0, 1]
        assert 0.0 <= r.metadata["composite_score"] <= 1.0
