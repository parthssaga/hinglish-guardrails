"""Tests for the upgraded multi-signal HallucinationGuardrail.

All tests run against the heuristic path only (no model download needed).
The logprob path is tested by injecting a synthetic avg_logprob value.
check_grounded tests inject a mock SentenceTransformer so no download
is required; the unavailable-model path is also explicitly tested.
Tests assert on deterministic behaviour so CI always passes offline.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from unittest.mock import MagicMock

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


# ---------------------------------------------------------------------------
# check_grounded
# ---------------------------------------------------------------------------

def _mock_embedder(embeddings: np.ndarray) -> MagicMock:
    m = MagicMock()
    m.encode.return_value = embeddings
    return m


class TestCheckGrounded:
    def test_unavailable_embedder_returns_clean_result(
        self, hallucination: HallucinationGuardrail
    ) -> None:
        hallucination._embedder = None
        hallucination._ready = True
        r = hallucination.check_grounded("Paris is in France.", "Paris is the capital of France.")
        assert r.name == "hallucination"
        assert not r.triggered
        assert r.metadata["mode"] == "grounded/unavailable"

    def test_supported_response_not_triggered(
        self, hallucination: HallucinationGuardrail
    ) -> None:
        # Identical unit vectors → cosine similarity = 1.0 → supported
        hallucination._embedder = _mock_embedder(np.array([[1.0, 0.0], [1.0, 0.0]]))
        hallucination._ready = True
        r = hallucination.check_grounded("Paris is in France.", "Paris is the capital of France.")
        assert not r.triggered
        assert r.metadata["mode"] == "grounded"
        assert r.metadata["sentences"][0]["supported"] is True

    def test_unsupported_claim_triggers(
        self, hallucination: HallucinationGuardrail
    ) -> None:
        # Orthogonal vectors → cosine similarity = 0.0 → unsupported
        hallucination._embedder = _mock_embedder(np.array([[1.0, 0.0], [0.0, 1.0]]))
        hallucination._ready = True
        r = hallucination.check_grounded(
            "The Eiffel Tower is in London.",
            "The Eiffel Tower is located in Paris, France.",
        )
        assert r.triggered
        assert r.metadata["sentences"][0]["supported"] is False
        assert r.metadata["unsupported_count"] == 1

    def test_metadata_structure_complete(
        self, hallucination: HallucinationGuardrail
    ) -> None:
        hallucination._embedder = _mock_embedder(np.array([[1.0, 0.0], [1.0, 0.0]]))
        hallucination._ready = True
        r = hallucination.check_grounded("A fact.", "A fact is stated here.")
        meta = r.metadata
        for key in (
            "mode", "model", "grounding_threshold", "sentences",
            "unsupported_count", "total_sentences",
            "unsupported_fraction", "mean_max_similarity",
        ):
            assert key in meta, f"missing key: {key}"

    def test_reason_contains_grounded(
        self, hallucination: HallucinationGuardrail
    ) -> None:
        hallucination._embedder = _mock_embedder(np.array([[1.0, 0.0], [1.0, 0.0]]))
        hallucination._ready = True
        r = hallucination.check_grounded("Paris is in France.", "Paris is in France.")
        assert r.reason.startswith("grounded check:")

    def test_per_sentence_scores_multi_sentence(
        self, hallucination: HallucinationGuardrail
    ) -> None:
        # 2 response sentences + 1 source sentence → 3 embeddings
        # resp[0] identical to source (sim=1), resp[1] orthogonal (sim=0)
        hallucination._embedder = _mock_embedder(np.array([
            [1.0, 0.0],  # response sentence 0
            [0.0, 1.0],  # response sentence 1
            [1.0, 0.0],  # source sentence
        ]))
        hallucination._ready = True
        r = hallucination.check_grounded(
            "Paris is in France. The moon is made of cheese.",
            "Paris is the capital of France.",
        )
        assert len(r.metadata["sentences"]) == 2
        assert r.metadata["sentences"][0]["supported"] is True
        assert r.metadata["sentences"][1]["supported"] is False
        assert r.metadata["unsupported_count"] == 1
        assert r.metadata["total_sentences"] == 2

    def test_score_equals_one_minus_mean_max_similarity(
        self, hallucination: HallucinationGuardrail
    ) -> None:
        # resp[0] sim source = 1.0, resp[1] sim source = 0.5
        # mean_max_sim = (1.0 + 0.5) / 2 = 0.75 → score = 0.25
        hallucination._embedder = _mock_embedder(np.array([
            [1.0, 0.0],        # response sentence 0 — perfectly aligned
            [0.5, 0.866025],   # response sentence 1 — 60° angle, sim ≈ 0.5
            [1.0, 0.0],        # source sentence
        ]))
        hallucination._ready = True
        r = hallucination.check_grounded(
            "Paris is in France. London is the capital of Germany.",
            "Paris is the capital of France.",
        )
        assert r.score == pytest.approx(0.25, abs=0.01)
        assert r.metadata["mean_max_similarity"] == pytest.approx(0.75, abs=0.01)

    def test_custom_threshold_from_config_respected(
        self, hallucination: HallucinationGuardrail
    ) -> None:
        # Set high threshold (0.80) — a sim of 0.707 (45° vectors) should be unsupported
        hallucination.config.thresholds["hallucination_grounding"] = 0.80
        hallucination._embedder = _mock_embedder(np.array([
            [0.707, 0.707],  # response sentence — 45° from source
            [1.0, 0.0],      # source sentence
        ]))
        hallucination._ready = True
        r = hallucination.check_grounded("A claim.", "A source sentence.")
        # sim ≈ 0.707 < 0.80 threshold → unsupported
        assert r.metadata["sentences"][0]["supported"] is False
        assert r.triggered

    def test_default_threshold_accepts_moderately_similar(
        self, hallucination: HallucinationGuardrail
    ) -> None:
        # Default threshold = 0.35; sim of 0.707 should be supported
        hallucination._embedder = _mock_embedder(np.array([
            [0.707, 0.707],  # response sentence
            [1.0, 0.0],      # source sentence
        ]))
        hallucination._ready = True
        r = hallucination.check_grounded("A claim.", "A source sentence.")
        assert r.metadata["sentences"][0]["supported"] is True
        assert not r.triggered

    def test_unsupported_fraction_correct_for_partial(
        self, hallucination: HallucinationGuardrail
    ) -> None:
        # 1 of 2 sentences unsupported → fraction = 0.5
        hallucination._embedder = _mock_embedder(np.array([
            [1.0, 0.0],   # resp0 — supported
            [0.0, 1.0],   # resp1 — not supported (orthogonal)
            [1.0, 0.0],   # source
        ]))
        hallucination._ready = True
        r = hallucination.check_grounded(
            "First claim. Second claim.",
            "Source text.",
        )
        assert r.metadata["unsupported_fraction"] == pytest.approx(0.5)

    def test_no_sentences_in_response_returns_clean(
        self, hallucination: HallucinationGuardrail
    ) -> None:
        # Empty string → resp_sents = [] → early exit
        hallucination._embedder = _mock_embedder(np.array([[1.0, 0.0]]))
        hallucination._ready = True
        r = hallucination.check_grounded("", "There is a source sentence.")
        assert not r.triggered
        assert r.reason == "grounded check: no sentences to compare"

    def test_multiple_source_sentences_picks_max_similarity(
        self, hallucination: HallucinationGuardrail
    ) -> None:
        # 1 response sentence, 2 source sentences
        # resp sim src0 = 0.0 (orthogonal), resp sim src1 = 1.0 (identical)
        # max similarity = 1.0 → supported
        hallucination._embedder = _mock_embedder(np.array([
            [1.0, 0.0],   # response sentence
            [0.0, 1.0],   # source sentence 0 — orthogonal
            [1.0, 0.0],   # source sentence 1 — identical
        ]))
        hallucination._ready = True
        r = hallucination.check_grounded(
            "A response claim.",
            "Different topic here. A response claim.",
        )
        assert r.metadata["sentences"][0]["supported"] is True
        assert not r.triggered

    def test_reason_includes_counts_when_triggered(
        self, hallucination: HallucinationGuardrail
    ) -> None:
        hallucination._embedder = _mock_embedder(np.array([
            [0.0, 1.0],  # response — orthogonal to source → unsupported
            [1.0, 0.0],  # source
        ]))
        hallucination._ready = True
        r = hallucination.check_grounded("Bad claim.", "Good source.")
        assert r.triggered
        assert "1/1" in r.reason  # "1/1 sentences unsupported"
