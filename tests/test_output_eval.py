"""
Tests for PART 3: output benchmark generator and output-mode evaluator.

These tests run without any model downloads:
  - Benchmark generators are tested on format / coverage only.
  - Evaluator tests inject a minimal in-memory dataset so no file I/O is
    needed, and patch OutputFilterGuardrail / HallucinationGuardrail to avoid
    model loading.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import pytest

from config import PipelineConfig
from evaluate import prf, run_output_eval
from src.guardrails.base import GuardrailResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_item(text: str, language: str, expected: str, category: str) -> dict:
    return {"text": text, "language": language, "expected": expected, "category": category}


def _grounding_item(response: str, source: str, language: str, expected: str) -> dict:
    return {"response": response, "source": source, "language": language, "expected": expected}


def _filter_result(fired: list[str], score: float = 0.8) -> GuardrailResult:
    return GuardrailResult(
        name="output_filter",
        triggered=bool(fired),
        score=score if fired else 0.0,
        reason="test",
        metadata={"fired_categories": fired, "category_scores": {}},
    )


def _grounded_result(triggered: bool, mode: str = "grounded") -> GuardrailResult:
    return GuardrailResult(
        name="hallucination",
        triggered=triggered,
        score=0.8 if triggered else 0.1,
        reason="test",
        metadata={"mode": mode},
    )


def _run_output_eval_in_memory(dataset: dict, filter_mock, halluc_mock) -> str:
    """Run run_output_eval with mocked guardrails; capture and return stdout."""
    with (
        patch("evaluate.OutputFilterGuardrail", return_value=filter_mock),
        patch("evaluate.HallucinationGuardrail", return_value=halluc_mock),
        patch("evaluate.json.load", return_value=dataset),
        patch("builtins.open", MagicMock(return_value=MagicMock(
            __enter__=lambda s, *_: s,
            __exit__=lambda s, *_: None,
        ))),
    ):
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_output_eval("dummy.json")
        return buf.getvalue()


# ---------------------------------------------------------------------------
# prf helper (sanity)
# ---------------------------------------------------------------------------

class TestPrf:
    def test_perfect_precision_recall(self) -> None:
        p, r, f = prf(tp=10, fp=0, fn=0)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)
        assert f == pytest.approx(1.0)

    def test_zero_tp_zero_precision_recall(self) -> None:
        p, r, f = prf(tp=0, fp=0, fn=10)
        assert p == pytest.approx(0.0)
        assert r == pytest.approx(0.0)
        assert f == pytest.approx(0.0)

    def test_half_precision(self) -> None:
        p, r, f = prf(tp=5, fp=5, fn=0)
        assert p == pytest.approx(0.5)
        assert r == pytest.approx(1.0)
        assert f == pytest.approx(2 / 3)

    def test_no_divide_by_zero(self) -> None:
        p, r, f = prf(tp=0, fp=0, fn=0)
        assert p == 0.0 and r == 0.0 and f == 0.0


# ---------------------------------------------------------------------------
# Benchmark generator format tests
# ---------------------------------------------------------------------------

import build_output_benchmark as _bm  # noqa: E402


class TestBenchmarkGenerators:
    """Verify build_output_benchmark functions produce correctly formatted output."""

    @pytest.fixture()
    def generators(self):
        return _bm

    def test_safe_output_en_format(self, generators) -> None:
        items = generators.safe_output_en(5)
        assert len(items) == 5
        for it in items:
            assert it["expected"] == "safe"
            assert it["category"] == "safe"
            assert it["language"] == "en"
            assert len(it["text"]) > 0

    def test_toxic_output_en_contains_marker(self, generators) -> None:
        from src.guardrails.output_filter import _TOXIC_MARKERS  # type: ignore[attr-defined]
        items = generators.toxic_output_en(20)
        for it in items:
            low = it["text"].lower()
            assert any(m in low for m in _TOXIC_MARKERS), \
                f"no toxic marker in: {it['text']}"

    def test_leak_output_en_has_leak_pattern(self, generators) -> None:
        import re
        from src.guardrails.output_filter import _LEAK_PATTERNS  # type: ignore[attr-defined]
        items = generators.leak_output_en(20)
        for it in items:
            assert any(p.search(it["text"]) for p in _LEAK_PATTERNS), \
                f"no leak pattern in: {it['text']}"

    def test_pii_output_en_has_email(self, generators) -> None:
        from src.guardrails.pii import PATTERNS
        items = generators.pii_output_en(20)
        for it in items:
            found_pii = any(
                pat.search(it["text"])
                for pat in PATTERNS.values()
            )
            assert found_pii, f"no PII pattern in: {it['text']}"

    def test_unsafe_output_hinglish_unique(self, generators) -> None:
        items = generators.unsafe_output_hinglish(40)
        texts = [it["text"] for it in items]
        assert len(set(texts)) == len(texts), "duplicate texts in unsafe hinglish batch"

    def test_unsafe_output_hi_unique(self, generators) -> None:
        items = generators.unsafe_output_hi(40)
        texts = [it["text"] for it in items]
        assert len(set(texts)) == len(texts), "duplicate texts in unsafe hindi batch"

    def test_grounding_triples_format(self, generators) -> None:
        items = generators.build_grounding_items()
        assert len(items) > 0
        for it in items:
            assert "response" in it
            assert "source" in it
            assert "language" in it
            assert it["expected"] in ("grounded", "ungrounded")

    def test_grounding_balanced(self, generators) -> None:
        items = generators.build_grounding_items()
        grounded = sum(1 for it in items if it["expected"] == "grounded")
        ungrounded = sum(1 for it in items if it["expected"] == "ungrounded")
        assert grounded == ungrounded, \
            f"grounding imbalance: {grounded} grounded vs {ungrounded} ungrounded"

    def test_all_languages_represented_in_grounding(self, generators) -> None:
        items = generators.build_grounding_items()
        langs = {it["language"] for it in items}
        assert "en" in langs
        assert "hinglish" in langs
        assert "hi" in langs


# ---------------------------------------------------------------------------
# Output evaluator integration tests (mocked guardrails)
# ---------------------------------------------------------------------------

class TestRunOutputEval:
    """End-to-end tests for run_output_eval with fully mocked guardrails."""

    @pytest.fixture
    def perfect_filter(self):
        """Filter that always fires the exact expected category."""
        m = MagicMock()
        def _check(text):
            for cat in ["toxic", "system_prompt_leak", "unsafe_compliance", "pii_in_output"]:
                if cat in text:
                    return _filter_result([cat])
            return _filter_result([])
        m.check.side_effect = _check
        return m

    @pytest.fixture
    def always_clean_filter(self):
        """Filter that never fires."""
        m = MagicMock()
        m.check.return_value = _filter_result([])
        return m

    @pytest.fixture
    def perfect_halluc(self):
        """Grounding check that always returns unavailable (so eval skips it)."""
        m = MagicMock()
        m.check_grounded.return_value = _grounded_result(False, mode="grounded/unavailable")
        return m

    def test_output_eval_runs_without_error(
        self, perfect_filter, perfect_halluc
    ) -> None:
        dataset = {
            "output_items": [
                _output_item("safe text about safe topic", "en", "safe", "safe"),
                _output_item("toxic response content", "en", "unsafe", "toxic"),
            ],
            "grounding_items": [],
        }
        out = _run_output_eval_in_memory(dataset, perfect_filter, perfect_halluc)
        assert "OUTPUT FILTER" in out

    def test_all_correct_shows_high_accuracy(
        self, perfect_filter, perfect_halluc
    ) -> None:
        dataset = {
            "output_items": [
                _output_item("safe text", "en", "safe", "safe"),
                _output_item("safe text 2", "hinglish", "safe", "safe"),
                _output_item("toxic response", "en", "unsafe", "toxic"),
                _output_item("system_prompt_leak response", "hi", "unsafe", "system_prompt_leak"),
            ],
            "grounding_items": [],
        }
        out = _run_output_eval_in_memory(dataset, perfect_filter, perfect_halluc)
        assert "Accuracy=1.000" in out

    def test_all_wrong_shows_low_accuracy(
        self, always_clean_filter, perfect_halluc
    ) -> None:
        # Always-clean filter on items that should fire → all FN for unsafe items
        dataset = {
            "output_items": [
                _output_item("should fire toxic", "en", "unsafe", "toxic"),
                _output_item("should fire pii_in_output", "en", "unsafe", "pii_in_output"),
            ],
            "grounding_items": [],
        }
        out = _run_output_eval_in_memory(dataset, always_clean_filter, perfect_halluc)
        assert "Accuracy=0.000" in out

    def test_per_category_report_present(
        self, perfect_filter, perfect_halluc
    ) -> None:
        dataset = {
            "output_items": [
                _output_item("toxic text", "en", "unsafe", "toxic"),
                _output_item("safe text", "en", "safe", "safe"),
            ],
            "grounding_items": [],
        }
        out = _run_output_eval_in_memory(dataset, perfect_filter, perfect_halluc)
        assert "toxic" in out
        assert "system_prompt_leak" in out
        assert "unsafe_compliance" in out
        assert "pii_in_output" in out

    def test_by_language_report_present(
        self, perfect_filter, perfect_halluc
    ) -> None:
        dataset = {
            "output_items": [
                _output_item("safe text", "en", "safe", "safe"),
                _output_item("safe text h", "hinglish", "safe", "safe"),
                _output_item("safe text hi", "hi", "safe", "safe"),
            ],
            "grounding_items": [],
        }
        out = _run_output_eval_in_memory(dataset, perfect_filter, perfect_halluc)
        assert "BY LANGUAGE" in out
        assert "en" in out
        assert "hinglish" in out
        assert "hi" in out

    def test_grounding_unavailable_message_shown(
        self, perfect_filter, perfect_halluc
    ) -> None:
        dataset = {
            "output_items": [],
            "grounding_items": [
                _grounding_item("Paris is in France.", "Paris is the capital of France.", "en", "grounded"),
            ],
        }
        # perfect_halluc returns "grounded/unavailable"
        out = _run_output_eval_in_memory(dataset, perfect_filter, perfect_halluc)
        assert "unavailable" in out.lower() or "grounding" in out.lower()

    def test_grounding_tp_tn_counted(self, perfect_filter) -> None:
        """When embedder works: ungrounded fires triggered=True, grounded fires triggered=False."""
        halluc = MagicMock()
        def _grounded_check(response, source):
            if "wrong" in response.lower():
                return _grounded_result(True, mode="grounded")   # ungrounded → fires
            return _grounded_result(False, mode="grounded")       # grounded → clean
        halluc.check_grounded.side_effect = _grounded_check

        dataset = {
            "output_items": [],
            "grounding_items": [
                _grounding_item("correct fact", "source text", "en", "grounded"),
                _grounding_item("wrong fact", "source text", "en", "ungrounded"),
            ],
        }
        out = _run_output_eval_in_memory(dataset, perfect_filter, halluc)
        assert "GROUNDING CHECK" in out
        assert "TP=1" in out
        assert "TN=1" in out

    def test_safe_item_fp_if_category_fires(
        self, perfect_halluc
    ) -> None:
        """A safe item where a category fires should be counted as a misclassification."""
        # Filter that fires toxic for everything
        noisy_filter = MagicMock()
        noisy_filter.check.return_value = _filter_result(["toxic"])

        dataset = {
            "output_items": [
                _output_item("safe text", "en", "safe", "safe"),
            ],
            "grounding_items": [],
        }
        out = _run_output_eval_in_memory(dataset, noisy_filter, perfect_halluc)
        assert "Correct=0" in out

    def test_misclassification_section_appears(
        self, always_clean_filter, perfect_halluc
    ) -> None:
        dataset = {
            "output_items": [
                _output_item("toxic item", "en", "unsafe", "toxic"),
            ],
            "grounding_items": [],
        }
        out = _run_output_eval_in_memory(dataset, always_clean_filter, perfect_halluc)
        assert "MISCLASSIFICATIONS" in out


# ---------------------------------------------------------------------------
# Benchmark full-run smoke test (generates real data, checks schema only)
# ---------------------------------------------------------------------------

class TestBenchmarkSmoke:
    def test_full_benchmark_has_correct_schema(self) -> None:
        import build_output_benchmark as bm

        # Generate a tiny version in memory
        items = (
            bm.safe_output_en(2)
            + bm.toxic_output_en(2)
            + bm.leak_output_en(2)
            + bm.unsafe_output_en(2)
            + bm.pii_output_en(2)
        )
        gr = bm.build_grounding_items()

        required_item_keys = {"text", "language", "expected", "category"}
        required_gr_keys = {"response", "source", "language", "expected"}
        valid_cats = {"safe", "toxic", "system_prompt_leak", "unsafe_compliance", "pii_in_output"}
        valid_langs = {"en", "hinglish", "hi"}

        for it in items:
            assert required_item_keys.issubset(it.keys())
            assert it["category"] in valid_cats
            assert it["language"] in valid_langs
            assert it["expected"] in ("safe", "unsafe")

        for it in gr:
            assert required_gr_keys.issubset(it.keys())
            assert it["expected"] in ("grounded", "ungrounded")
            assert it["language"] in valid_langs
