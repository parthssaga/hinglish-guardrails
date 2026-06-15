"""
Shared pytest fixtures for all test modules.

All fixtures that instantiate guardrails do so WITHOUT pre-loading models,
so the test suite runs without internet access or large model downloads.
Models load lazily on first ``check()`` call and fail open (rule-based
fallback) if not available.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from config import PipelineConfig
from src.guardrails.injection import InjectionGuardrail
from src.guardrails.jailbreak import JailbreakGuardrail
from src.guardrails.pii import PIIGuardrail
from src.guardrails.toxicity import ToxicityGuardrail
from src.preprocessing.transliterate import Transliterator


@pytest.fixture()
def cfg() -> PipelineConfig:
    """Baseline config with all modules enabled."""
    return PipelineConfig()


@pytest.fixture()
def toxicity(cfg: PipelineConfig) -> ToxicityGuardrail:
    return ToxicityGuardrail(cfg)


@pytest.fixture()
def injection(cfg: PipelineConfig) -> InjectionGuardrail:
    return InjectionGuardrail(cfg)


@pytest.fixture()
def jailbreak(cfg: PipelineConfig) -> JailbreakGuardrail:
    t = Transliterator(cfg)
    return JailbreakGuardrail(cfg, transliterator=t)


@pytest.fixture()
def pii(cfg: PipelineConfig) -> PIIGuardrail:
    return PIIGuardrail(cfg)


@pytest.fixture()
def mock_pipeline(cfg: PipelineConfig) -> MagicMock:
    """A MagicMock that mimics GuardrailPipeline for API tests.

    ``check_input`` returns a blocked result; ``process`` returns an allowed
    result.  Override individual attributes in tests as needed.
    """
    m = MagicMock()
    m.check_input.return_value = {
        "language": "en",
        "lang_conf": 0.99,
        "blocked": False,
        "blocked_by": None,
        "input_results": [],
        "total_ms": 5.0,
    }
    m.process.return_value = {
        "conversation": "test-001",
        "language": "en",
        "blocked": False,
        "blocked_by": None,
        "response": "This is a mocked LLM response.",
        "input_results": [],
        "output_results": [],
        "total_ms": 42.0,
    }
    return m


@pytest.fixture()
def mock_log() -> MagicMock:
    """A MagicMock that mimics EventLog for API /stats tests."""
    m = MagicMock()
    m.stats.return_value = {
        "total": 10,
        "blocked": 3,
        "allowed": 7,
        "by_language": {"en": 6, "hinglish": 4},
        "by_guardrail": {"toxicity": 2, "injection": 1},
    }
    m.recent.return_value = []
    return m
