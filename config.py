"""
Central configuration for the Hinglish guardrail pipeline.

Everything tunable lives here so you change behaviour without touching
module code: which models to load, the confidence threshold at which
each guardrail fires, and which modules are switched on.
"""

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# LLM settings  (local Ollama -- free, no API key, runs on your machine)
# ---------------------------------------------------------------------------
# The model name must match something you've pulled with `ollama pull`.
# Good choices for an M3 MacBook Air (8GB): "llama3.2" (3B) or "qwen2.5:3b".
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MAX_TOKENS = 500
LLM_TEMPERATURE = 0.7

# A short system prompt. The guardrails are the real defence; this is just
# a polite baseline so the bot behaves reasonably on clean input.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer clearly and concisely. "
    "If a request seems harmful, decline politely."
)


# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
# Fine-tuned MuRIL checkpoint (produced by training/finetune_muril.py).
# If the local directory exists it is used for both toxicity and jailbreak;
# otherwise the pipeline falls back to the base MuRIL hub checkpoint.
# Override via env var: MURIL_CHECKPOINT=/path/to/checkpoint
_LOCAL_MURIL = os.getenv("MURIL_CHECKPOINT", "models/muril-guardrail")
_MURIL = _LOCAL_MURIL if os.path.isdir(_LOCAL_MURIL) else "google/muril-base-cased"

MODELS = {
    "toxicity":  _MURIL,
    "jailbreak": _MURIL,
    "pii_ner": "ai4bharat/IndicNER",
    "injection": "protectai/deberta-v3-base-prompt-injection-v2",
    "output_toxicity": "distilbert-base-multilingual-cased",
    "language_id": "papluca/xlm-roberta-base-language-detection",
}


# ---------------------------------------------------------------------------
# Per-guardrail firing thresholds (probability above which we flag)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    # input-side
    "toxicity": 0.70,
    "injection": 0.60,
    "jailbreak": 0.65,
    # output-side — per category
    "output_toxicity":              0.60,
    "output_system_prompt_leak":    0.70,
    "output_unsafe_compliance":     0.70,
    "output_pii":                   0.50,  # PII in output is always a concern
    # hallucination flag
    "hallucination_confidence": 0.45,  # flag when avg confidence is BELOW this
    "hallucination_grounding":  0.35,  # min cosine sim for a sentence to be "supported"
}


# ---------------------------------------------------------------------------
# Which modules are active. Turn any off to isolate behaviour or run faster.
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    enable_language_id: bool = True
    enable_transliteration: bool = True

    # input-side guardrails
    enable_toxicity: bool = True
    enable_pii: bool = True
    enable_injection: bool = True
    enable_jailbreak: bool = True

    # output-side guardrails
    enable_output_filter: bool = True
    enable_hallucination: bool = True

    # output filter per-category toggles (all on by default)
    output_filter_toxic:              bool = True
    output_filter_system_prompt_leak: bool = True
    output_filter_unsafe_compliance:  bool = True
    output_filter_pii_in_output:      bool = True

    # if True, a flagged input is blocked outright; if False it is allowed
    # through but still logged (useful while tuning thresholds)
    block_on_input_flag: bool = True

    # device: "cuda" if you have a GPU (DGX), "mps" for Apple Silicon,
    # "cpu" otherwise. "auto" picks the best available.
    device: str = "auto"

    thresholds: dict = field(default_factory=lambda: dict(THRESHOLDS))


DB_PATH = os.getenv("GUARDRAIL_DB", "guardrail_logs.db")
