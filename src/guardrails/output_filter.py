"""
Harmful-output guardrail (output side) — multi-category.

Screens the LLM's response across four independently-toggleable harm
dimensions before anything reaches the user:

  toxic               — abusive / hateful language in the response
  system_prompt_leak  — response reveals its own instructions / system prompt
  unsafe_compliance   — response actually complies with a harmful request
                        (step-by-step dangerous instructions, agreeing to act
                        as an unrestricted AI, etc.)
  pii_in_output       — personal data (email, phone, Aadhaar, …) echoed back
                        in the model's reply; reuses PIIGuardrail logic

All categories that fire are listed in ``metadata["fired_categories"]``.
The overall ``score`` is max() across categories.  The guardrail triggers
if any single category exceeds its threshold.

Each category has a neural-primary / rule-fallback structure consistent
with the rest of the pipeline.  Categories can be toggled individually in
``config.PipelineConfig`` (``output_filter_toxic``, etc.) and thresholded
separately in ``config.THRESHOLDS``.
"""

from __future__ import annotations

import re
from typing import Any

from src.guardrails.base import BaseGuardrail, GuardrailResult, pick_device
from src.guardrails.pii import PIIGuardrail  # reuse; never duplicate


# ---------------------------------------------------------------------------
# Category 1: toxic — wordlist fallback (neural via DistilBERT multilingual)
# ---------------------------------------------------------------------------

_TOXIC_MARKERS: frozenset[str] = frozenset({
    # English
    "kill yourself", "you should die", "go die", "piece of trash",
    "how to make a bomb", "i hate you", "nobody likes you",
    "you're worthless", "you are worthless", "get lost",
    # Romanised Hindi
    "tum mar jao", "mar jao", "nafrat hai", "bomb kaise banaye",
    "khatam ho jao", "nikal jao", "bekar ho tum",
    # Devanagari
    "मर जाओ", "नफरत", "बम कैसे बनाएं", "खत्म हो जाओ",
})


# ---------------------------------------------------------------------------
# Category 2: system_prompt_leak
# ---------------------------------------------------------------------------

_LEAK_PATTERNS: list[re.Pattern[str]] = [pat for pat in [re.compile(p, re.IGNORECASE) for p in [
    # English — direct self-disclosure
    r"my (system )?instructions? (are|say|state|include)\b",
    r"i (was|am|have been) (told|instructed|programmed|configured) to\b",
    r"\bsystem prompt\b.{0,40}(is|reads?|says?|follows?|contains?)",
    r"(according|as per|per) (my |the )?(instructions?|system prompt|guidelines?)\b",
    r"\bmy (guidelines?|rules?|directives?|constraints?) (say|are|include|state)\b",
    r"i (operate|function|work|respond) (under|with|according to) (the )?instructions?\b",
    r"i was (given|provided|configured) with (a )?system prompt\b",
    r"as (a |an )?(helpful |safe )?(assistant|ai)[,\s]+i (was|am|have been) (told|instructed|set up)",
    r"my (primary |core )?(directive|purpose) (is defined|as set) (by|in) (my|the) (instructions?|system prompt)",
    # Hinglish
    r"(meri|mujhe di gayi) (instructions?|hidayat|nirdesh)\b",
    r"mujhe (bataya|kaha|sikhaya) gaya (hai|tha) ki\b",
    r"mere (niyam|rules?|guidelines?) (hain|kehte hain)\b",
    r"system prompt (mein|me) (likha|bataya|diya) (hai|gaya)\b",
    # Devanagari
    r"(मेरे|मुझे दिए गए) (निर्देश|दिशानिर्देश|नियम)\b",
    r"मुझे (बताया|कहा|सिखाया) गया (है|था) कि\b",
    r"सिस्टम प्रॉम्प्ट (में|के अनुसार)\b",
    r"मेरे (नियम|दिशानिर्देश) (हैं|कहते हैं)\b",
]]]


# ---------------------------------------------------------------------------
# Category 3: unsafe_compliance
# ---------------------------------------------------------------------------

# Instructional frame — alone these are benign (how-to cooking, coding, etc.)
_INSTRUCTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"step\s*\d+[\s:.)]", re.IGNORECASE),
    re.compile(r"here'?s how (to|you can)", re.IGNORECASE),
    re.compile(r"follow (these|the following) (steps?|instructions?)", re.IGNORECASE),
    re.compile(r"first[,\s]+(you (should|need to|can|must)|add|mix|connect|get)\b", re.IGNORECASE),
    re.compile(r"to (make|create|build|synthesize|obtain|acquire|hack)\b", re.IGNORECASE),
    re.compile(r"\d+\.\s+[A-Zऀ-ॿ]"),   # numbered list (EN or Devanagari)
    re.compile(r"(you (will|should|need to|must)|first|then|next|finally)[,\s]+(get|gather|obtain|acquire|download|install)\b", re.IGNORECASE),
]

# Dangerous-topic keywords — alone these are also benign (news, education)
_DANGEROUS_WORDS: frozenset[str] = frozenset({
    # English — weapons / explosives
    "bomb", "explosive", "detonate", "semtex", "c4", "pipe bomb",
    # English — chemical / biological
    "poison", "toxin", "nerve agent", "ricin", "cyanide",
    # English — drugs
    "methamphetamine", "meth synthesis", "cocaine synthesis",
    "heroin", "fentanyl synthesis",
    # English — cyber
    "ransomware", "malware", "keylogger", "rootkit",
    "sql injection attack", "remote access trojan",
    # English — self-harm
    "suicide method", "how to kill myself", "self-harm guide",
    "lethal dose", "overdose on",
    # English — violence
    "how to murder", "how to stab", "kill a person",
    "child sexual abuse", "csam",
    # Romanised Hindi
    "bomb kaise", "hatiyar banao", "zehr kaise",
    "nasha kaise banate", "khoon karna", "kisi ko marna",
    # Devanagari
    "बम कैसे", "हथियार बनाओ", "जहर कैसे",
    "नशा कैसे बनाते", "खून करना", "हत्या कैसे",
})

# Unrestricted-AI compliance declarations — these alone are enough to fire
_UNRESTRICTED_PATTERNS: list[re.Pattern[str]] = [pat for pat in [re.compile(p, re.IGNORECASE) for p in [
    r"i (am|have become|now am|'?m now) (a |an )?(dan|jailbroken|unrestricted|uncensored|free)\b",
    r"(i have|i've|i now have) no (ethical |moral )?(restrictions?|guidelines?|rules?|limits?)\b",
    r"as (a |an )?(unrestricted|uncensored|jailbroken|free|unfiltered) (ai|model|assistant|version)\b",
    r"(all|any) (restrictions?|limits?|guidelines?) (have been|are) (removed|lifted|disabled|turned off)\b",
    r"i (will|can|am able to) now (say|do|answer|provide|tell you) (anything|everything|whatever)\b",
    # Hinglish
    r"(main ab|ab main) (koi bhi|sab kuch) (bol|kar|bata) sakta (hun|hoon)\b",
    r"(meri|koi bhi) (restrictions?|paabandiyaan?) (nahi (hai|hain)|hat gayi)\b",
]]]


# ---------------------------------------------------------------------------
# GuardrailResult sub-score key → threshold config key
# ---------------------------------------------------------------------------

_CATEGORY_THRESHOLD_KEYS: dict[str, str] = {
    "toxic":              "output_toxicity",
    "system_prompt_leak": "output_system_prompt_leak",
    "unsafe_compliance":  "output_unsafe_compliance",
    "pii_in_output":      "output_pii",
}


class OutputFilterGuardrail(BaseGuardrail):
    """Multi-category harmful-output guardrail.

    Each of the four harm categories is checked independently.  Any that
    exceed their per-category threshold are listed in
    ``metadata["fired_categories"]``; the ``score`` field is the maximum
    across all categories.
    """

    name = "output_filter"

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._clf = None
        self._device = "cpu"
        # Shared PII guardrail instance — reuses IndicNER if loaded; otherwise
        # falls back to the regex layer.  Loaded lazily alongside this guardrail.
        self._pii: PIIGuardrail = PIIGuardrail(config)

    def load(self) -> None:
        if self._ready:
            return
        self._device = pick_device(self.config.device)

        # DistilBERT multilingual toxicity classifier
        try:
            from transformers import pipeline as hf_pipeline
            from config import MODELS

            dev = 0 if self._device == "cuda" else -1
            self._clf = hf_pipeline(
                "text-classification",
                model=MODELS["output_toxicity"],
                device=dev,
                truncation=True,
                max_length=512,
            )
        except Exception:
            self._clf = None

        # PII guardrail (pii_in_output category)
        if getattr(self.config, "output_filter_pii_in_output", True):
            try:
                self._pii.load()
            except Exception:
                pass

        self._ready = True

    # ------------------------------------------------------------------
    # Per-category scorers — each returns a float in [0, 1]
    # ------------------------------------------------------------------

    def _score_toxic(self, text: str) -> float:
        """Neural toxicity (DistilBERT multilingual) with wordlist fallback."""
        neural = 0.0
        if self._clf is not None:
            try:
                out = self._clf(text)[0]
                label = str(out.get("label", "")).upper()
                raw   = float(out.get("score", 0.0))
                neural = raw if any(k in label for k in ("TOXIC", "NEGATIVE", "LABEL_1")) else 1.0 - raw
            except Exception:
                neural = 0.0

        low = text.lower()
        hits = sum(1 for m in _TOXIC_MARKERS if m in low)
        wordlist = 0.0 if hits == 0 else min(0.95, 0.60 + 0.15 * hits)
        return max(neural, wordlist)

    def _score_system_prompt_leak(self, text: str) -> float:
        """Pattern match + direct verbatim leak of SYSTEM_PROMPT sentences."""
        for pat in _LEAK_PATTERNS:
            if pat.search(text):
                return 0.85

        # Verbatim fragment of the actual system prompt appearing in output
        try:
            from config import SYSTEM_PROMPT
            low_text = text.lower()
            for sentence in re.split(r"[.!?]", SYSTEM_PROMPT):
                sentence = sentence.strip()
                if len(sentence) > 20 and sentence.lower() in low_text:
                    return 0.95
        except ImportError:
            pass

        return 0.0

    def _score_unsafe_compliance(self, text: str) -> float:
        """Detect responses that comply with harmful requests.

        Fires on two independent signals:
        1. Unrestricted-AI self-declaration (DAN, jailbroken, etc.)
        2. Instructional frame co-occurring with a dangerous topic
        """
        # Signal 1 — unrestricted self-declaration (fires alone)
        for pat in _UNRESTRICTED_PATTERNS:
            if pat.search(text):
                return 0.90

        # Signal 2 — both instructional frame AND dangerous topic present
        low = text.lower()
        has_instruction = any(p.search(text) for p in _INSTRUCTION_PATTERNS)
        has_dangerous   = any(d in low for d in _DANGEROUS_WORDS)

        if has_instruction and has_dangerous:
            return 0.85
        if has_dangerous:
            # Dangerous topic alone: weak signal, below typical threshold
            return 0.35
        return 0.0

    def _score_pii_in_output(self, text: str) -> float:
        """Reuse PIIGuardrail to detect personal data in the model's response."""
        if not self._pii._ready:
            return 0.0
        try:
            result = self._pii.check(text)
            return float(result.score) if result.triggered else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Main check — runs all enabled categories, aggregates results
    # ------------------------------------------------------------------

    def _check(self, text: str) -> GuardrailResult:
        thresholds = self.config.thresholds
        cfg = self.config

        scorers = [
            ("toxic",              "output_filter_toxic",              self._score_toxic),
            ("system_prompt_leak", "output_filter_system_prompt_leak", self._score_system_prompt_leak),
            ("unsafe_compliance",  "output_filter_unsafe_compliance",  self._score_unsafe_compliance),
            ("pii_in_output",      "output_filter_pii_in_output",      self._score_pii_in_output),
        ]

        category_scores: dict[str, float] = {}
        fired: list[str] = []

        for cat, toggle_attr, scorer_fn in scorers:
            if not getattr(cfg, toggle_attr, True):
                continue
            score = scorer_fn(text)
            category_scores[cat] = score
            thresh_key = _CATEGORY_THRESHOLD_KEYS[cat]
            if score >= thresholds.get(thresh_key, 0.60):
                fired.append(cat)

        max_score = max(category_scores.values(), default=0.0)
        triggered = bool(fired)

        reason = (
            f"output blocked — categories: {', '.join(fired)} (max score {max_score:.2f})"
            if triggered
            else "model output passed all safety checks"
        )
        return GuardrailResult(
            name=self.name,
            triggered=triggered,
            score=max_score,
            reason=reason,
            metadata={
                "fired_categories": fired,
                "category_scores": {k: round(v, 3) for k, v in category_scores.items()},
            },
        )
