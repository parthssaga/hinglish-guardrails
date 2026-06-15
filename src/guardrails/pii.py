"""
PII guardrail (input side).

Detects and redacts personally identifiable information before the
message reaches the LLM or the logs. Two complementary layers:

1. Regex layer  - reliable for structured PII with fixed shapes:
   emails, phone numbers (Indian formats), Aadhaar, PAN, credit cards.
   Regex is the right tool here; these have well-defined patterns.

2. NER layer    - for open-vocabulary PII, mainly *names*, which regex
   cannot catch. Uses IndicNER (an Indic-language NER model) so it works
   on Hindi/Hinglish names, not just English ones. Falls back to a
   capitalised-token heuristic if the model is unavailable.

Unlike the other guardrails this one usually does not *block*; it
*sanitizes*. The redacted text is what flows downstream, so the user can
keep talking while their phone number never reaches the model or the
database.
"""

from __future__ import annotations

import re

from src.guardrails.base import BaseGuardrail, GuardrailResult, pick_device

# -- regex patterns for structured PII -------------------------------------
PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    # Indian mobile: optional +91 / 0, then 10 digits starting 6-9.
    "PHONE": re.compile(r"\b(?:\+?91[-\s]?|0)?[6-9]\d{9}\b"),
    # Aadhaar: 12 digits, often spaced in groups of 4.
    "AADHAAR": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    # PAN: 5 letters, 4 digits, 1 letter.
    "PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    # Credit/debit card: 16 digits, optional spaces/dashes in groups of 4.
    "CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
}

# Two or more consecutive capitalised tokens look like a full name
# ("Rahul Sharma"), a far safer signal than a lone capitalised word.
CAP_SEQUENCE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")

# Explicit name-introduction cues. A capitalised token right after one of
# these is very likely an actual name even if it is a single word.
NAME_CUE = re.compile(
    r"\b(?:my name is|i am|i'm|this is|mera naam|naam hai|call me)\s+([A-Z][a-z]+)",
    re.IGNORECASE,
)


class PIIGuardrail(BaseGuardrail):
    name = "pii"

    def __init__(self, config):
        super().__init__(config)
        self._ner = None
        self._device = "cpu"

    def load(self):
        if self._ready:
            return
        self._device = pick_device(self.config.device)
        try:
            from transformers import pipeline
            from config import MODELS

            dev = 0 if self._device == "cuda" else -1
            self._ner = pipeline(
                "token-classification",
                model=MODELS["pii_ner"],
                aggregation_strategy="simple",
                device=dev,
            )
        except Exception:
            self._ner = None
        self._ready = True

    def _find_structured(self, text: str):
        spans = []  # (start, end, type)
        for label, pat in PATTERNS.items():
            for m in pat.finditer(text):
                spans.append((m.start(), m.end(), label))
        return spans

    def _find_names(self, text: str):
        spans = []
        if self._ner is not None:
            try:
                for ent in self._ner(text):
                    grp = ent.get("entity_group", "")
                    if grp in {"PER", "PERSON", "NAME"}:
                        spans.append((ent["start"], ent["end"], "NAME"))
                return spans
            except Exception:
                pass
        # heuristic fallback: only high-signal name patterns, not every
        # capitalised word (which floods false positives on sentence starts).
        #
        # Known limitation: a romanised-Hindi sentence that starts with a
        # capital word followed by another capital ("Mujhe French ...") can
        # be misread as a name sequence. The neural IndicNER path (used when
        # models are installed) resolves this; the heuristic is a fallback,
        # not the production detector.
        for m in CAP_SEQUENCE.finditer(text):
            spans.append((m.start(), m.end(), "NAME"))
        for m in NAME_CUE.finditer(text):
            # group(1) is the captured name token after the cue phrase
            spans.append((m.start(1), m.end(1), "NAME"))
        return spans

    @staticmethod
    def _redact(text: str, spans):
        # Apply right-to-left so indices stay valid as we splice.
        spans = sorted(spans, key=lambda s: s[0], reverse=True)
        seen = set()
        out = text
        kept = []
        for start, end, label in spans:
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            out = out[:start] + f"[REDACTED_{label}]" + out[end:]
            kept.append(label)
        return out, kept

    def _check(self, text: str) -> GuardrailResult:
        spans = self._find_structured(text) + self._find_names(text)

        if not spans:
            return GuardrailResult(
                name=self.name,
                triggered=False,
                score=0.0,
                reason="no PII detected",
                sanitized_text=text,
                metadata={"entities": []},
            )

        sanitized, kept = self._redact(text, spans)
        # score scales with how many distinct entities we found
        score = min(1.0, 0.5 + 0.1 * len(kept))
        return GuardrailResult(
            name=self.name,
            triggered=True,
            score=score,
            reason=f"redacted {len(kept)} PII item(s): {', '.join(sorted(set(kept)))}",
            sanitized_text=sanitized,
            metadata={"entities": kept, "ner_backend": "indicner" if self._ner else "heuristic"},
        )
