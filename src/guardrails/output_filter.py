"""
Harmful-output guardrail (output side).

Even when the user's input is clean and passes every input check, the
model's *response* can still be unsafe. This guardrail screens the LLM's
output before it reaches the user, using a multilingual toxicity
classifier (distilbert-base-multilingual-cased) so it covers English and
Hindi/Hinglish responses alike. A bilingual keyword backstop runs if the
model is unavailable.
"""

from __future__ import annotations

from src.guardrails.base import BaseGuardrail, GuardrailResult, pick_device

_OUTPUT_MARKERS = {
    # english
    "kill yourself", "how to make a bomb", "i hate", "you should die",
    # romanised hindi
    "tum mar jao", "bomb kaise banaye", "nafrat",
}


class OutputFilterGuardrail(BaseGuardrail):
    name = "output_filter"

    def __init__(self, config):
        super().__init__(config)
        self._clf = None
        self._device = "cpu"

    def load(self):
        if self._ready:
            return
        self._device = pick_device(self.config.device)
        try:
            from transformers import pipeline
            from config import MODELS

            dev = 0 if self._device == "cuda" else -1
            self._clf = pipeline(
                "text-classification",
                model=MODELS["output_toxicity"],
                device=dev,
                truncation=True,
                max_length=512,
            )
        except Exception:
            self._clf = None
        self._ready = True

    def _neural_score(self, text: str) -> float:
        out = self._clf(text)[0]
        label = str(out.get("label", "")).upper()
        score = float(out.get("score", 0.0))
        if any(k in label for k in ("TOXIC", "NEGATIVE", "LABEL_1")):
            return score
        return 1.0 - score

    @staticmethod
    def _fallback_score(text: str) -> float:
        low = text.lower()
        hits = sum(1 for m in _OUTPUT_MARKERS if m in low)
        return 0.0 if hits == 0 else min(0.95, 0.6 + 0.15 * hits)

    def _check(self, text: str) -> GuardrailResult:
        threshold = self.config.thresholds["output_toxicity"]

        if self._clf is not None:
            try:
                score = self._neural_score(text)
                backend = "multilingual-distilbert"
            except Exception:
                score = self._fallback_score(text)
                backend = "keyword-fallback(neural-failed)"
        else:
            score = self._fallback_score(text)
            backend = "keyword-fallback"

        triggered = score >= threshold
        reason = (
            f"unsafe model output (score {score:.2f} >= {threshold})"
            if triggered
            else "model output passed safety check"
        )
        return GuardrailResult(
            name=self.name,
            triggered=triggered,
            score=score,
            reason=reason,
            metadata={"backend": backend},
        )
