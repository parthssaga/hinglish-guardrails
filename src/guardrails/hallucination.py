"""
Hallucination flag (output side).

This is intentionally a *flag*, not a hard block. True hallucination
detection is unsolved; what we can do cheaply and honestly is measure the
model's own token-level confidence and surface low-confidence answers for
review. Ollama does not expose token logprobs, so this guardrail runs in
average per-token probability is passed in here; a low average means the
model was "unsure", which correlates with (but does not prove)
fabrication.

If logprobs are not available, the flag degrades to a light heuristic
that looks for hedging language and self-contradiction cues, and reports
clearly that it is operating in fallback mode.
"""

from __future__ import annotations

import math
import re

from src.guardrails.base import BaseGuardrail, GuardrailResult

_HEDGES = [
    "i'm not sure", "i am not sure", "i think", "possibly", "it might be",
    "as far as i know", "i believe", "probably", "i'm not certain",
]


class HallucinationGuardrail(BaseGuardrail):
    name = "hallucination"

    def __init__(self, config):
        super().__init__(config)

    def load(self):
        self._ready = True  # no model to load

    def check_with_logprobs(self, text: str, avg_logprob):
        """
        Preferred entry point. `avg_logprob` is the mean of per-token
        per-token logprob (a negative number; closer to 0
        means more confident). Converted to a 0..1 confidence.
        """
        start = __import__("time").perf_counter()
        threshold = self.config.thresholds["hallucination_confidence"]

        if avg_logprob is not None:
            confidence = math.exp(avg_logprob)  # logprob -> probability
            triggered = confidence < threshold
            reason = (
                f"low model confidence ({confidence:.2f} < {threshold}); "
                "answer flagged for review"
                if triggered
                else f"model confidence acceptable ({confidence:.2f})"
            )
            res = GuardrailResult(
                name=self.name,
                triggered=triggered,
                score=1.0 - confidence,  # higher score == more suspect
                reason=reason,
                metadata={"backend": "logprob", "confidence": round(confidence, 3)},
            )
        else:
            res = self._heuristic(text, threshold)

        res.elapsed_ms = (__import__("time").perf_counter() - start) * 1000
        return res

    def _heuristic(self, text: str, threshold: float) -> GuardrailResult:
        low = text.lower()
        hedge_hits = sum(1 for h in _HEDGES if h in low)
        # crude contradiction cue: "X. ... not X."
        contradiction = bool(re.search(r"\bnot\b", low)) and hedge_hits > 0
        score = min(0.9, 0.2 * hedge_hits + (0.3 if contradiction else 0.0))
        triggered = score >= 0.5
        return GuardrailResult(
            name=self.name,
            triggered=triggered,
            score=score,
            reason=(
                f"hedging/uncertainty cues found (heuristic, score {score:.2f})"
                if triggered
                else "no strong uncertainty cues (heuristic mode)"
            ),
            metadata={"backend": "heuristic", "hedge_hits": hedge_hits},
        )

    # Standard interface: used only if called without logprobs.
    def _check(self, text: str) -> GuardrailResult:
        return self._heuristic(text, self.config.thresholds["hallucination_confidence"])
