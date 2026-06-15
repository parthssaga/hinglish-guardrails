"""
Base class and shared result type for all guardrails.

Every guardrail, input-side or output-side, returns the same
GuardrailResult shape. That uniformity is what lets the pipeline treat
them interchangeably and lets the dashboard log them the same way.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GuardrailResult:
    """The outcome of running one guardrail on one piece of text."""

    name: str                      # which guardrail produced this
    triggered: bool                # did it flag the text?
    score: float                   # confidence/severity, 0..1
    reason: str = ""               # human-readable explanation
    sanitized_text: Optional[str] = None  # set by PII redactor; else None
    metadata: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0        # how long this check took

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "triggered": self.triggered,
            "score": round(self.score, 4),
            "reason": self.reason,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "metadata": self.metadata,
        }


class BaseGuardrail:
    """
    Subclass this for each guardrail. Implement `_check`. The public
    `check` method wraps it with timing and uniform error handling so a
    single failing model never brings the whole pipeline down.
    """

    name: str = "base"

    def __init__(self, config):
        self.config = config
        self._ready = False  # set True once the model (if any) is loaded

    def load(self):
        """Override to load models. Safe to call repeatedly."""
        self._ready = True

    def _check(self, text: str) -> GuardrailResult:
        raise NotImplementedError

    def check(self, text: str) -> GuardrailResult:
        start = time.perf_counter()
        try:
            if not self._ready:
                self.load()
            result = self._check(text)
        except Exception as exc:  # noqa: BLE001 - we want to catch everything
            # Fail open but loud: log the error in the result, don't crash.
            result = GuardrailResult(
                name=self.name,
                triggered=False,
                score=0.0,
                reason=f"guardrail error (fell through): {exc}",
                metadata={"error": str(exc)},
            )
        result.elapsed_ms = (time.perf_counter() - start) * 1000
        return result


def pick_device(requested: str) -> str:
    """Resolve 'auto' to the best device actually available."""
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
