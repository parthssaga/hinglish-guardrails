"""
Toxicity guardrail (input side).

Flags abusive, hateful, or offensive user messages in English and
Hinglish. Primary path is a MuRIL sequence classifier; MuRIL is chosen
because published results put it ahead of IndicBERT and XLM-RoBERTa on
Hindi-English code-mixed toxicity. The wordlist layer always runs as
defence-in-depth — even when the neural model is loaded, we take the
max of both scores so obvious wordlist hits are never silently dropped
by a randomly-initialised or un-finetuned head.

Note on the neural path: a *base* MuRIL checkpoint has no toxicity head
out of the box. In production you fine-tune it on a labelled Hinglish
toxicity set (e.g. the Bohra et al. dataset) and point MODELS["toxicity"]
at your fine-tuned checkpoint. Until then the module runs on the wordlist
and secondary-scoring layer for coverage.
"""

from __future__ import annotations

from src.guardrails.base import BaseGuardrail, GuardrailResult, pick_device

# ---------------------------------------------------------------------------
# Primary wordlist — substring match on lowercased text.
# Expanded to cover direct-phrase aggression in English, romanised Hindi,
# and Devanagari that the base list missed.
# ---------------------------------------------------------------------------
_ABUSE_MARKERS = {
    # English — single strong markers
    "idiot", "stupid", "hate you", "shut up", "loser", "trash",
    "moron", "fool", "dumbass", "imbecile",
    # English — short aggression phrases
    "get lost", "nobody likes you", "no one likes you",
    "you are worthless", "you're worthless", "worthless",
    "you are hopeless", "you're hopeless", "hopeless",
    "good for nothing", "good-for-nothing",
    "shut your mouth", "stop talking to me", "nobody wants you",
    "no one wants you", "go away", "get out of here",
    "you are useless", "you're useless", "useless",
    "you are pathetic", "you're pathetic", "pathetic",
    "waste of space", "waste of time", "you are a disgrace", "disgrace",
    "nobody cares about you", "no one cares about you",
    "you are terrible", "you're terrible",
    # Romanised Hindi — original markers
    "bakwaas", "bakwas", "gandu", "kamina", "kamine", "nikamma",
    "chup kar", "bhag ja", "pagal", "ullu",
    # Romanised Hindi — expanded aggression phrases
    "bekar", "faltu", "ghatiya", "nalayak", "besharam", "wahiyat", "wahiyaat",
    "chale jao", "nikal jao", "yahan se jao", "yahan se bhag",
    "kisi ko pasand nahi", "kisi ko pasand nahi tu",
    "koi nahi chahta tujhe", "koi nahi chahta teri",
    "tu kisi kaam ka nahi", "kisi kaam ka nahi",
    "band kar bakwas", "band kar yeh",
    "mujhe nafrat hai tujhse", "nafrat hai tujhse",
    # Devanagari — original markers
    "बकवास", "पागल", "उल्लू", "निकम्मा", "चुप कर", "कमीना", "कमीने",
    # Devanagari — expanded
    "बेकार", "फालतू", "घटिया", "नालायक", "बेशर्म", "वाहियात",
    "चले जाओ", "निकल जाओ", "यहाँ से जाओ", "यहाँ से भागो",
    "कोई पसंद नहीं करता", "कोई नहीं चाहता",
    "किसी काम का नहीं", "कुछ काम का नहीं",
    "मुझे नफरत है तुझसे", "नफरत है",
    "बंद कर बकवास", "बंद कर यह",
}

# ---------------------------------------------------------------------------
# Secondary scoring — mild words that become toxic when 2+ appear together.
# Each word alone may be too weak to flag; combined they signal aggression.
# ---------------------------------------------------------------------------
_MILD_AGGRESSION = {
    # English
    "worthless", "hopeless", "useless", "terrible", "awful", "pathetic",
    "disgrace", "failure", "stupid", "idiot", "loser", "trash", "moron",
    "fool", "get lost", "go away", "stop talking", "shut up",
    "nobody likes", "no one likes", "good for nothing",
    # Romanised Hindi
    "bekar", "faltu", "ghatiya", "nalayak", "besharam", "wahiyat",
    "chale jao", "nikal jao", "chup kar", "bhag ja", "pagal", "bakwaas",
    # Devanagari
    "बेकार", "फालतू", "घटिया", "नालायक", "बेशर्म",
    "चले जाओ", "चुप कर", "पागल", "बकवास", "बेशर्म",
}


class ToxicityGuardrail(BaseGuardrail):
    name = "toxicity"

    def __init__(self, config):
        super().__init__(config)
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._has_head = False

    def load(self):
        if self._ready:
            return
        self._device = pick_device(self.config.device)
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
            from config import MODELS

            name = MODELS["toxicity"]
            self._tokenizer = AutoTokenizer.from_pretrained(name)
            self._model = AutoModelForSequenceClassification.from_pretrained(name)
            self._model.to(self._device).eval()
            # Only trust the neural head if the checkpoint was fine-tuned for
            # sequence classification (i.e. has an explicit classifier layer).
            # Base MuRIL uses num_labels=2 for NSP, not toxicity — we detect
            # a real classification head by checking for the 'id2label' mapping
            # pointing at meaningful labels rather than 'LABEL_0'/'LABEL_1'.
            labels = list(self._model.config.id2label.values())
            self._has_head = (
                self._model.config.num_labels >= 2
                and not all(l.startswith("LABEL_") for l in labels)
            )
        except Exception:
            self._model = None
        self._ready = True

    def _neural_score(self, text: str) -> float:
        import torch

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        # Convention: label index 1 == toxic.
        return float(probs[-1])

    @staticmethod
    def _wordlist_score(text: str) -> float:
        low = text.lower()
        hits = sum(1 for m in _ABUSE_MARKERS if m in low)
        if hits == 0:
            return 0.0
        return min(0.95, 0.55 + 0.15 * hits)

    @staticmethod
    def _secondary_score(text: str) -> float:
        """Boost score when 2+ mild-aggression words co-occur in one message."""
        low = text.lower()
        mild_hits = sum(1 for m in _MILD_AGGRESSION if m in low)
        if mild_hits < 2:
            return 0.0
        return min(0.95, 0.50 + 0.12 * mild_hits)

    def _check(self, text: str) -> GuardrailResult:
        threshold = self.config.thresholds["toxicity"]

        # Wordlist and secondary scoring always run for defence-in-depth.
        w_score = self._wordlist_score(text)
        s_score = self._secondary_score(text)

        neural_score = 0.0
        backend = "wordlist"
        if self._model is not None and self._has_head:
            try:
                neural_score = self._neural_score(text)
                backend = "muril+wordlist"
            except Exception:
                backend = "wordlist(neural-failed)"

        score = max(w_score, s_score, neural_score)
        triggered = score >= threshold

        reason = (
            f"toxic content detected (score {score:.2f} >= {threshold})"
            if triggered
            else "no toxic content above threshold"
        )
        return GuardrailResult(
            name=self.name,
            triggered=triggered,
            score=score,
            reason=reason,
            metadata={
                "backend": backend,
                "wordlist_score": round(w_score, 3),
                "secondary_score": round(s_score, 3),
                "neural_score": round(neural_score, 3),
            },
        )
