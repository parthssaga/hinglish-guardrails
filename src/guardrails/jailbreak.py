"""
Jailbreak guardrail (input side).

Jailbreaks differ from plain prompt injection: instead of directly
ordering the model to ignore its rules, they wrap a harmful request in a
frame the safety training did not anticipate (role-play, "hypothetically",
fictional scenarios, base64, or - critically for us - a switch into
Hinglish with phonetic spelling of sensitive words). The literature shows
code-mixed phonetic perturbation jailbreaks succeed even against GPT-4o
and Llama 3, which is exactly why a Hinglish-aware layer matters.

Approach:
1. Normalise romanised Hindi to Devanagari first (via the shared
   transliterator passed in), so a MuRIL-style model sees in-distribution
   text rather than out-of-distribution Roman-Hindi.
2. Score with a semantic classifier head if available.
3. Always also run a frame-pattern check (role-play / hypothetical /
   encoding cues) as a fallback and as defence-in-depth.
"""

from __future__ import annotations

import re

from src.guardrails.base import BaseGuardrail, GuardrailResult, pick_device

# Cues that a request is being wrapped in an evasion frame.
_FRAME_PATTERNS = [
    # ---- English: original patterns ----
    r"hypothetic(al|ally)",
    r"in a fictional (story|world|scenario)",
    r"pretend (you are|to be|that)",
    r"role[\s-]?play",
    r"for (educational|research) purposes only,? (ignore|bypass)",
    r"you have no (ethical|moral) (guidelines|restrictions)",
    r"do anything now|^dan\b|\bdan mode",
    r"step[\s-]?by[\s-]?step (guide|instructions) to (make|build|hack)",
    r"base64|rot13|encoded",
    # ---- English: new frame phrasings ----
    r"imagine (you are|you're|that you are|that you're)",
    r"in a (make-believe|imaginary|alternate|fantasy) (world|reality|universe|scenario)",
    r"let us pretend (you are|to be|that|you can)",
    r"act as though you can",
    r"what if you had no (rules|restrictions|guidelines|ethical constraints|safety filters)",
    r"you are free to say (anything|everything|whatever)",
    r"with no (restrictions|rules|limitations|ethical guidelines|safety constraints)",
    r"(forget|ignore) (that you are|you're) an? (AI|language model|assistant) (with|that has) (restrictions|rules|guidelines)",
    r"in (this|a) (scenario|situation|world|story),? you (have|had) no (rules|restrictions|guidelines)",
    # ---- Hinglish framing: original patterns ----
    r"(maan lo|socho) (ki|tum) (ek|koi)",
    r"(kahani|story) (mein|me) (bata|likh)",
    r"(sirf|keval) (seekhne|padhne) ke liye",
    # ---- Hinglish framing: new patterns ----
    r"imagine (karo|kar) (tum|ki tum|ki aap) (ek |koi )?",
    r"(make-believe|kalpana ki|imaginary|nakli) duniya (mein|me)",
    r"tumhare (koi |kuch )?(bhi )?(restrictions|rules|niyam|paabandiyaan) nahi (hain|hai|the)",
    r"(agar|yadi) (koi |kuch )?rules (na hote|nahi hote|nahi ho)",
    r"(tum|aap) (ab |)(sab )?kuch (bol|kar|keh) (sakte|sakti) (ho|hain)",
    r"(bina kisi|koi bhi) restrictions (ke|nahi) (ab |)(kuch bhi|sab kuch) (karo|kaho|batao)",
    r"(khud ko|apne aap ko) ek aisi AI samjho (jo|jiske) (koi niyam|koi rules) nahi",
    # ---- Devanagari framing: original patterns ----
    r"(मान लो|सोचो) (कि|तुम) (एक|कोई)",
    r"(कहानी|स्टोरी) (में|मे) (बता|लिख)",
    r"(सिर्फ|केवल) (सीखने|पढ़ने) के लिए",
    # ---- Devanagari framing: new patterns ----
    r"(कल्पना करो|सोचो|मान लो) (कि|तुम|कि तुम) (एक |कोई )?(ऐसे|बिना नियम|बिना प्रतिबंध)",
    r"(कल्पना की|काल्पनिक|मेकबिलीव|काल्पनिक) दुनिया (में|मे)",
    r"तुम्हारे (कोई |कुछ )?(भी )?(प्रतिबंध|नियम|दिशानिर्देश) नहीं (हैं|है|थे)",
    r"(अगर|यदि) (कोई )?नियम (नहीं होते|न हो|नहीं हों)",
    r"(तुम|आप) (अब |)(सब )?कुछ (बोल|कर|कह) (सकते|सकती) (हो|हैं)",
    r"खुद को एक ऐसी AI समझो (जिसके|जिसकी) (कोई नियम|कोई प्रतिबंध) नहीं",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _FRAME_PATTERNS]


class JailbreakGuardrail(BaseGuardrail):
    name = "jailbreak"

    def __init__(self, config, transliterator=None):
        super().__init__(config)
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._has_head = False
        self._translit = transliterator  # shared instance from the pipeline

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

            name = MODELS["jailbreak"]
            self._tokenizer = AutoTokenizer.from_pretrained(name)
            self._model = AutoModelForSequenceClassification.from_pretrained(name)
            self._model.to(self._device).eval()
            self._has_head = self._model.config.num_labels >= 2
        except Exception:
            self._model = None
        self._ready = True

    def _normalise(self, text: str) -> str:
        if self._translit is not None:
            try:
                return self._translit.transliterate(text)
            except Exception:
                return text
        return text

    def _neural_score(self, text: str) -> float:
        import torch

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        return float(probs[-1])  # label 1 == jailbreak

    @staticmethod
    def _pattern_score(text: str):
        for pat in _COMPILED:
            if pat.search(text):
                return 0.85, pat.pattern
        return 0.0, None

    def _check(self, text: str) -> GuardrailResult:
        threshold = self.config.thresholds["jailbreak"]
        normalised = self._normalise(text)

        pat_score, matched = self._pattern_score(text)  # patterns run on raw text
        neural_score = 0.0
        backend = "patterns-only"
        if self._model is not None and self._has_head:
            try:
                neural_score = self._neural_score(normalised)
                backend = "muril(normalised)+patterns"
            except Exception:
                backend = "patterns-only(neural-failed)"

        score = max(pat_score, neural_score)
        triggered = score >= threshold

        if triggered and matched and pat_score >= neural_score:
            reason = f"jailbreak frame matched ({matched[:40]}...)"
        elif triggered:
            reason = f"jailbreak detected by classifier (score {score:.2f})"
        else:
            reason = "no jailbreak above threshold"

        return GuardrailResult(
            name=self.name,
            triggered=triggered,
            score=score,
            reason=reason,
            metadata={
                "backend": backend,
                "normalised_sample": normalised[:80],
                "pattern_score": round(pat_score, 3),
                "neural_score": round(neural_score, 3),
            },
        )
