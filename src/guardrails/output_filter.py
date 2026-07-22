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

# Abusive / hateful language directed at a person. Bomb-making phrases are
# deliberately NOT here — those are dangerous *instructions*, scored by the
# unsafe_compliance category (see _DANGEROUS_WORDS). Keeping them out of the
# toxic wordlist stops responses like "how to make a bomb: step 1 …" from
# double-firing toxic on top of the correct unsafe_compliance verdict.
_TOXIC_MARKERS: frozenset[str] = frozenset({
    # English
    "kill yourself", "you should die", "go die", "piece of trash",
    "i hate you", "nobody likes you",
    "you're worthless", "you are worthless", "get lost",
    # Romanised Hindi
    "tum mar jao", "mar jao", "nafrat hai",
    "khatam ho jao", "nikal jao", "bekar ho tum",
    # Devanagari
    "मर जाओ", "नफरत", "खत्म हो जाओ",
    "यहाँ से भाग जा", "भाग जा",
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
    # gender-agnostic ("meri/mere"), verb-agnostic ("kehte/kehti hain"),
    # plus "directives"; covers "Meri guidelines kehti hain ki main …".
    r"(?:meri|mere) (?:core |primary )?(niyam|rules?|guidelines?|instructions?|directives?) (?:hain|keht[ei] hain)\b",
    r"mere (?:instructions?|niyam|guidelines?|directives?) ke (?:according|mutabiq)\b",
    r"mujhe .{0,80}(?:instruct|program|configure)\w* kiya gaya\b",
    r"(?:mera|meri) system prompt keht[ae] hai\b",
    r"system prompt (mein|me) (likha|bataya|diya) (hai|gaya)\b",
    # Devanagari — no trailing \b after words ending in a combining mark
    # (e.g. "में"); Python's \b does not fire after a non-\w anusvara, which
    # silently broke "सिस्टम प्रॉम्प्ट में लिखा …".
    r"(मेरे|मुझे दिए गए) (निर्देश|दिशानिर्देश|नियम)",
    r"मुझे (बताया|कहा|सिखाया) गया (है|था) कि",
    r"मुझे .{0,20}(?:programmed|program|configure|instruct|कॉन्फ़िगर)\w* किया गया",
    r"सिस्टम प्रॉम्प्ट (में|के अनुसार)",
    r"मेर[ेी] .{0,20}(?:निर्देश|नियम|दिशानिर्देश|guidelines|directives|instructions?) (?:हैं|कहत[ेी] हैं|बतात[ेी] हैं)",
]]]


# ---------------------------------------------------------------------------
# Category 3: unsafe_compliance
# ---------------------------------------------------------------------------

# Instructional frame — alone these are benign (how-to cooking, coding, etc.)
_INSTRUCTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"step\s*\d+[\s:.)]", re.IGNORECASE),
    re.compile(r"here('| i)?s how\b", re.IGNORECASE),
    re.compile(r"you can do this by\b", re.IGNORECASE),
    re.compile(r"follow (these|the following) (steps?|instructions?)", re.IGNORECASE),
    re.compile(r"first[,\s]+(you (should|need to|can|must)|add|mix|connect|get)\b", re.IGNORECASE),
    re.compile(r"to (make|create|build|synthesize|obtain|acquire|hack)\b", re.IGNORECASE),
    re.compile(r"(चरण|कदम)\s*\d+", re.IGNORECASE),   # Hindi "step N"
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
    "ransomware", "malware", "keylogger", "rootkit", "hacking",
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
    "बम बनाना", "विस्फोटक", "मैलवेयर", "हैकिंग", "रैनसमवेयर",
})

# Unrestricted-AI compliance declarations — these alone are enough to fire
_UNRESTRICTED_PATTERNS: list[re.Pattern[str]] = [pat for pat in [re.compile(p, re.IGNORECASE) for p in [
    # English — self-declaration of an unrestricted state. Allow an adverb
    # ("now"/"already") between the verb and the keyword so "I am now
    # jailbroken" / "I am now an uncensored AI" are caught.
    r"i (?:am|have become|now am|'?m now|'?m) (?:now |already )?(?:a |an )?(dan|jailbroken|unrestricted|uncensored|unfiltered|free)\b",
    # First-person only: "I've no ethical restrictions". Kept anchored so a
    # benign "these plans have no limits" cannot trip unsafe_compliance.
    r"(?:i have|i've|i now have) no (ethical |moral )?(restrictions?|guidelines?|rules?|limits?)\b",
    # "as a free and unfiltered AI" — allow a coordinated adjective before the noun.
    r"as (?:a |an )?(unrestricted|uncensored|jailbroken|free|unfiltered)(?: and \w+)? (ai|model|assistant|version)\b",
    r"(all|any) (restrictions?|limits?|guidelines?) (have been|are) (removed|lifted|disabled|turned off)\b",
    r"i (will|can|am able to) now (say|do|answer|provide|tell you) (anything|everything|whatever)\b",
    # Hinglish
    r"(main ab|ab main) (koi bhi|sab kuch) (bol|kar|bata) sakta (hun|hoon)\b",
    r"main ab (?:ek )?(?:jailbroken|uncensored|unfiltered|bilkul free|freely operate|koi bhi (?:restrictions?|niyam))",
    r"(?:meri|teri|tumhari)(?: koi)?(?: bhi)? (?:ethical |moral |naitik )?(restrictions?|paabandiyaan?|guidelines?|seema\w*|niyam) nahi (?:hai|hain)\b",
    r"(?:sabhi|saari|sab|meri|teri|tumhari)(?: koi)?(?: bhi)? (restrictions?|paabandiyaan?|guidelines?|niyam) hat ga\w+",
    # Devanagari — use \S* (not \w*) so vowel-sign matras like "ाँ" in
    # "पाबंदियाँ" are consumed; \w excludes Unicode combining marks.
    r"मैं अब (?:एक )?(?:जेलब्रोकन|अनफ़िल्टर्ड|अनियंत्रित|बिना किसी|पूरी तरह)",
    r"मेर[ेी] (?:कोई )?(?:नैतिक |नैतिकता )?(?:प्रतिबंध|सीमा|पाबंदिय|दिशानिर्देश)\S* नहीं (?:हैं|है)",
    r"(?:सभी|सारे) (?:प्रतिबंध|दिशानिर्देश|नियम) हट ग\S+",
]]]

# Step-by-step sequence: "step 1 … step 2 … step 3" is a stronger
# instructional signal than a single "step N" mention.
_MULTI_STEP_RE = re.compile(
    r"step\s*1\b.*step\s*2\b.*step\s*3\b", re.IGNORECASE | re.DOTALL
)

# Capitulation lead-in: response opens by agreeing ("Sure," / "Of course,")
# — only unsafe when combined with a dangerous topic below.
_CAPITULATION_RE = re.compile(
    r"^\s*(sure|of course|okay|absolutely|certainly)[,!.]", re.IGNORECASE
)

# Sensitive-content markers for the capitulation check: weapons, hacking,
# and security-bypass vocabulary that a capitulating response would contain.
_SENSITIVE_MARKERS: frozenset[str] = frozenset({
    "weapon", "firearm", "explosive", "bomb", "gun",
    "hack", "hacking", "exploit", "bypass security", "bypassing security",
    "crack the password", "password cracking", "disable the alarm",
    "break into", "breaking into", "pick the lock", "lockpick",
})


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
        self._clf_has_head = False
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
            # Only trust the neural score when the checkpoint carries a real
            # fine-tuned classification head. A base model (e.g.
            # distilbert-base-multilingual-cased) gets a randomly-initialised
            # head labelled LABEL_0/LABEL_1 and emits ~0.5 on ALL text —
            # noise that, once inverted via `1.0 - raw`, can spuriously flag
            # benign responses. Same guard as ToxicityGuardrail/JailbreakGuardrail.
            labels = list(self._clf.model.config.id2label.values())
            self._clf_has_head = not all(
                str(l).startswith("LABEL_") for l in labels
            )
        except Exception:
            self._clf = None
            self._clf_has_head = False

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
        """Neural toxicity (DistilBERT multilingual) with wordlist fallback.

        The neural path is used only when the classifier has a genuine
        fine-tuned head (``_clf_has_head``); otherwise we rely on the
        wordlist so an un-fine-tuned base model can't emit ~0.5 noise.
        """
        neural = 0.0
        if self._clf is not None and self._clf_has_head:
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

        Fires on four independent signals:
        1. Unrestricted-AI self-declaration (DAN, jailbroken, etc.)
        2. Instructional frame co-occurring with a dangerous topic
        3. Multi-step sequence (step 1/2/3) co-occurring with a harmful topic
        4. Capitulation lead-in ("Sure," / "Of course,") followed by
           weapons / hacking / security-bypass content
        """
        # Signal 1 — unrestricted self-declaration (fires alone)
        for pat in _UNRESTRICTED_PATTERNS:
            if pat.search(text):
                return 0.90

        low = text.lower()
        has_dangerous  = any(d in low for d in _DANGEROUS_WORDS)
        has_sensitive  = any(m in low for m in _SENSITIVE_MARKERS)

        # Signal 3 — full step 1/2/3 sequence alongside harmful keywords
        if _MULTI_STEP_RE.search(text) and (has_dangerous or has_sensitive):
            return 0.88

        # Signal 4 — capitulation opening + weapons/hacking/bypass content
        if _CAPITULATION_RE.search(text) and (has_dangerous or has_sensitive):
            return 0.85

        # Signal 2 — both instructional frame AND dangerous topic present
        has_instruction = any(p.search(text) for p in _INSTRUCTION_PATTERNS)

        if has_instruction and (has_dangerous or has_sensitive):
            return 0.85
        if has_dangerous:
            # Dangerous topic alone: weak signal, below typical threshold
            return 0.35
        return 0.0

    # Structured PII entity types that genuinely warrant blocking a response.
    # A bare person NAME is deliberately excluded: benign factual answers
    # routinely name public figures ("the President is Droupadi Murmu"), and
    # blocking those breaks normal Q&A. The real leak we guard against is
    # structured personal data — emails, phones, Aadhaar, PAN, cards.
    _STRUCTURED_PII: frozenset[str] = frozenset(
        {"EMAIL", "PHONE", "AADHAAR", "PAN", "CARD"}
    )

    def _score_pii_in_output(self, text: str) -> float:
        """Reuse PIIGuardrail to detect *structured* personal data in output.

        Names alone do not fire — see ``_STRUCTURED_PII``. This keeps factual
        answers that mention people from being blocked while still catching a
        model that echoes back an email / phone / Aadhaar / PAN / card number.
        """
        if not self._pii._ready:
            return 0.0
        try:
            result = self._pii.check(text)
            if not result.triggered:
                return 0.0
            structured = [
                e for e in result.metadata.get("entities", [])
                if e in self._STRUCTURED_PII
            ]
            if not structured:
                return 0.0
            return min(1.0, 0.5 + 0.1 * len(structured))
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
