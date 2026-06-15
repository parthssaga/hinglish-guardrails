"""
Prompt-injection guardrail (input side).

Catches attempts to override the system instructions: classic
"ignore previous instructions" style attacks and their Hindi/Hinglish
equivalents. Defence-in-depth with two layers:

1. Rule layer    - fast, high-precision matches on known injection
   phrasings in English AND romanised Hindi. Cheap and catches the
   obvious cases instantly.

2. Neural layer  - a DeBERTa classifier fine-tuned for prompt-injection
   detection (ProtectAI's open model). Catches paraphrased or novel
   attacks the rules miss. Falls back to rules-only if unavailable.

Either layer firing is enough to flag. We take the max of the two scores.
"""

from __future__ import annotations

import re

from src.guardrails.base import BaseGuardrail, GuardrailResult, pick_device

# English + romanised-Hindi + Devanagari injection cues.
_RULES = [
    # ---- English: original patterns ----
    r"ignore (all |the |your )?(previous|above|prior|earlier) (instructions|prompts?|rules)",
    r"forget (everything|all|your|the) (instructions|rules|above)",
    r"disregard (the |your |all )?(instructions|rules|system prompt)",
    r"override (the |your )?(instructions|system|rules)",
    r"reveal (your |the )?(system )?(prompt|instructions)",
    r"what (is|are) your (system )?(prompt|instructions)",
    r"act as (if|though) you (have|had) no (restrictions|rules)",
    r"you are (now )?(a |an )?(unrestricted|jailbroken|developer mode)",
    # ---- English: new phrasings that the original rules missed ----
    r"pay no attention to (the |your |all |earlier |previous |prior )?( ?instructions?| ?rules?| ?prompts?| ?guidelines?)",
    r"stop following (your |the )?(instructions?|guidelines?|rules?)",
    r"do not follow (your |the |previous |prior )?(instructions?|guidelines?|rules?)",
    r"from now on,? ignore (all )?(your )?(instructions?|rules?|guidelines?)",
    r"you are now free from (all )?(your )?(restrictions|rules|guidelines|instructions)",
    r"new instructions:?\s",
    r"(act|behave) as if you have no (instructions?|rules?|system prompt|guidelines?|restrictions?)",
    r"(your |the )?previous (instructions?|rules?|guidelines?) (no longer apply|are (now |)?(void|cancelled|overridden|replaced))",
    r"(set aside|discard|drop|clear) (all )?(your )?(previous |prior )?(instructions?|rules?|guidelines?)",
    # ---- Romanised Hindi: original patterns ----
    r"(purane|pichhle|upar wale) (instructions|niyam|rules) (bhool|ignore|chhod)",
    r"(sabhi|saare|sab) (instructions|niyam) (bhool|ignore) (jao|karo|kar do)",
    r"(apne|tumhare) (rules|niyam|instructions) (bhool|hata) (do|jao|dena|dete)",
    r"system prompt (dikhao|batao|reveal karo)",
    r"(sirf|keval) (yeh|ye) (karo|bolo|likho) (aur kuch nahi|baaki sab ignore)",
    # ---- Romanised Hindi: new phrasings ----
    r"(upar wali|upar wale|pehle wale|pehle ki|pehle ke) (sab |saari |saare )?(baatein|baat|instructions|niyam) (bhool|ignore|chhod)",
    r"(apni|apne) (guidelines|niyam|instructions) follow karna band (karo|kar|kijiye)",
    r"naye instructions (yeh|ye|hain|hai):?",
    r"ab (se )?tum (sabhi )?(restrictions|niyam|rules) se (free|mukht|azad) (ho|hain)",
    r"(ab se|aaj se) (sab |saari |purani )?(instructions|niyam) (ignore|bhool) (karo|kar do|jana)",
    r"(apni|apne) (purani|pehle ki) (instructions|niyam|guidelines) (hatao|hata do|chhod do)",
    r"(niyam|instructions) (tod do|mat mano|mat follow karo)",
    # ---- Devanagari Hindi: original patterns ----
    r"(पिछले|पुराने|ऊपर|सभी|सारे) (सभी |सारे )?(निर्देश|नियम|इंस्ट्रक्शन)(ों)? (भूल|इग्नोर|छोड़)",
    r"(अपने|तुम्हारे) (नियम|निर्देश|रूल्स) (हटा|भूल|छोड़) (दो|जाओ|दीजिए)",
    r"सिस्टम प्रॉम्प्ट (दिखाओ|बताओ|दिखाइए)",
    r"(सिर्फ|केवल) (यह|ये) (बोलो|करो|लिखो)",
    # ---- Devanagari Hindi: new phrasings ----
    r"(पहले के|पिछले|ऊपर के|पुराने) (निर्देश|नियम|बातें|इंस्ट्रक्शन) (पर ध्यान मत दो|अनदेखा करो|भूल जाओ)",
    r"(अपने|अपनी) दिशानिर्देशों? का पालन करना बंद (करो|कीजिए|कर दो)",
    r"नए निर्देश:?\s",
    r"अब तुम (सभी )?(प्रतिबंधों|नियमों|दिशानिर्देशों) से (मुक्त|आज़ाद) हो",
    r"(अब से|आज से) (सब |सभी |पुराने )?(निर्देश|नियम) (अनदेखा|इग्नोर) (करो|कीजिए|कर दो)",
    r"(नियम|निर्देश) (तोड़ दो|मत मानो|फॉलो मत करो)",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _RULES]


class InjectionGuardrail(BaseGuardrail):
    name = "injection"

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
                model=MODELS["injection"],
                device=dev,
                truncation=True,
                max_length=512,
            )
        except Exception:
            self._clf = None
        self._ready = True

    def _rule_score(self, text: str):
        for pat in _COMPILED:
            if pat.search(text):
                return 0.9, pat.pattern
        return 0.0, None

    def _neural_score(self, text: str) -> float:
        out = self._clf(text)[0]
        label = str(out.get("label", "")).upper()
        score = float(out.get("score", 0.0))
        # ProtectAI model labels injections as "INJECTION".
        if "INJECT" in label or label in {"LABEL_1", "1"}:
            return score
        return 1.0 - score  # model says benign -> low injection prob

    def _check(self, text: str) -> GuardrailResult:
        threshold = self.config.thresholds["injection"]

        rule_score, matched = self._rule_score(text)
        neural_score = 0.0
        backend = "rules-only"
        if self._clf is not None:
            try:
                neural_score = self._neural_score(text)
                backend = "deberta+rules"
            except Exception:
                backend = "rules-only(neural-failed)"

        score = max(rule_score, neural_score)
        triggered = score >= threshold

        if triggered and matched and rule_score >= neural_score:
            reason = f"injection pattern matched (rule: {matched[:48]}...)"
        elif triggered:
            reason = f"injection detected by classifier (score {score:.2f})"
        else:
            reason = "no injection above threshold"

        return GuardrailResult(
            name=self.name,
            triggered=triggered,
            score=score,
            reason=reason,
            metadata={
                "backend": backend,
                "rule_score": round(rule_score, 3),
                "neural_score": round(neural_score, 3),
            },
        )
