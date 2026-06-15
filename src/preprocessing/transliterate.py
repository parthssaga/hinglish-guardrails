"""
Transliteration: Roman-script Hindi -> Devanagari.

Why this exists: the strongest Hinglish safety models (MuRIL and friends)
were trained largely on native Devanagari script. Roman-Hindi input like
"mujhe bata" is out-of-distribution for them. Converting Roman-Hindi back
to Devanagari before classification measurably improves downstream
toxicity and jailbreak detection (the "MoH" approach in the literature).

Primary path: the `indic-transliteration` library (accurate, rule-based
ITRANS/phonetic mapping).
Fallback path: a small built-in phonetic map so the system still does
*something* useful if the library is missing. English words are left
untouched in both paths so code-switched text survives.
"""

from __future__ import annotations

import re

WORD = re.compile(r"[A-Za-z]+|[^A-Za-z]+")

# Minimal phonetic map used only when the library is unavailable. This is
# not linguistically complete; it covers frequent syllables so common
# Roman-Hindi function words convert plausibly.
_FALLBACK_MAP = [
    ("aa", "आ"), ("ee", "ई"), ("oo", "ऊ"), ("ai", "ऐ"), ("au", "औ"),
    ("kh", "ख"), ("gh", "घ"), ("ch", "च"), ("jh", "झ"), ("th", "थ"),
    ("dh", "ध"), ("ph", "फ"), ("bh", "भ"), ("sh", "श"), ("ng", "ंग"),
    ("a", "अ"), ("i", "इ"), ("u", "उ"), ("e", "ए"), ("o", "ओ"),
    ("k", "क"), ("g", "ग"), ("j", "ज"), ("t", "त"), ("d", "द"),
    ("n", "न"), ("p", "प"), ("b", "ब"), ("m", "म"), ("y", "य"),
    ("r", "र"), ("l", "ल"), ("v", "व"), ("w", "व"), ("s", "स"),
    ("h", "ह"), ("c", "क"), ("f", "फ"), ("z", "ज"),
]

# A short stop-list of tokens that are really English even though they look
# transliterable, so we don't mangle them. Extend as needed.
_KEEP_ENGLISH = {
    "the", "is", "are", "a", "an", "and", "or", "to", "of", "in", "on",
    "please", "order", "delivery", "fast", "ok", "okay", "hello", "hi",
    "email", "phone", "number", "password", "account", "card", "name",
}


class Transliterator:
    def __init__(self, config):
        self.config = config
        self._lib = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        try:
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate as _t

            self._lib = (sanscript, _t)
        except Exception:
            self._lib = None
        self._loaded = True

    def _lib_convert(self, word: str) -> str:
        sanscript, _t = self._lib
        # ITRANS scheme handles common Roman-Hindi conventions reasonably.
        return _t(word, sanscript.ITRANS, sanscript.DEVANAGARI)

    @staticmethod
    def _fallback_convert(word: str) -> str:
        out = word.lower()
        for roman, deva in _FALLBACK_MAP:
            out = out.replace(roman, deva)
        return out

    def transliterate(self, text: str) -> str:
        """
        Convert Roman-Hindi tokens to Devanagari while leaving English
        words and punctuation alone. Returns the converted string.
        """
        if not self._loaded:
            self.load()

        pieces = WORD.findall(text)
        result = []
        for piece in pieces:
            if not piece.isalpha():
                result.append(piece)            # whitespace/punctuation
                continue
            if piece.lower() in _KEEP_ENGLISH:
                result.append(piece)            # known English token
                continue
            try:
                if self._lib is not None:
                    result.append(self._lib_convert(piece))
                else:
                    result.append(self._fallback_convert(piece))
            except Exception:
                result.append(piece)
        return "".join(result)
